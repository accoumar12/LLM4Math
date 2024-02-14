import logging
import gradio as gr

import pandas as pd
import os
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

from ctransformers import AutoModelForCausalLM as c_AutoModelForCausalLM
from llama_cpp import Llama

from utils.constants import (
    MODELS_ID,
    DEFAULT_MODEL,
    DEFAULT_PRECISION,
    DEFAULT_HF_CACHE,
    DEFAULT_GGUF_CACHE,
    GENERATION_CONFIG,
    ORIGINAL_MODEL,
)


# -------------------------------------------
# logger
logger = logging.getLogger(__name__)


# -------------------------------------------
# extra option
def save_option(
        max_lengh : int,
        answer_prefix : str,
        end_token : str
):
    global Max_n, prefix, endfix
    Max_n = max_lengh
    prefix = answer_prefix
    endfix = end_token


# -------------------------------------------
# loading models functions
def load_model(
    model_name: str,
    precision_chosen: str,
    gpu_layers: int = 0,
):
    """
    Root function for loading models.
    Dispatches the loading to the right function based on the chosen model.

    Parameters:
        model_name: name of the model to load.
        precision_chosen: precision to load the model in (quantization). Default value or chosen by the user.
        gpu_layers: number of layers to off-load on GPU. Default value or chosen by the user.

    """

    useLlama = "mixtral" in model_name 
    if "gguf" in model_name:
        load_gguf_model(model_name, gpu_layers, useLlama=useLlama)
    else:
        load_hf_model(model_name, precision_chosen)


def load_gguf_model(
    model_name: str,
    gpu_layers: int = 0,
    cache_dir: str = DEFAULT_GGUF_CACHE,
    useLlama: bool = False,
):
    """
    Load a GGUF model through the ctransformers library.

    Parameters:
        model_name: name of the model file to load.
        gpu_layers: number of layers to off-load on GPU. Default value or chosen by the user.
        cache_dir: path to the cache directory. Default value.
    """

    global model, tokenizer, cache_model_name
    model_path = cache_dir + model_name
    model_type = "llama" if "llama" in model_path else "mistral"

    logger.info(f"Loading {model_type}-type model from : \n {model_path}")

    try:
        if useLlama :
            model = Llama(
                model_path, 
                n_gpu_layers= gpu_layers,
                verbose=False
            )
        else :
            model = c_AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type=model_type,
                model_file=model_name,
                gpu_layers=gpu_layers,
            )
        if model_name in ORIGINAL_MODEL:
            model_id = MODELS_ID[ORIGINAL_MODEL[model_name]]
        else:
            logger.info("Impossible de retrouver le modèle original")
            model_id = (
                "meta-llama/Llama-2-7b-chat-hf"
                if "llama" in model_path
                else "mistralai/Mistral-7B-Instruct-v0.1"
            )
            logger.info(f"utilisation de {model_id} par defaut pour le tokeniser")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )

        gr.Info("Le modèle " + model_name + " à été correctement chargé")
        logger.info("Model loaded successfully :)")
        cache_model_name = model_name

    except Exception as e:
        gr.Warning("Erreur : " + e)
        logger.info(f"Error loading model through ctransformers: \n{e} ")
        model, tokenizer, cache_model_name = None, None, None


def load_hf_model(
    model_name: str,
    precision: str = DEFAULT_PRECISION,
    cache_dir: str = DEFAULT_HF_CACHE,
):
    """
    Loads a model and its tokenizer from the hugging-face hub through the transformers library.

    Parameters:
        model_name: name of the model to load.
        precision: precision to load the model in (quantization). Default value or chosen by the user.
        cache_dir: path to the cache directory. Default value.
    """

    global model, tokenizer, cache_model_name
    model_id = MODELS_ID[model_name]

    logger.info(f"Loading {model_name} at {model_id}")

    try:
        if precision == "4":
            logger.info("Loading model in 4 bits")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=nf4_config,
                device_map="auto",  # accelerate dispatches layers to ram, vram or disk
                cache_dir=cache_dir,
            )
        elif precision == "8":
            logger.info("Loading model in 8 bits")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
            int8_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=int8_config,
                device_map="auto",
                cache_dir=cache_dir,
            )
        elif precision == "16":
            logger.info("Loading model in 16 bits")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # f16_config = BitsAndBytesConfig(
            #    load_in_16bit=True,
            #    bnb_16bit_compute_dtype=torch.bfloat16,
            # )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                # quantization_config=f16_config,
                device_map="auto",
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            )
        else:
            logger.info("Loading model in 32 bits")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                cache_dir=cache_dir,
            )
        cache_model_name = model_name
        gr.Info("Le modèle " + model_name + " à été correctement chargé")
        logger.info("Model loaded successfully :)")
    except Exception as e:
        gr.Warning("Erreur : " + e)
        logger.info(f"Error loading model through transformers: \n{e} ")
        model, tokenizer, cache_model_name = None, None, None


# -------------------------------------------
# generation functions
def predict(
    message: str,
    history: list[list[str, str]],
    model_name: str,
    no_log: bool = False,
):
    """
    Root function for generating answers.
    Dispatches the generation to the right function based on the chosen model.

    Parameters:
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
        model_name: name of the model to use to generate the response. Default value or chosen by the user.
    """

    global model, tokenizer

    useLlama = "mixtral" in model_name
    if model is None:
        yield "No LLM has been loaded yet :( Please load a model first."
    elif "gguf" in model_name:
        yield from predict_gguf(message, history, model, no_log, useLlama)
    else:
        yield from predict_hf(message, history, model, tokenizer, no_log)


def predict_gguf(
    message: str,
    history: list[list[str, str]],
    model: c_AutoModelForCausalLM,
    no_log: bool = False,
    useLlama : bool = False
):
    """
    Generates an answer using GGUF's ctransformers.
    Responds to the user's message based on the dialogue history.

    Parameters:
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
        model: model to use to generate the response.
    """

    global prefix, endfix

    if model is None:
        return "No LLM has been loaded yet ..."

    else:
        messages = []
        for item in history:
            messages += [
                {"role": "user", "content": item[0]},
                {"role": "assistant", "content": item[1]},
            ]
        messages.append({"role": "user", "content": message})
        messages = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        messages += prefix
        if not no_log:
            print(messages)

        if not no_log:
            logger.info("Started generating text ...")
        partial_message = prefix
        if useLlama:
            tokenStream = model(messages, stream=True, max_tokens=32000, stop=[endfix])
        else :
            tokenStream = model(messages, stream=True)
        for new_token in tokenStream:
            # looking for the end of the answer and the beginning of the next one
            if "human>:" in partial_message or "bot>:" in partial_message or endfix in partial_message:
                break
            else:
                if useLlama:
                    partial_message += new_token["choices"][0]["text"]
                else :
                    partial_message += new_token
                yield partial_message
        if not no_log:
            logger.info("Answer generated :)")


def predict_hf(
    message,
    history,
    model,
    tokenizer,
    no_log: bool = False,
):
    """
    Generates an answer using Hugging-Face's transformers.
    Responds to the user's message based on the dialogue history.

    Parameters:
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
        model: model to use to generate the response.
        tokenizer: tokenizer to use to generate the response.
    """
    messages = []
    for item in history:
        messages += [
            {"role": "user", "content": item[0]},
            {"role": "assistant", "content": item[1]},
        ]
    messages.append({"role": "user", "content": message})
    if not no_log:
            print(messages)
    input_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_tokens += prefix

    input_tokens = tokenizer(input_tokens, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=None,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generate_kwargs = dict(
        input_tokens,
        streamer=streamer,
        **GENERATION_CONFIG,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)

    if not no_log:
            logger.info("Started generating text ...")
    t.start()
    partial_message = prefix
    for new_token in streamer:
        if new_token != "<":
            partial_message += new_token
            yield partial_message
    if not no_log:
            logger.info("Answer generated :)")


def test_a_file(
        file_name : str,
        progress = gr.Progress(track_tqdm=True)
        ):

    global model, tokenizer, cache_model_name

    ###### Fetching the prompts ######
    PROMPTS_FOLDER = "./src/generation/prompts/"
    try:
        prompts_df = pd.read_csv(file_name)
    except FileNotFoundError:
        raise Exception(
            f"The prompts file {file_name} was not found in the folder {PROMPTS_FOLDER} :("
        )
    prompt_col_name = prompts_df.columns[0]
    prompts = prompts_df[prompt_col_name].to_list()

    ###### Generating ######
    answers = []
    print(f"Generating answers")
    for prompt in progress.tqdm(prompts, desc="Prompt"):
        prompt = prompt.strip()

        the_answer = list(predict(prompt, [], cache_model_name, no_log= True))[-1]
        answers.append(the_answer)


    ###### Saving prompts:answers into a .csv ######
    print(f"Saving the answers")
    assert len(prompts) == len(answers), "The number of prompts and answers is not the same"
    prompts_df["answer"] = answers
    ANSWERS_FOLDER = os.path.dirname(file_name)
    answers_file_name = f"answers_{cache_model_name}_{os.path.basename(file_name)}"
    prompts_df.to_csv(ANSWERS_FOLDER + "/" + answers_file_name, index=False)
    return ANSWERS_FOLDER + "/" + answers_file_name

def multiply_two_numbers_question(a, b):
    result = a * b
    return f"The result of multiplying {a} and {b} is {result}."

def create_contextualized_prompt(context: str, question: str):
    return(f"###CONTEXT###\n{context}\n###QUESTION###\n{question}\n###END###")