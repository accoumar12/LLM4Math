import logging
import os
import tempfile
import time
from pathlib import Path
from threading import Thread

import gradio as gr
import pandas as pd
import torch
from ctransformers import AutoModelForCausalLM as c_AutoModelForCausalLM

## For image generation
from diffusers import (
    AutoencoderTiny,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# To use the retrieval QA chain
from langchain.chains import RetrievalQA

# RAG Imports
# To load an LLM through Ollama
from langchain_community.embeddings import OllamaEmbeddings

# To load the LLM through Ollama
from langchain_community.llms import Ollama

# To create/ use the vector database
from langchain_community.vectorstores import Chroma
from llama_cpp import Llama
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from utils.constants import (
    BASE,
    CHECKPOINT,
    DEFAULT_GGUF_CACHE,
    DEFAULT_HF_CACHE,
    DEFAULT_MODEL,
    DEFAULT_PRECISION,
    GENERATION_CONFIG,
    MODELS_ID,
    ORIGINAL_MODEL,
    RAG_DATABASE,
    RAG_FOLDER_PATH,
    REPO,
    TAESD_MODEL,
    THEME,
)
from utils.RAG_utils import build_prompt, process_llm_response

# ------
# logger
logger = logging.getLogger(__name__)


# ------------
# extra option
def save_option(max_lengh: int, answer_prefix: str, end_token: str):
    global Max_n, prefix, endfix
    Max_n = max_lengh
    prefix = answer_prefix
    endfix = end_token


# ------------------------
# loading models functions
def load_model(model_name: str, precision: str, gpu_layer: int, Rag_db: str, mode: str):
    """
    Root function for loading models.
    Dispatches the loading to the right function based on the chosen model.

    Parameters:
        model_name: name of the model to load.
        precision_chosen: precision to load the model in (quantization). Default value or chosen by the user.
        gpu_layers: number of layers to off-load on GPU. Default value or chosen by the user.

    """

    global Mode
    Mode = mode

    if mode == "RAG":
        load_RAG(model_name, Rag_db)
        return

    if mode == "gguf":
        useLlama = "mixtral" in model_name
        load_gguf_model(model_name, gpu_layer, useLlama=useLlama)
    else:
        load_hf_model(model_name, precision)


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
        if useLlama:
            model = Llama(model_path, n_gpu_layers=gpu_layers, verbose=False)
        else:
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


def load_RAG(model_name: str, Rag_db: str):

    global model, tokenizer, cache_model_name
    logger.info(f"loading model with RAG {model_name}")
    tokenizer = None
    cache_model_name = None
    # name of the vector database stored on the disk
    # persist_directory = 'vdb_gsm8k'
    persist_directory = RAG_FOLDER_PATH + RAG_DATABASE[Rag_db]

    # ollama embeddings used to vectorize the data
    embeddings_open = OllamaEmbeddings(model="phi")

    # load the vector database from disk
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings_open,
    )
    # instanciate a retriever to fetch the most similar vectors in the vdb
    retriever = vectordb.as_retriever()

    # -----------------
    # LLM

    llm_open = Ollama(
        model=model_name,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    # ---------------------
    # Enhance the Q&A chain:
    model = RetrievalQA.from_chain_type(
        llm=llm_open,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
        chain_type_kwargs={"prompt": build_prompt("template_with_context_1")},
    )
    gr.Info("Le modèle " + model_name + " à été correctement chargé")
    logger.info("Model loaded successfully :)")


# -----------------------
# LLM generation functions
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

    global model, tokenizer, Mode
    if model is None:
        yield "No LLM has been loaded yet :( Please load a model first."

    if Mode == "RAG":
        yield from predict_RAG(message)
    elif Mode == "gguf":
        useLlama = "mixtral" in model_name
        yield from predict_gguf(message, history, model, no_log, useLlama)
    else:
        yield from predict_hf(message, history, model, tokenizer, no_log)


def predict_gguf(
    message: str,
    history: list[list[str, str]],
    model: c_AutoModelForCausalLM,
    no_log: bool = False,
    useLlama: bool = False,
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
        else:
            tokenStream = model(messages, stream=True)
        for new_token in tokenStream:
            # looking for the end of the answer and the beginning of the next one
            if (
                "human>:" in partial_message
                or "bot>:" in partial_message
                or endfix in partial_message
            ):
                break
            else:
                if useLlama:
                    partial_message += new_token["choices"][0]["text"]
                else:
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
        if new_token != "<" and (endfix not in partial_message):
            partial_message += new_token
            yield partial_message
    if not no_log:
        logger.info("Answer generated :)")


def test_a_file(file_name: str, progress=gr.Progress(track_tqdm=True)):

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

        the_answer = list(predict(prompt, [], cache_model_name, no_log=True))[-1]
        answers.append(the_answer)

    ###### Saving prompts:answers into a .csv ######
    print(f"Saving the answers")
    assert len(prompts) == len(
        answers
    ), "The number of prompts and answers is not the same"
    prompts_df["answer"] = answers
    ANSWERS_FOLDER = os.path.dirname(file_name)
    answers_file_name = f"answers_{cache_model_name}_{os.path.basename(file_name)}"
    prompts_df.to_csv(ANSWERS_FOLDER + "/" + answers_file_name, index=False)
    return ANSWERS_FOLDER + "/" + answers_file_name


def predict_RAG(msg: str):
    global model
    llm_response = model.invoke(msg)
    yield process_llm_response(llm_response)


# ----------------
# Image generation

# Code copy-pasted from https://huggingface.co/spaces/radames/Real-Time-Text-to-Image-SDXL-Lightning/blob/main/app.py
# And simplified


USE_TAESD = os.environ.get("USE_TAESD", "0") == "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_device = device
torch_dtype = torch.float16


unet_config = UNet2DConditionModel.load_config(BASE, subfolder="unet")
unet = UNet2DConditionModel.from_config(unet_config).to("cuda", torch.float16)

unet.load_state_dict(load_file(hf_hub_download(REPO, CHECKPOINT), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE, unet=unet, torch_dtype=torch.float16, variant="fp16", safety_checker=False
).to("cuda")


if USE_TAESD:
    pipe.vae = AutoencoderTiny.from_pretrained(
        TAESD_MODEL, torch_dtype=torch_dtype, use_safetensors=True
    ).to(device)


# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)
pipe.set_progress_bar_config(disable=True)


def predict_image(prompt, seed=1231231):
    generator = torch.manual_seed(seed)
    last_time = time.time()
    results = pipe(
        prompt=prompt,
        generator=generator,
        num_inference_steps=2,
        guidance_scale=0.0,
        # width=768,
        # height=768,
        output_type="pil",
    )
    print(f"Pipe took {time.time() - last_time} seconds")

    image = results.images[0]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
        image.save(tmpfile, "JPEG", quality=80, optimize=True, progressive=True)
        return Path(tmpfile.name)


def set_generation_theme(chosen_theme: str | None):
    global THEME
    THEME = chosen_theme
    return
