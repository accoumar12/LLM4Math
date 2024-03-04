import os

MODELS_ID = {
    "Yi-34b_chat": "01-ai/Yi-34b-Chat",
    ### Mistral-based ###
    "mixtral-8x7b_instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral-7b_instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral-7b_orca": "Open-Orca/Mistral-7B-OpenOrca",
    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "vigostral-7b": "bofenghuang/vigostral-7b-chat",
    "mistral-7b_original": "mistralai/Mistral-7B-v0.1",
    ### Llama-based ###
    "llama2-chat-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-chat-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-chat-70b": "meta-llama/Llama-2-70b-chat-hf",
    # wizard series
    "wizard-7b_math": "WizardLM/WizardMath-7B-V1.0",
    "wizard-13b_math": "WizardLM/WizardMath-13B-V1.0",
    "wizard-15b_coder": "WizardLM/WizardCoder-15B-V1.0",
    "wizard-34b_coder": "WizardLM/WizardCoder-Python-34B-V1.0",
    # french focused
    "vigogne-7b": "bofenghuang/vigogne-2-7b-chat",
    "vigogne-7b_instruct": "bofenghuang/vigogne-2-7b-instruct",  # ok pour les licenses
    # bigscience bloom (7b)
    "bloom-7b": "bigscience/bloom-7b1",
    # GPT-neo
    "gptNeo_original": "EleutherAI/gpt-neo-2.7B",
    # GPT-J
    "gptJ_original": "EleutherAI/gpt-j-6B",
    # Math-oriented models
    "deepseek": "deepseek-ai/deepseek-math-7b-instruct",
}

OLLAMA_MODEL = ["mixtral", "llama2"]
OLLAMA_default = "mixtral"

RAG_FOLDER_PATH = "../vdb/"
RAG_DATABASE = {
    "prof en Poche": "vdb_profEnPoche_examples",
    "GSM8k": "vdb_gsm8k",
    "GSM8k no answer": "vdb_gsm8k_no_answ",
}

ORIGINAL_MODEL = {
    "llama-2-13b-chat.Q4_K_M.gguf": "llama2-chat-13b",
    "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf": "mixtral-8x7b_instruct",
}


IS_CHAT = {
    model_name: ("chat" in full_model_id.lower())
    or ("instruct" in full_model_id.lower())
    for model_name, full_model_id in MODELS_ID.items()
}


# locally stored models
def load_available_models_paths(path: str):
    """
    Load the locally available models' paths

    Parameters:
        path: path to the directory containing all models already available locally.
    """
    models_paths = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if "gguf" in file or "hf" in file:
                file_path = os.path.join(root, file)
                models_paths[file] = file_path
    return models_paths


# locally stored models + interesting models stored on the hub
# MODELS_PATH = "/home/pie2023/dataSSD/models/"
MODELS_PATH = "/home/pie2023/data/models/"

available_models_paths = load_available_models_paths(MODELS_PATH)
MODEL_NAMES = list(available_models_paths.keys()) + list(MODELS_ID.keys())

DEFAULT_HF_CACHE = MODELS_PATH + "hf_models/"
DEFAULT_GGUF_CACHE = MODELS_PATH + "gguf_models/"

DEFAULT_MODEL = "llama2-chat-7b"

PRECISIONS = ["4", "8", "16", "32"]
DEFAULT_PRECISION = "4"


# hf generation config
GENERATION_CONFIG = dict(
    max_new_tokens=1024,
    do_sample=True,
    top_p=0.95,
    top_k=1000,
    temperature=1.0,
    num_beams=1,
)

# For image generation

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
REPO = "ByteDance/SDXL-Lightning"
# 1-step
CHECKPOINT = "sdxl_lightning_2step_unet.safetensors"
TAESD_MODEL = "madebyollin/taesdxl"

THEME = "dinosaurs"