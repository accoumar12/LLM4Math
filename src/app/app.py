import gradio as gr

import argparse
import logging

from utils.utils import load_model, predict, test_a_file, save_option
from utils.constants import MODEL_NAMES, DEFAULT_MODEL, PRECISIONS, DEFAULT_PRECISION, OLLAMA_MODEL, OLLAMA_default


# ---------------------------------------------------------------------------
# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def changedropdownvalue(use_rag : bool):
    if use_rag:
        return gr.Dropdown(choices=OLLAMA_MODEL, value=OLLAMA_default)
    else :
        return gr.Dropdown(choices=MODEL_NAMES, value=DEFAULT_MODEL)

# ---------------------------------------------------------------------------
# Gradio app
def gradio_app():
    global model, tokenizer
    model, tokenizer = None, None
    with gr.Blocks("soft") as iface:
        # create an option menu to choose the model to load
        with gr.Accordion("Model choice and options"):
            with gr.Tab("model option"):
                model_name_chosen = gr.Dropdown(
                    choices=MODEL_NAMES,
                    value=DEFAULT_MODEL,
                    label="Choose a model",
                )
                precision_chosen = gr.Dropdown(
                    choices=PRECISIONS,
                    value=DEFAULT_PRECISION,
                    label="Choose a quantization precision",
                    info="HF-models only",
                )
                gpu_layers_chosen = gr.Slider(
                    minimum=0.0,
                    maximum=50.0,
                    value=10.0,
                    step=1,
                    label="Choose the #layers to off-load on GPU",
                    info="GGUF-models only",
                )
                RAG_button = gr.Checkbox(
                    label="should you use RAG to retreive exemple"
                )
                RAG_button.change(changedropdownvalue, RAG_button, model_name_chosen)
                b1 = gr.Button("Load model")
                b1.click(
                    load_model,
                    inputs=[model_name_chosen, precision_chosen, gpu_layers_chosen, RAG_button],
                )
            with gr.Tab("generation option"):
                max_length = gr.Slider(
                    minimum = 0,
                    maximum=2048,
                    value=500,
                    step=1,
                    label="How many token can the model generate max (not working yet)"
                )
                prefix_text = gr.Text(
                    label="what does the answer should start with (can force an AI to follow a patern)",
                    placeholder="###CONTEXT###"
                )
                end_text = gr.Text(
                    placeholder="</s>",
                    label="what should be consider the end of the bot generation (can cause the AI to stop generating sonner)"
                )
                b2 = gr.Button("Save options")
                b2.click(save_option, [max_length, prefix_text, end_text])
        # Create a Gradio Chatbot Interface
        with gr.Tab("Teacher Assistant"):
            gr.ChatInterface(
                predict,
                additional_inputs=[model_name_chosen],
            )

        with gr.Tab("Test a file"):
            gr.Interface(
                test_a_file,
                [gr.File(file_count='single', file_types=[".csv"], label="Fichier a tester au format csv")],
                "file",
            )

    # Launch Gradio Interface
    logger.info("Launching Gradio Interface...")
    iface.launch()


# ---------------------------------------------------------------------------
# CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web-app to interact with a LLM")
    save_option(500, "", "</s>")
    gradio_app()
