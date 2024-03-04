import argparse
import logging

import gradio as gr
from utils.constants import (
    DEFAULT_MODEL,
    DEFAULT_PRECISION,
    MODELS_ID,
    OLLAMA_MODEL,
    PRECISIONS,
    RAG_DATABASE,
    THEME,
    OLLAMA_default,
    available_models_paths,
)
from utils.utils import (
    load_model,
    predict,
    predict_image,
    save_option,
    set_generation_theme,
    test_a_file,
)

# ---------------------------------------------------------------------------
# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def changedropdownvalue(mode: str):
    if mode == "hugging face":
        model_name = gr.Dropdown(choices=list(MODELS_ID.keys()), value=DEFAULT_MODEL)
        return model_name
    elif mode == "gguf":
        model_name = gr.Dropdown(choices=list(available_models_paths.keys()))

        return model_name
    else:
        model_name = gr.Dropdown(choices=OLLAMA_MODEL, value=OLLAMA_default)
        return model_name


# ---------------------------------------------------------------------------
# Gradio app
def gradio_app():
    global model, tokenizer
    model, tokenizer = None, None
    with gr.Blocks("soft") as iface:
        # create an option menu to choose the model to load
        with gr.Accordion("Model choice and options"):
            with gr.Tab("model option"):
                Mode_radio = gr.Radio(
                    choices=["hugging face", "gguf", "RAG"],
                    value="hugging face",
                    label="What type of model to use",
                )
                model_name_chosen = gr.Dropdown(
                    choices=MODELS_ID.keys(),
                    value=DEFAULT_MODEL,
                    label="Choose a model",
                )
                precision = gr.Dropdown(
                    choices=PRECISIONS,
                    value=DEFAULT_PRECISION,
                    label="Choose a quantization precision",
                    info="Hugging face-models only",
                )
                gpu_layer = gr.Slider(
                    minimum=0.0,
                    maximum=50.0,
                    value=10.0,
                    step=1,
                    label="Choose the #layers to off-load on GPU",
                    info="GGUF-models only",
                )
                RAG_database = gr.Dropdown(
                    choices=list(RAG_DATABASE.keys()),
                    label="what database to use",
                    info="RAG-models only",
                )
                Mode_radio.change(
                    changedropdownvalue, Mode_radio, outputs=model_name_chosen
                )
                b1 = gr.Button("Load model")
                b1.click(
                    load_model,
                    inputs=[
                        model_name_chosen,
                        precision,
                        gpu_layer,
                        RAG_database,
                        Mode_radio,
                    ],
                )
            with gr.Tab("generation option"):
                max_length = gr.Slider(
                    minimum=0,
                    maximum=2048,
                    value=500,
                    step=1,
                    label="How many token can the model generate max (not working yet)",
                )
                prefix_text = gr.Text(
                    label="what does the answer should start with (can force an AI to follow a patern)",
                    placeholder="###CONTEXT###",
                )
                end_text = gr.Text(
                    placeholder="</s>",
                    label="what should be consider the end of the bot generation (can cause the AI to stop generating sonner)",
                )
                b2 = gr.Button("Save options")
                b2.click(save_option, [max_length, prefix_text, end_text])
        # Create a Gradio Chatbot Interface
        with gr.Tab("Teacher Assistant"):
            gr.Textbox(
                # value=set_generation_theme,
                placeholder=THEME,
                label="Choose your theme",
            )

            gr.ChatInterface(
                predict,
                additional_inputs=[model_name_chosen],
            )

        with gr.Tab("Test a file"):
            gr.Interface(
                test_a_file,
                [
                    gr.File(
                        file_count="single",
                        file_types=[".csv"],
                        label="Fichier a tester au format csv",
                    )
                ],
                "file",
            )
        with gr.Tab("Image generation"):
            with gr.Row():
                prompt = gr.Textbox(
                    placeholder="Insert your prompt here:", scale=5, container=False
                )
                generate_bt = gr.Button("Generate", scale=1)

            image = gr.Image(type="filepath")
        with gr.Accordion("Advanced options", open=False):
            seed = gr.Slider(
                randomize=True, minimum=0, maximum=12013012031030, label="Seed", step=1
            )
        inputs = [prompt, seed]
        outputs = [image]
        generate_bt.click(
            fn=predict_image, inputs=inputs, outputs=outputs, show_progress=False
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
