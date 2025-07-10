import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
Models_dir = os.path.join(PROJECT_DIR, 'Models')

caption_model_paths = {
    "WDTagger": f"{Models_dir}/SmilingWolf_wd-swinv2-tagger-v3",
    "JoyTag": f"{Models_dir}/fancyfeast_joytag",
    "LlamaJoycaptionAlphaTwo": f"{Models_dir}/fancyfeast_llama-joycaption-alpha-two-hf-llava",
    "LLaVANeXT": f"{Models_dir}/lmms-lab_llama3-llava-next-8b",
}