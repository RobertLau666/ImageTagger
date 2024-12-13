from .llava.model.builder import load_pretrained_model
from .llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from .llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

class LLaVANeXT():
    def __init__(self, model_dir):
        self.model_dir = model_dir
        model_name = "llava_llama3"
        self.device = "cuda"
        device_map = "auto"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(self.model_dir, None, model_name, device_map=device_map, attn_implementation=None) # Add any other thing you want to pass in llava_model_args
        # , attn_implementation=None

        self.model.eval()
        self.model.tie_weights()

    def __call__(self, img_path):
        # url = "assert/25.png"
        image = Image.open(img_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size]

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        text = text_outputs[0].replace('\n', '')

        return text


if __name__ == "__main__":
    llavanext = LLaVANeXT("/maindata/data/shared/public/chenyu.liu/pulid_models/llama3-llava-next-8b")
    img_path = "assert/25.png"
    text = llavanext(img_path)
    print("text: ", text)

    # llavanext = LLaVANeXT("/maindata/data/shared/public/chenyu.liu/pulid_models/llama3-llava-next-8b")
    # img_dir = "/maindata/data/shared/public/chenyu.liu/Datasets/MGC/00000_jpg_txt"
    # import os
    # from tqdm import tqdm
    # files = os.listdir(img_dir)
    # img_names = []
    # for img_name in files:
    #     if img_name.endswith('.jpg'):
    #         img_names.append(img_name)
    # print("len(img_names): ", len(img_names))
    # for img_name in tqdm(img_names):
    #     img_path = os.path.join(img_dir, img_name)
    #     text = llavanext(img_path)
    #     print("text: ", text)
    #     with open(img_path.replace('.jpg','.txt'), "w", encoding='utf-8') as file:
    #         file.write(text)