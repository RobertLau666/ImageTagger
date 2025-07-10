import os
from pathlib import Path
import torch
import torch.amp
import torch.amp.autocast_mode
from torchvision import transforms
import torchvision.transforms.functional as TVF
from transformers import pipeline, AutoModel, AutoTokenizer, LlavaForConditionalGeneration
from PIL import Image
import numpy as np
import onnxruntime as rt
import pandas as pd
from .JoyTag_Models import VisionModel
import argparse
import copy

from .LLaVA_NeXT.llava.model.builder import load_pretrained_model
from .LLaVA_NeXT.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from .LLaVA_NeXT.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from .LLaVA_NeXT.llava.conversation import conv_templates, SeparatorStyle


class WDTagger:
    def __init__(self, model_dir):
        self.model_target_size = None
        self.last_loaded_repo = None

        # self.TITLE = "WaifuDiffusion Tagger"
        # self.DESCRIPTION = """
        # Demo for the WaifuDiffusion tagger models
        # Example image by [ほし☆☆☆](https://www.pixiv.net/en/users/43565085)
        # """

        self.model_dir = model_dir
        # # Dataset v3 series of models:
        # self.SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
        # self.CONV_MODEL_DSV3_REPO = "SmilingWolf/wd-convnext-tagger-v3"
        # self.VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"
        # self.VIT_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
        # self.EVA02_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

        # # Dataset v2 series of models:
        # self.MOAT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
        # self.SWIN_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
        # self.CONV_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
        # self.CONV2_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
        # self.VIT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

        # Files to download from the repos
        self.MODEL_FILENAME = "model.onnx"
        self.LABEL_FILENAME = "selected_tags.csv"

        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
        self.kaomojis = [
            "0_0",
            "(o)_(o)",
            "+_+",
            "+_-",
            "._.",
            "<o>_<o>",
            "<|>_<|>",
            "=_=",
            ">_<",
            "3_3",
            "6_9",
            ">_o",
            "@_@",
            "^_^",
            "o_o",
            "u_u",
            "x_x",
            "|_|",
            "||_||",
        ]


    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--score-slider-step", type=float, default=0.05)
        parser.add_argument("--score-general-threshold", type=float, default=0.35)
        parser.add_argument("--score-character-threshold", type=float, default=0.85)
        parser.add_argument("--share", action="store_true")
        return parser.parse_args()


    def load_labels(self, dataframe) -> list[str]:
        name_series = dataframe["name"]
        name_series = name_series.map(
            lambda x: x.replace("_", " ") if x not in self.kaomojis else x
        )
        tag_names = name_series.tolist()

        rating_indexes = list(np.where(dataframe["category"] == 9)[0])
        general_indexes = list(np.where(dataframe["category"] == 0)[0])
        character_indexes = list(np.where(dataframe["category"] == 4)[0])
        return tag_names, rating_indexes, general_indexes, character_indexes


    def mcut_threshold(self, probs):
        """
        Maximum Cut Thresholding (MCut)
        Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
        for Multi-label Classification. In 11th International Symposium, IDA 2012
        (pp. 172-183).
        """
        sorted_probs = probs[probs.argsort()[::-1]]
        difs = sorted_probs[:-1] - sorted_probs[1:]
        t = difs.argmax()
        thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
        return thresh


    def download_model(self, model_repo):
        # csv_path = huggingface_hub.hf_hub_download(
        #     model_repo,
        #     self.LABEL_FILENAME,
        # )
        # model_path = huggingface_hub.hf_hub_download(
        #     model_repo,
        #     self.MODEL_FILENAME,
        # )

        # 指定本地的 CSV 和模型文件路径
        csv_path = os.path.join(self.model_dir, self.LABEL_FILENAME)
        model_path = os.path.join(self.model_dir, self.MODEL_FILENAME)
        return csv_path, model_path

    def load_model(self, model_repo):
        if model_repo == self.last_loaded_repo:
            return
        csv_path, model_path = self.download_model(model_repo)
        tags_df = pd.read_csv(csv_path)
        sep_tags = self.load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        model = rt.InferenceSession(model_path)
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model

    def prepare_image(self, image):
        target_size = self.model_target_size

        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def __call__(
        self,
        image_path,
        model_repo="",
        general_thresh=0.35,
        general_mcut_enabled=False,
        character_thresh=0.85,
        character_mcut_enabled=False,
    ):
        self.load_model(model_repo)

        image = Image.open(image_path)
        image = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = self.mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = self.mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = (
            ", ".join(sorted_general_strings).replace("(", "\(").replace(")", "\)")
        )

        # return sorted_general_strings, rating, character_res, general_res
        return sorted_general_strings


class JoyTag():
    def __init__(self, model_dir):
        # Demo for the JoyTag model: https://huggingface.co/fancyfeast/joytag
        # self.MODEL_REPO = "fancyfeast/joytag"
        self.THRESHOLD = 0.4
        self.model_dir = model_dir
        self.model = VisionModel.load_model(self.model_dir)
        self.model.eval()
        with open(Path(self.model_dir) / 'top_tags.txt', 'r') as f:
            self.top_tags = [line.strip() for line in f.readlines() if line.strip()]

    def prepare_image(self, image: Image.Image, target_size: int) -> torch.Tensor:
        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2
        padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))
        # Resize image
        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
        # Convert to tensor
        image_tensor = TVF.pil_to_tensor(padded_image) / 255.0
        # Normalize
        image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        return image_tensor

    @torch.no_grad()
    def predict(self, image: Image.Image):
        image_tensor = self.prepare_image(image, self.model.image_size)
        batch = {
            'image': image_tensor.unsqueeze(0),
        }
        with torch.amp.autocast_mode.autocast('cpu', enabled=True):
            preds = self.model(batch)
            tag_preds = preds['tags'].sigmoid().cpu()
        scores = {self.top_tags[i]: tag_preds[0][i] for i in range(len(self.top_tags))}
        predicted_tags = [tag for tag, score in scores.items() if score > self.THRESHOLD]
        tag_string = ', '.join(predicted_tags)
        return tag_string, scores

    def __call__(self, img_path):
        img = Image.open(img_path)
        tag_string, scores = self.predict(img)
        return tag_string


class LlamaJoycaptionAlphaTwo():
    def __init__(self, model_dir):
        self.PROMPT = "Write a very long descriptive caption for this image in a formal tone."
        self.model_dir = model_dir

        # Load JoyCaption
        # bfloat16 is the native dtype of the LLM used in JoyCaption (Llama 3.1)
        # device_map=0 loads the model into the first GPU
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True, local_files_only=True)
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(self.model_dir, torch_dtype="bfloat16", device_map=0, local_files_only=True)
        self.llava_model.eval()

    def __call__(self, img_path):
        image = Image.open(img_path)

        if image.size != (384, 384):
            image = image.resize((384, 384), Image.LANCZOS)

        image = image.convert("RGB")
        pixel_values = TVF.pil_to_tensor(image)

        # Normalize the image
        pixel_values = pixel_values / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(torch.bfloat16).unsqueeze(0)

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": self.PROMPT,
            },
        ]

        # Format the conversation
        convo_string = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        # Tokenize the conversation
        convo_tokens = self.tokenizer.encode(convo_string, add_special_tokens=False, truncation=False)

        # Repeat the image tokens
        input_tokens = []
        for token in convo_tokens:
            if token == self.llava_model.config.image_token_index:
                input_tokens.extend([self.llava_model.config.image_token_index] * self.llava_model.config.image_seq_length)
            else:
                input_tokens.append(token)

        input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        # Generate the caption
        generate_ids = self.llava_model.generate(input_ids=input_ids.to('cuda'), pixel_values=pixel_values.to('cuda'), attention_mask=attention_mask.to('cuda'), max_new_tokens=300, do_sample=True, suppress_tokens=None, use_cache=True)[0]

        # Trim off the prompt
        generate_ids = generate_ids[input_ids.shape[1]:]

        # Decode the caption
        caption = self.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = caption.replace('\n', '').strip()

        return caption


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