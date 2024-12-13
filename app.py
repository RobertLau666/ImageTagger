import os
import json
import torch
from models import WDTagger, JoyTag, LlamaJoycaptionAlphaTwo, LLaVANeXT
from tqdm import tqdm
import argparse
import config


class PipesTag():
    def __init__(self, args):
        caption_model_paths = config.caption_model_paths
        caption_model_names_list = [item.strip() for item in args.caption_model_names.split(",")]
        if 'WDTagger' in caption_model_names_list:
            self.wd_tagger = WDTagger(model_dir=caption_model_paths['WDTagger'])
        if 'JoyTag' in caption_model_names_list:
            self.joy_tag = JoyTag(model_dir=caption_model_paths['JoyTag'])
        if 'LlamaJoycaptionAlphaTwo' in caption_model_names_list:
            self.llama_joycaption_alphatwo = LlamaJoycaptionAlphaTwo(model_dir=caption_model_paths['LlamaJoycaptionAlphaTwo'])
        if 'LLaVANeXT' in caption_model_names_list:
            self.llava_next = LLaVANeXT(model_dir=caption_model_paths['LLaVANeXT'])

    def __call__(self, tag_img_path):
        result = {}
        with torch.no_grad():
            result["wd_tagger"] = self.wd_tagger(tag_img_path)
            result["joy_tag"] = self.joy_tag(tag_img_path)
            result["llama_joycaption_alphatwo"] = self.llama_joycaption_alphatwo(tag_img_path)
            result["llava_next"] = self.llava_next(tag_img_path)
        return result

def main(args):
    tag_record_dict = {}
    assert args.caption_img_num >= -1, "Warning: caption_img_num is less than -1."
    tag_img_names = os.listdir(args.caption_imgs_dir) if args.caption_img_num == -1 else os.listdir(args.caption_imgs_dir)[:args.caption_img_num]
    pipes_tag = PipesTag(args)

    for tag_img_name in tqdm(tag_img_names):
        tag_img_path = os.path.join(args.caption_imgs_dir, tag_img_name)
        tag_record_dict[tag_img_name] = pipes_tag(tag_img_path)

        with open(args.record_caption_json_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(tag_record_dict, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PuLID for FLUX.1-dev")
    parser.add_argument("--caption_model_names", type=str, default="WDTagger,JoyTag,LlamaJoycaptionAlphaTwo,LLaVANeXT", help="separated by comma")
    parser.add_argument("--caption_imgs_dir", type=str, default="", help="caption_imgs_dir")
    parser.add_argument("--caption_img_num", type=int, default=-1, help="-1 represents all elements")
    parser.add_argument("--record_caption_json_path", type=str, default="", help="record_caption_json_path")
    args = parser.parse_args()

    main(args)