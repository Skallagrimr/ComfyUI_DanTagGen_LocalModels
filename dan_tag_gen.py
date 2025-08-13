import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from .lib_dantaggen.app import get_result
from .lib_dantaggen.kgen.metainfo import TARGET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

_TEXT_MODEL = None
_TOKENIZER = None


def _load_local_model_and_tokenizer():
    global _TEXT_MODEL, _TOKENIZER
    if _TEXT_MODEL is not None and _TOKENIZER is not None:
        return _TEXT_MODEL, _TOKENIZER

    text_model = LlamaForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        local_files_only=True,
    )
    # Keep fp32 on CPU; use fp16 on CUDA like original code
    if DEVICE == "cuda":
        text_model = text_model.half()
    text_model = text_model.requires_grad_(False).eval().to(DEVICE)

    tokenizer = LlamaTokenizer.from_pretrained(
        LOCAL_MODEL_DIR,
        local_files_only=True,
    )

    _TEXT_MODEL = text_model
    _TOKENIZER = tokenizer
    return _TEXT_MODEL, _TOKENIZER


class DanTagGen:
    """DanTagGen node."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "artist": ("STRING", {"default": ""}),
                "characters": ("STRING", {"default": ""}),
                "copyrights": ("STRING", {"default": ""}),
                "special_tags": ("STRING", {"default": ""}),
                "general": ("STRING", {"default": "", "multiline": True}),
                "blacklist": ("STRING", {"default": ""}),
                "rating": (["safe", "sensitive", "nsfw", "nsfw, explicit"],),
                "target": (list(TARGET.keys()),),
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 32},
                ),
                "escape_bracket": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 1.35, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output", "llm_output")
    FUNCTION = "generate"
    CATEGORY = "_for_testing"

    def generate(
        self,
        rating: str,
        artist: str,
        characters: str,
        copyrights: str,
        target: str,
        special_tags: str,
        general: str,
        width: float,
        height: float,
        blacklist: str,
        escape_bracket: bool,
        temperature: float,
    ):
        text_model, tokenizer = _load_local_model_and_tokenizer()
        result = list(
            get_result(
                text_model,
                tokenizer,
                rating,
                artist,
                characters,
                copyrights,
                target,
                [s.strip() for s in special_tags.split(",") if s],
                general,
                width / height,
                blacklist,
                escape_bracket,
                temperature,
            )
        )[-1]
        output, llm_output, _ = result
        return {"result": (output, llm_output)}


NODE_CLASS_MAPPINGS = {
    "PromptDanTagGen": DanTagGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptDanTagGen": "Danbooru Tag Generator",
}
