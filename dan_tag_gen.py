import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from .lib_dantaggen.app import get_result
from .lib_dantaggen.kgen.metainfo import TARGET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

_MODEL_CACHE: dict[str, tuple[LlamaForCausalLM, LlamaTokenizer]] = {}


def _resolve_model_dir(model_dir: str | None) -> str:
    if not model_dir:
        return LOCAL_MODEL_DIR
    candidate = os.path.expanduser(model_dir)
    if os.path.isabs(candidate):
        return candidate
    # try relative to current working directory first
    cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), candidate))
    if os.path.isdir(cwd_candidate):
        return cwd_candidate
    # fallback to path relative to this file
    return os.path.abspath(os.path.join(os.path.dirname(__file__), candidate))


def _load_model_and_tokenizer(model_dir: str | None):
    resolved = _resolve_model_dir(model_dir)
    if resolved in _MODEL_CACHE:
        return _MODEL_CACHE[resolved]

    text_model = LlamaForCausalLM.from_pretrained(
        resolved,
        local_files_only=True,
    )
    if DEVICE == "cuda":
        text_model = text_model.half()
    text_model = text_model.requires_grad_(False).eval().to(DEVICE)

    tokenizer = LlamaTokenizer.from_pretrained(
        resolved,
        local_files_only=True,
    )

    _MODEL_CACHE[resolved] = (text_model, tokenizer)
    return _MODEL_CACHE[resolved]


class DanTagGen:
    """DanTagGen node."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": LOCAL_MODEL_DIR}),
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
        model: str,
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
        text_model, tokenizer = _load_model_and_tokenizer(model)
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
