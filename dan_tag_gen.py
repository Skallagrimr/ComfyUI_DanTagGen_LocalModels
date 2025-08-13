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

    candidate_input = os.path.expanduser(model_dir)
    # Absolute path handling
    if os.path.isabs(candidate_input):
        if os.path.isdir(candidate_input):
            return candidate_input
        # If absolute path is invalid but contains a 'custom_nodes' segment, try resolving from CWD
        marker = f"{os.sep}custom_nodes{os.sep}"
        if marker in candidate_input:
            tail = candidate_input.split(marker, maxsplit=1)[-1]
            cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), "custom_nodes", tail))
            if os.path.isdir(cwd_candidate):
                return cwd_candidate
        # fallback to default
        return LOCAL_MODEL_DIR

    # Relative path handling: try CWD first (ComfyUI root), then relative to this file
    cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), candidate_input))
    if os.path.isdir(cwd_candidate):
        return cwd_candidate
    filedir_candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), candidate_input))
    if os.path.isdir(filedir_candidate):
        return filedir_candidate

    # fallback to default models dir
    return LOCAL_MODEL_DIR


def _load_model_and_tokenizer(model_dir: str | None):
    resolved = _resolve_model_dir(model_dir)
    if resolved in _MODEL_CACHE:
        return _MODEL_CACHE[resolved]

    # Validate expected files exist to avoid huggingface repo-id validation path
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    missing = [f for f in required_files if not os.path.isfile(os.path.join(resolved, f))]
    if missing:
        raise FileNotFoundError(
            f"Model directory does not contain required files: {', '.join(missing)} | resolved: {resolved}"
        )

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
