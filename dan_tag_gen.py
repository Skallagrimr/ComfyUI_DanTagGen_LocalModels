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
    # If root doesn't have config.json, try to auto-descend into a single subdir that does
    if not os.path.isfile(os.path.join(resolved, "config.json")):
        try:
            subdirs = [
                os.path.join(resolved, name)
                for name in os.listdir(resolved)
                if os.path.isdir(os.path.join(resolved, name))
            ]
        except FileNotFoundError:
            subdirs = []
        candidates = [d for d in subdirs if os.path.isfile(os.path.join(d, "config.json"))]
        if len(candidates) == 1:
            resolved = candidates[0]
    if resolved in _MODEL_CACHE:
        return _MODEL_CACHE[resolved]

    # Validate local layout with minimal requirements
    required_present = [os.path.isfile(os.path.join(resolved, "config.json"))]
    model_file_present = any(
        os.path.isfile(os.path.join(resolved, fname))
        for fname in ("model.safetensors", "pytorch_model.bin")
    )
    tokenizer_present = any(
        os.path.isfile(os.path.join(resolved, fname)) for fname in ("tokenizer.model", "tokenizer.json")
    )
    if not all(required_present) or not model_file_present or not tokenizer_present:
        dir_listing = []
        try:
            dir_listing = sorted(os.listdir(resolved))[:50]
        except Exception:
            pass
        missing_parts = []
        if not required_present[0]:
            missing_parts.append("config.json")
        if not model_file_present:
            missing_parts.append("model.safetensors|pytorch_model.bin")
        if not tokenizer_present:
            missing_parts.append("tokenizer.model|tokenizer.json")
        raise FileNotFoundError(
            "Model directory does not contain required files: "
            + ", ".join(missing_parts)
            + f" | resolved: {resolved} | found: {dir_listing}"
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
