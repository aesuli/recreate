from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import pipeline

DEFAULT_VISION_MODEL = "openbmb/MiniCPM-V-4.6"
DEFAULT_MAX_SIZE = 512
DEFAULT_MAX_NEW_TOKENS = 256
DIMENSION_MULTIPLE = 8

VISION_MODEL_PRESETS: tuple[tuple[str, str, int, int], ...] = (
    ("openbmb/MiniCPM-V-4.6", "default image-text-to-text model", 512, 256),
    ("HuggingFaceTB/SmolVLM-256M-Instruct", "smallest and fastest local VLM", 256, 128),
    ("llava-hf/llava-1.5-7b-hf", "stronger general-purpose 7B vision-language model", 1024, 512),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "largest and strongest prompt-generation option", 2048, 1024),
)
VISION_MODEL_OPTION_NUMBERS = tuple(range(1, len(VISION_MODEL_PRESETS) + 1))

PROMPT_REQUEST = """
Write a prompt describing the image to enable an image generation model to replicate it.
Include any element that is relevant to exactly replicate the image not only in its content but also in its visual appearance.
Relevant elements are the subject, setting, composition, camera angle, lighting, colors, materials, textures, background, image style, mood, any legible text, number, or symbol.
""".strip()


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def format_model_presets(title: str, presets: tuple[tuple[str, str, int, int], ...]) -> str:
    lines = [title]
    for index, (model_id, description, max_size, max_new_tokens) in enumerate(presets, start=1):
        lines.append(
            f"  {index}. {model_id} - {description} (Max Size: {max_size}, Max New Tokens: {max_new_tokens})"
        )
    return "\n".join(lines)


MODEL_PRESET_HELP = format_model_presets(
    "Vision model presets, from lighter to stronger:",
    VISION_MODEL_PRESETS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a text prompt from an input image using a local vision-language model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=MODEL_PRESET_HELP,
    )
    parser.add_argument("input_image", type=Path, help="Path to the image to describe.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output prompt path. Defaults to <input_stem>_recreated.prompt.txt beside the input.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output prompt file.",
    )
    vision_group = parser.add_mutually_exclusive_group()
    vision_group.add_argument(
        "--vision-model",
        default=DEFAULT_VISION_MODEL,
        help=f"Local Hugging Face VLM used to describe the input image. Default: {DEFAULT_VISION_MODEL}",
    )
    vision_group.add_argument(
        "--vision-model-option",
        type=int,
        choices=VISION_MODEL_OPTION_NUMBERS,
        help="Select a numbered vision-model preset. See the preset list in --help.",
    )
    parser.add_argument(
        "--max-size",
        type=positive_int,
        default=DEFAULT_MAX_SIZE,
        help=f"Resize the longest side before prompt generation. Default: {DEFAULT_MAX_SIZE}",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device. Default: auto",
    )
    return parser


def default_prompt_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_recreated.prompt.txt")


def resolve_model_choice(
    explicit_model: str,
    preset_number: int | None,
    presets: tuple[tuple[str, str], ...],
) -> str:
    if preset_number is None:
        return explicit_model
    return presets[preset_number - 1][0]


def validate_paths(input_path: Path, output_path: Path, force: bool) -> None:
    if not input_path.exists():
        raise ValueError(f"input image does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"input path is not a file: {input_path}")
    if not output_path.parent.exists():
        raise ValueError(f"output directory does not exist: {output_path.parent}")
    if output_path.exists() and not force:
        raise ValueError(f"refusing to overwrite existing file: {output_path}. Use --force to overwrite.")


def resolve_device_and_dtype(requested_device: str) -> tuple[str, Any]:
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested_device

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false")
    if device == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            raise ValueError("MPS was requested, but torch.backends.mps.is_available() is false")

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    return device, dtype


def round_to_multiple(value: float) -> int:
    rounded = int(round(value / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
    return max(DIMENSION_MULTIPLE, rounded)


def resize_for_diffusion(image: Any, max_size: int) -> Any:
    target_long_side = max_size - (max_size % DIMENSION_MULTIPLE)
    if target_long_side < DIMENSION_MULTIPLE:
        raise ValueError(f"--max-size must be at least {DIMENSION_MULTIPLE}")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("input image has invalid dimensions")

    scale = target_long_side / max(width, height)
    resized_width = round_to_multiple(width * scale)
    resized_height = round_to_multiple(height * scale)

    if width >= height:
        resized_width = target_long_side
    else:
        resized_height = target_long_side

    return image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)


def load_image(input_path: Path, max_size: int) -> Any:
    try:
        image = Image.open(input_path).convert("RGB")
    except OSError as exc:
        raise ValueError(f"could not open input image: {input_path}") from exc
    return resize_for_diffusion(image, max_size)


def extract_generated_text(outputs: Any) -> str:
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, list) and outputs:
        return extract_generated_text(outputs[0])
    if isinstance(outputs, dict):
        if "generated_text" in outputs:
            return extract_generated_text(outputs["generated_text"])
        if "content" in outputs:
            return extract_generated_text(outputs["content"])
        if outputs.get("role") == "assistant" and "text" in outputs:
            return str(outputs["text"])
    return str(outputs)


def clean_prompt(prompt: str) -> str:
    cleaned = prompt.strip()
    for prefix in ("Prompt:", "Image generation prompt:", "Reconstruction prompt:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
            break
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    return " ".join(cleaned.split())


def generate_reconstruction_prompt(
    image: Any, model_id: str, device: str, dtype: Any
) -> str:
    device_arg: int | str
    if device == "cuda":
        device_arg = 0
    elif device == "mps":
        device_arg = device
    else:
        device_arg = -1

    pipeline_attempts = [
        {
            "task": "image-text-to-text",
            "model": model_id,
            "device": device_arg,
            "dtype": dtype,
            "trust_remote_code": True,
        },
        {
            "task": "image-text-to-text",
            "model": model_id,
            "device": device_arg,
            "trust_remote_code": True,
        },
        {
            "task": "image-to-text",
            "model": model_id,
            "device": device_arg,
            "dtype": dtype,
            "trust_remote_code": True,
        },
        {
            "task": "image-to-text",
            "model": model_id,
            "device": device_arg,
            "trust_remote_code": True,
        },
    ]

    last_error: Exception | None = None
    vlm = None
    for pipe_kwargs in pipeline_attempts:
        try:
            vlm = pipeline(**pipe_kwargs)
            break
        except (TypeError, ValueError, OSError) as exc:
            last_error = exc

    if vlm is None:
        raise ValueError(
            "failed to load the vision model. If you are using SmolVLM, upgrade "
            "transformers to the latest version. "
            f"Underlying error: {last_error}"
        )

    message_variants = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_REQUEST},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": PROMPT_REQUEST},
                ],
            }
        ],
    ]

    outputs = None
    for messages in message_variants:
        try:
            outputs = vlm(
                text=messages,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                return_full_text=False,
                clean_up_tokenization_spaces=False,
            )
            break
        except (TypeError, ValueError):
            continue

    if outputs is None:
        try:
            outputs = vlm(
                images=[image],
                prompt=PROMPT_REQUEST,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            outputs = vlm(
                image,
                prompt=PROMPT_REQUEST,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                clean_up_tokenization_spaces=False,
            )

    prompt = clean_prompt(extract_generated_text(outputs))
    if not prompt:
        raise ValueError("vision model returned an empty prompt")
    return prompt


def main(args: argparse.Namespace) -> None:
    input_path = args.input_image
    output_path = args.output or default_prompt_output_path(input_path)
    validate_paths(input_path, output_path, args.force)

    vision_model = resolve_model_choice(args.vision_model, args.vision_model_option, VISION_MODEL_PRESETS)
    device, dtype = resolve_device_and_dtype(args.device)
    print(f"Using device: {device}")

    source_image = load_image(input_path, args.max_size)

    print(f"Loading vision model: {vision_model}")
    prompt = generate_reconstruction_prompt(source_image, vision_model, device, dtype)
    print(f"Generated prompt ({len(prompt)} characters).")

    output_path.write_text(prompt + "\n", encoding="utf-8")
    print(f"Saved prompt: {output_path}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
