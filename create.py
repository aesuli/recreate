from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import ImageStat

DEFAULT_IMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_STEPS = 28  # Default for Z-Image-Turbo (adjust if needed)
DEFAULT_GUIDANCE_SCALE = 3.5  # Default for Z-Image-Turbo (adjust if needed)
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DIMENSION_MULTIPLE = 8

IMAGE_MODEL_PRESETS: tuple[tuple[str, str], ...] = (
    ("Tongyi-MAI/Z-Image-Turbo", "Z-Image-Turbo: fastest Z-Image variant (default)"),
    ("RunDiffusion/Juggernaut-Z-Image", "Juggernaut Z: cinematic, sharp, and balanced"),
    ("stabilityai/sdxl-turbo", "fastest SDXL option for quick recreations"),
    ("stabilityai/stable-diffusion-xl-base-1.0", "higher-quality SDXL base model"),
    ("stabilityai/stable-diffusion-3-medium-diffusers", "most capable preset, but heaviest"),
)

# Per-preset default values for steps and guidance scale
IMAGE_MODEL_PRESET_DEFAULTS = {
    "Tongyi-MAI/Z-Image-Turbo": {"steps": 28, "guidance_scale": 3.5},
    "RunDiffusion/Juggernaut-Z-Image": {"steps": 35, "guidance_scale": 6.0},
    "stabilityai/sdxl-turbo": {"steps": 28, "guidance_scale": 3.5},
    "stabilityai/stable-diffusion-xl-base-1.0": {"steps": 30, "guidance_scale": 5.0},
    "stabilityai/stable-diffusion-3-medium-diffusers": {"steps": 30, "guidance_scale": 5.0},
}
IMAGE_MODEL_OPTION_NUMBERS = tuple(range(1, len(IMAGE_MODEL_PRESETS) + 1))
FP32_PREFERRED_MODELS = {
    "Tongyi-MAI/Z-Image-Turbo",
    "RunDiffusion/Juggernaut-Z-Image",
}


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def format_model_presets(title: str, presets: tuple[tuple[str, str], ...]) -> str:
    lines = [title]
    for index, (model_id, description) in enumerate(presets, start=1):
        lines.append(f"  {index}. {model_id} - {description}")
    return "\n".join(lines)


MODEL_PRESET_HELP = format_model_presets(
    "Image model presets, from lighter to stronger:",
    IMAGE_MODEL_PRESETS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an image from a prompt text file using a local Diffusers model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=MODEL_PRESET_HELP,
    )
    parser.add_argument("input_prompt", type=Path, help="Path to the input prompt text file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path. Defaults to <prompt_stem>.png beside the prompt file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output image.",
    )
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--image-model",
        default=DEFAULT_IMAGE_MODEL,
        help=f"Local Diffusers model used to generate the image. Default: {DEFAULT_IMAGE_MODEL}",
    )
    image_group.add_argument(
        "--image-model-option",
        type=int,
        choices=IMAGE_MODEL_OPTION_NUMBERS,
        help="Select a numbered image-model preset. See the preset list in --help.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible generation.",
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=None,
        help="Diffusion inference steps. Uses the preset's recommended value if not set.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale. Uses the preset's recommended value if not set.",
    )
    parser.add_argument(
        "--width",
        type=positive_int,
        default=DEFAULT_WIDTH,
        help=f"Output image width. Must be a multiple of {DIMENSION_MULTIPLE}. Default: {DEFAULT_WIDTH}",
    )
    parser.add_argument(
        "--height",
        type=positive_int,
        default=DEFAULT_HEIGHT,
        help=f"Output image height. Must be a multiple of {DIMENSION_MULTIPLE}. Default: {DEFAULT_HEIGHT}",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device. Default: auto",
    )
    return parser


def default_output_path(prompt_path: Path) -> Path:
    stem = prompt_path.stem
    if stem.endswith(".prompt"):
        stem = stem[: -len(".prompt")]
    return prompt_path.with_name(f"{stem}.png")


def resolve_model_choice(
    explicit_model: str,
    preset_number: int | None,
    presets: tuple[tuple[str, str], ...],
) -> str:
    if preset_number is None:
        return explicit_model
    return presets[preset_number - 1][0]


def validate_dimensions(width: int, height: int) -> None:
    if width % DIMENSION_MULTIPLE != 0 or height % DIMENSION_MULTIPLE != 0:
        raise ValueError(f"--width and --height must be multiples of {DIMENSION_MULTIPLE}")


def validate_paths(prompt_path: Path, output_path: Path, force: bool) -> None:
    if not prompt_path.exists():
        raise ValueError(f"input prompt does not exist: {prompt_path}")
    if not prompt_path.is_file():
        raise ValueError(f"input prompt path is not a file: {prompt_path}")
    if not output_path.parent.exists():
        raise ValueError(f"output directory does not exist: {output_path.parent}")
    if output_path.exists() and not force:
        raise ValueError(f"refusing to overwrite existing file: {output_path}. Use --force to overwrite.")


def resolve_device_and_dtype(requested_device: str, model_id: str) -> tuple[str, Any]:
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

    # Keep non-CUDA in fp32 for stability.
    if device == "cuda":
        # Some SDXL derivatives are more stable in full precision.
        dtype = torch.float32 if model_id in FP32_PREFERRED_MODELS else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def is_nearly_black_image(image: Any) -> bool:
    rgb = image.convert("RGB")
    stat = ImageStat.Stat(rgb)
    mean_luma = sum(stat.mean) / 3.0
    extrema = rgb.getextrema()
    max_channel_value = max(channel_max for _, channel_max in extrema)
    return mean_luma <= 3.0 and max_channel_value <= 12


def load_diffusion_pipeline(pipeline_cls: Any, model_id: str, dtype: Any) -> Any:
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "use_safetensors": True}
    if dtype is torch.float16:
        kwargs["variant"] = "fp16"

    try:
        return pipeline_cls.from_pretrained(model_id, **kwargs)
    except (OSError, ValueError):
        if "variant" not in kwargs:
            raise
        kwargs.pop("variant")
        return pipeline_cls.from_pretrained(model_id, **kwargs)


def generate_image(
    prompt: str,
    model_id: str,
    device: str,
    dtype: Any,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    width: int,
    height: int,
) -> Any:
    from diffusers import AutoPipelineForText2Image

    pipe = load_diffusion_pipeline(AutoPipelineForText2Image, model_id, dtype)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    try:
        pipe = pipe.to(device=device, torch_dtype=dtype)
    except TypeError:
        pipe = pipe.to(device)
        for module_name in ("unet", "vae", "text_encoder", "text_encoder_2"):
            module = getattr(pipe, module_name, None)
            if module is not None and hasattr(module, "to"):
                module.to(device=device, dtype=dtype)

    generator = None
    if seed is not None:
        generator_device = "cuda" if device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)

    call_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
    }
    if generator is not None:
        call_kwargs["generator"] = generator

    try:
        result = pipe(**call_kwargs)
    except RuntimeError as exc:
        error_text = str(exc)
        if "Input type (struct c10::Half) and bias type (float) should be the same" not in error_text:
            raise

        # Recover from VAE/unet mixed precision by promoting core modules to fp32.
        for module_name in ("unet", "vae", "text_encoder", "text_encoder_2"):
            module = getattr(pipe, module_name, None)
            if module is not None and hasattr(module, "to"):
                module.to(device=device, dtype=torch.float32)

        retry_kwargs = dict(call_kwargs)
        if seed is not None:
            generator_device = "cuda" if device == "cuda" else "cpu"
            retry_kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(seed)
        result = pipe(**retry_kwargs)

    if not getattr(result, "images", None):
        raise ValueError("image model did not return an image")

    generated_image = result.images[0]
    if model_id in FP32_PREFERRED_MODELS and is_nearly_black_image(generated_image):
        for module_name in ("unet", "vae", "text_encoder", "text_encoder_2"):
            module = getattr(pipe, module_name, None)
            if module is not None and hasattr(module, "to"):
                module.to(device=device, dtype=torch.float32)

        retry_kwargs = dict(call_kwargs)
        retry_kwargs["num_inference_steps"] = max(steps, 8)
        if seed is not None:
            generator_device = "cuda" if device == "cuda" else "cpu"
            retry_kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(seed)

        retried = pipe(**retry_kwargs)
        if getattr(retried, "images", None):
            generated_image = retried.images[0]

    return generated_image


def read_prompt(prompt_path: Path) -> str:
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"input prompt file is empty: {prompt_path}")
    return prompt



def main(args: argparse.Namespace) -> None:
    output_path = args.output or default_output_path(args.input_prompt)
    validate_paths(args.input_prompt, output_path, args.force)
    validate_dimensions(args.width, args.height)

    image_model = resolve_model_choice(args.image_model, args.image_model_option, IMAGE_MODEL_PRESETS)
    prompt = read_prompt(args.input_prompt)

    # Use per-preset defaults if steps/guidance_scale are not set
    preset_defaults = IMAGE_MODEL_PRESET_DEFAULTS.get(image_model, {})
    steps = args.steps if args.steps is not None else preset_defaults.get("steps", DEFAULT_STEPS)
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else preset_defaults.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)

    device, dtype = resolve_device_and_dtype(args.device, image_model)
    print(f"Using device: {device}")

    print(f"Loading image model: {image_model}")
    print("Generating image from prompt...")
    generated_image = generate_image(
        prompt=prompt,
        model_id=image_model,
        device=device,
        dtype=dtype,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )

    generated_image.save(output_path)
    print(f"Saved image: {output_path}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
