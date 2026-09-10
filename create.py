from __future__ import annotations

import argparse
import sys
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
PROMPT_FILE_SUFFIX = ".txt"
PROMPT_STEM_SUFFIX = ".prompt"
DEFAULT_GENERATED_IMAGE_NAME = "generated.png"

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
    "RunDiffusion/Juggernaut-Z-Image": {"steps": 35, "guidance_scale": 3.5},
    "stabilityai/sdxl-turbo": {"steps": 28, "guidance_scale": 3.5},
    "stabilityai/stable-diffusion-xl-base-1.0": {"steps": 30, "guidance_scale": 5.0},
    "stabilityai/stable-diffusion-3-medium-diffusers": {"steps": 30, "guidance_scale": 5.0},
}
IMAGE_MODEL_PRESET_NUMBERS = tuple(range(1, len(IMAGE_MODEL_PRESETS) + 1))
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
        description="Generate an image from a prompt (file, stdin, or interactive input) using a local Diffusers model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=MODEL_PRESET_HELP,
    )
    parser.add_argument(
        "input_prompt",
        nargs="?",
        type=Path,
        help="Path to the input prompt text file (required unless --stdin, --ask, or --ask-multi is used).",
    )
    prompt_source_group = parser.add_mutually_exclusive_group()
    prompt_source_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read the prompt text from standard input.",
    )
    prompt_source_group.add_argument(
        "--ask",
        action="store_true",
        help="Read the prompt text using Python input().",
    )
    prompt_source_group.add_argument(
        "--ask-multi",
        action="store_true",
        help="Repeatedly ask for prompts using input(); stop when an empty prompt is entered.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            f"Output image path. Defaults to <prompt_stem>.png for file input, "
            f"or {DEFAULT_GENERATED_IMAGE_NAME} for --stdin/--ask/--ask-multi."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output image.",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip processing if output image already exists (ignored if --force is set).",
    )
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--image-model",
        default=DEFAULT_IMAGE_MODEL,
        help=f"Local Diffusers model used to generate the image. Default: {DEFAULT_IMAGE_MODEL}",
    )
    image_group.add_argument(
        "--image-model-preset",
        type=int,
        choices=IMAGE_MODEL_PRESET_NUMBERS,
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
    if stem.endswith(PROMPT_STEM_SUFFIX):
        stem = stem[: -len(PROMPT_STEM_SUFFIX)]
    return prompt_path.with_name(f"{stem}.png")


def default_output_name(prompt_path: Path) -> str:
    return default_output_path(prompt_path).name


def default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}.generated")


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


def validate_output_path(output_path: Path, force: bool) -> None:
    if not output_path.parent.exists():
        raise ValueError(f"output directory does not exist: {output_path.parent}")
    if output_path.exists() and not force:
        raise ValueError(f"refusing to overwrite existing file: {output_path}. Use --force to overwrite.")


def is_prompt_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == PROMPT_FILE_SUFFIX and path.stem.endswith(PROMPT_STEM_SUFFIX)


def list_prompt_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if is_prompt_file(path))


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
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "use_safetensors": True, "low_cpu_mem_usage": False}
    if dtype is torch.float16:
        kwargs["variant"] = "fp16"

    try:
        return pipeline_cls.from_pretrained(model_id, **kwargs)
    except (OSError, ValueError):
        if "variant" not in kwargs:
            raise
        kwargs.pop("variant")
        return pipeline_cls.from_pretrained(model_id, **kwargs)


def load_text2image_pipeline(model_id: str, device: str, dtype: Any) -> Any:
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

    return pipe


def _build_generator(seed: int | None, device: str, batch_size: int) -> Any:
    if seed is None:
        return None
    generator_device = "cuda" if device == "cuda" else "cpu"
    if batch_size == 1:
        return torch.Generator(device=generator_device).manual_seed(seed)
    # Match per-item deterministic behavior from the previous one-by-one generation path.
    return [torch.Generator(device=generator_device).manual_seed(seed) for _ in range(batch_size)]


def _promote_core_modules_to_fp32(pipe: Any, device: str) -> None:
    for module_name in ("unet", "vae", "text_encoder", "text_encoder_2"):
        module = getattr(pipe, module_name, None)
        if module is not None and hasattr(module, "to"):
            module.to(device=device, dtype=torch.float32)


def generate_images(
    pipe: Any,
    prompts: list[str],
    model_id: str,
    device: str,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    width: int,
    height: int,
) -> list[Any]:
    if not prompts:
        return []

    prompt_input: str | list[str] = prompts[0] if len(prompts) == 1 else prompts
    call_kwargs: dict[str, Any] = {
        "prompt": prompt_input,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
    }

    generator = _build_generator(seed=seed, device=device, batch_size=len(prompts))
    if generator is not None:
        call_kwargs["generator"] = generator

    try:
        result = pipe(**call_kwargs)
    except RuntimeError as exc:
        error_text = str(exc)
        if "Input type (struct c10::Half) and bias type (float) should be the same" not in error_text:
            raise

        _promote_core_modules_to_fp32(pipe, device)
        retry_kwargs = dict(call_kwargs)
        retry_generator = _build_generator(seed=seed, device=device, batch_size=len(prompts))
        if retry_generator is not None:
            retry_kwargs["generator"] = retry_generator
        result = pipe(**retry_kwargs)

    images = list(getattr(result, "images", []) or [])
    if not images:
        raise ValueError("image model did not return an image")

    if model_id in FP32_PREFERRED_MODELS:
        black_indices = [index for index, image in enumerate(images) if is_nearly_black_image(image)]
        if black_indices:
            _promote_core_modules_to_fp32(pipe, device)
            retry_steps = max(steps, 8)
            for index in black_indices:
                retry_kwargs = {
                    "prompt": prompts[index],
                    "num_inference_steps": retry_steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                }
                retry_generator = _build_generator(seed=seed, device=device, batch_size=1)
                if retry_generator is not None:
                    retry_kwargs["generator"] = retry_generator
                retried = pipe(**retry_kwargs)
                retried_images = list(getattr(retried, "images", []) or [])
                if retried_images:
                    images[index] = retried_images[0]

    return images


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
    pipe = load_text2image_pipeline(model_id=model_id, device=device, dtype=dtype)
    return generate_images(
        pipe=pipe,
        prompts=[prompt],
        model_id=model_id,
        device=device,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        width=width,
        height=height,
    )[0]


def read_prompt(prompt_path: Path) -> str:
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"input prompt file is empty: {prompt_path}")
    return prompt


def read_prompt_from_stdin() -> str:
    prompt = sys.stdin.read().strip()
    if not prompt:
        raise ValueError("stdin prompt is empty")
    return prompt


def read_prompt_from_interactive_input() -> str:
    try:
        prompt = input("Enter prompt: ").strip()
    except EOFError as exc:
        raise ValueError("no prompt received from interactive input") from exc
    if not prompt:
        raise ValueError("interactive prompt is empty")
    return prompt


def ask_for_existing_file_resolution(output_path: Path) -> Path:
    candidate = output_path
    while candidate.exists():
        answer = input(f"Output file already exists: {candidate}. Overwrite? [y/N]: ").strip().lower()
        if answer in {"y", "yes"}:
            return candidate

        while True:
            new_name = input("Enter a new output filename: ").strip()
            if not new_name:
                print("Filename cannot be empty.")
                continue
            new_path = candidate.parent / new_name
            if not new_path.suffix:
                new_path = new_path.with_suffix(".png")
            candidate = new_path
            break

    return candidate


def default_multi_output_path(base_output: Path, index: int) -> Path:
    if index <= 1:
        return base_output
    return base_output.with_name(f"{base_output.stem}_{index:03d}{base_output.suffix}")



def main(args: argparse.Namespace) -> None:
    validate_dimensions(args.width, args.height)

    image_model = resolve_model_choice(args.image_model, args.image_model_preset, IMAGE_MODEL_PRESETS)

    # Use per-preset defaults if steps/guidance_scale are not set
    preset_defaults = IMAGE_MODEL_PRESET_DEFAULTS.get(image_model, {})
    steps = args.steps if args.steps is not None else preset_defaults.get("steps", DEFAULT_STEPS)
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else preset_defaults.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)

    device, dtype = resolve_device_and_dtype(args.device, image_model)
    print(f"Using device: {device}")

    if (args.stdin or args.ask or args.ask_multi) and args.input_prompt is not None:
        raise ValueError("input_prompt cannot be used with --stdin, --ask, or --ask-multi")
    if not args.stdin and not args.ask and not args.ask_multi and args.input_prompt is None:
        raise ValueError("input_prompt is required unless --stdin, --ask, or --ask-multi is used")

    if args.input_prompt is not None and args.input_prompt.is_dir():
        if args.output is not None:
            raise ValueError("--output cannot be used when input_prompt is a directory")
        if args.stdin or args.ask or args.ask_multi:
            raise ValueError("--stdin, --ask, and --ask-multi cannot be used when input_prompt is a directory")

        output_dir = default_output_dir(args.input_prompt)
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt_paths = list_prompt_files(args.input_prompt)
        if not prompt_paths:
            raise ValueError(f"no prompt files found in directory: {args.input_prompt}")

        jobs: list[tuple[Path, Path]] = []
        for prompt_path in prompt_paths:
            output_path = output_dir / default_output_name(prompt_path)
            if output_path.exists():
                if args.force:
                    pass  # Overwrite
                elif args.skip:
                    print(f"Skipping existing file: {output_path}")
                    continue
                else:
                    raise ValueError(f"refusing to overwrite existing file: {output_path}. Use --force to overwrite or --skip to skip.")

            jobs.append((prompt_path, output_path))

        if not jobs:
            print("Nothing to process after applying --skip/--force rules.")
            return

        print(f"Loading image model: {image_model}")
        pipe = load_text2image_pipeline(model_id=image_model, device=device, dtype=dtype)
        print(f"Processing {len(jobs)} prompt file(s) from: {args.input_prompt}")

        for prompt_path, output_path in jobs:
            prompt = read_prompt(prompt_path)
            print("Generating image from prompt...")
            generated_image = generate_images(
                pipe=pipe,
                prompts=[prompt],
                model_id=image_model,
                device=device,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=args.seed,
                width=args.width,
                height=args.height,
            )[0]
            generated_image.save(output_path)
            print(f"Saved image: {output_path}")
        return

    if args.input_prompt is not None:
        output_path = args.output or default_output_path(args.input_prompt)
    else:
        output_path = args.output or Path(DEFAULT_GENERATED_IMAGE_NAME)

    if args.ask_multi:
        validate_output_path(output_path, force=True)
        print(f"Loading image model: {image_model}")
        pipe = load_text2image_pipeline(model_id=image_model, device=device, dtype=dtype)

        generated_count = 0
        while True:
            prompt = input("Enter prompt (empty to finish): ").strip()
            if not prompt:
                break

            generated_count += 1
            candidate_output_path = default_multi_output_path(output_path, generated_count)
            if candidate_output_path.exists() and not args.force:
                candidate_output_path = ask_for_existing_file_resolution(candidate_output_path)

            print("Generating image from prompt...")
            generated_image = generate_images(
                pipe=pipe,
                prompts=[prompt],
                model_id=image_model,
                device=device,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=args.seed,
                width=args.width,
                height=args.height,
            )[0]
            generated_image.save(candidate_output_path)
            print(f"Saved image: {candidate_output_path}")

        if generated_count == 0:
            print("No prompts entered. Nothing generated.")
        return

    if output_path.exists():
        if args.force:
            pass  # Overwrite
        elif args.skip:
            print(f"Skipping existing file: {output_path}")
            return
        else:
            if args.input_prompt is not None:
                validate_paths(args.input_prompt, output_path, args.force)
            else:
                validate_output_path(output_path, args.force)
    else:
        if args.input_prompt is not None:
            validate_paths(args.input_prompt, output_path, args.force)
        else:
            validate_output_path(output_path, args.force)

    if args.stdin:
        prompt = read_prompt_from_stdin()
    elif args.ask:
        prompt = read_prompt_from_interactive_input()
    else:
        prompt = read_prompt(args.input_prompt)

    print(f"Loading image model: {image_model}")
    pipe = load_text2image_pipeline(model_id=image_model, device=device, dtype=dtype)
    print("Generating image from prompt...")
    generated_image = generate_images(
        pipe=pipe,
        prompts=[prompt],
        model_id=image_model,
        device=device,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )[0]

    generated_image.save(output_path)
    print(f"Saved image: {output_path}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
