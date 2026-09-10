from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import torch

import describe
import create


def _format_vision_presets() -> str:
    lines = ["Vision model presets:"]
    for i, (model_id, description, _, _) in enumerate(describe.VISION_MODEL_PRESETS, 1):
        lines.append(f"  {i}. {model_id} - {description}")
    return "\n".join(lines)


def _format_image_presets() -> str:
    lines = ["Image model presets:"]
    for i, (model_id, description) in enumerate(create.IMAGE_MODEL_PRESETS, 1):
        lines.append(f"  {i}. {model_id} - {description}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infinite loop: prompt -> image -> next prompt -> next image, until killed."
    )
    parser.add_argument("directory", type=Path, help="Directory containing prompt_N.txt files.")
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Index of the first prompt to read (default: 0).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device. Default: auto",
    )
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--image-model",
        default=create.DEFAULT_IMAGE_MODEL,
        help=f"Diffusers model for text-to-image generation. Default: {create.DEFAULT_IMAGE_MODEL}\n\n{_format_image_presets()}",
    )
    image_group.add_argument(
        "--image-model-preset",
        type=int,
        choices=create.IMAGE_MODEL_PRESET_NUMBERS,
        help="Select a numbered image-model preset. See the preset list in --help.",
    )
    vision_group = parser.add_mutually_exclusive_group()
    vision_group.add_argument(
        "--vision-model",
        default=describe.DEFAULT_VISION_MODEL,
        help=f"Hugging Face VLM for image-to-prompt generation. Default: {describe.DEFAULT_VISION_MODEL}\n\n{_format_vision_presets()}",
    )
    vision_group.add_argument(
        "--vision-model-preset",
        type=int,
        choices=describe.VISION_MODEL_PRESET_NUMBERS,
        help="Select a numbered vision-model preset. See the preset list in --help.",
    )
    parser.add_argument(
        "--width",
        type=create.positive_int,
        default=create.DEFAULT_WIDTH,
        help=f"Output image width (default: {create.DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=create.positive_int,
        default=create.DEFAULT_HEIGHT,
        help=f"Output image height (default: {create.DEFAULT_HEIGHT}).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of loop iterations to run before exiting (default: run indefinitely).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be >= 0")

    # Resolve model choices
    image_model = create.resolve_model_choice(
        args.image_model,
        args.image_model_preset,
        create.IMAGE_MODEL_PRESETS,
    )
    vision_model = describe.resolve_model_choice(
        args.vision_model,
        args.vision_model_preset,
        describe.VISION_MODEL_PRESETS,
    )

    work_dir = args.directory.resolve()
    if not work_dir.is_dir():
        print(f"error: not a directory: {work_dir}", file=sys.stderr)
        sys.exit(1)

    running = True

    def handle_signal(signum: int, frame: object) -> None:
        nonlocal running
        if running:
            print("\nReceived signal, shutting down after current iteration...")
            running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    idx = args.start_from
    device = args.device
    remaining_steps = args.steps

    print("Loading vision model (describe)...")
    vlm_device, vlm_dtype = describe.resolve_device_and_dtype(device)
    vlm = describe.load_vlm_pipeline(
        model_id=vision_model,
        device=vlm_device,
        dtype=vlm_dtype,
    )
    print(f"Vision model loaded on {vlm_device}")

    print("Loading image model (create)...")
    create_device, create_dtype = create.resolve_device_and_dtype(device, image_model)

    preset_defaults = create.IMAGE_MODEL_PRESET_DEFAULTS.get(
        image_model,
        {"steps": create.DEFAULT_STEPS, "guidance_scale": create.DEFAULT_GUIDANCE_SCALE},
    )
    pipe = create.load_text2image_pipeline(
        model_id=image_model,
        device=create_device,
        dtype=create_dtype,
    )
    print(f"Image model loaded on {create_device}")

    while running and (remaining_steps is None or remaining_steps > 0):
        prompt_path = work_dir / f"prompt_{idx}.txt"
        image_path = work_dir / f"image_{idx}.png"

        if image_path.exists():
            print(f"[{idx}] {image_path} exists, using it directly...")
            source_image = describe.load_image(image_path, describe.DEFAULT_MAX_SIZE)
            print(f"[{idx}] Describing {image_path}...")
            with torch.no_grad():
                next_prompt = describe.generate_reconstruction_prompt(source_image, vlm)
            next_path = work_dir / f"prompt_{idx + 1}.txt"
            next_path.write_text(next_prompt + "\n", encoding="utf-8")
            print(f"[{idx}] Saved {next_path}")
            idx += 1
            if remaining_steps is not None:
                remaining_steps -= 1
            continue

        if not prompt_path.exists():
            print(f"Waiting for {prompt_path} to exist...")
            signal.pause()
            continue

        prompt_text = create.read_prompt(prompt_path)
        print(f"[{idx}] Generating image from prompt_{idx}.txt...")

        with torch.no_grad():
            generated = create.generate_images(
                pipe=pipe,
                prompts=[prompt_text],
                model_id=image_model,
                device=create_device,
                steps=preset_defaults["steps"],
                guidance_scale=preset_defaults["guidance_scale"],
                seed=None,
                width=args.width,
                height=args.height,
            )[0]
        generated.save(image_path)
        print(f"[{idx}] Saved {image_path}")

        if not running:
            break

        source_image = describe.load_image(image_path, describe.DEFAULT_MAX_SIZE)

        print(f"[{idx}] Describing {image_path}...")
        with torch.no_grad():
            next_prompt = describe.generate_reconstruction_prompt(source_image, vlm)

        next_path = work_dir / f"prompt_{idx + 1}.txt"
        next_path.write_text(next_prompt + "\n", encoding="utf-8")
        print(f"[{idx}] Saved {next_path}")

        idx += 1
        if remaining_steps is not None:
            remaining_steps -= 1

    if remaining_steps == 0:
        print("Requested number of steps completed. Exiting.")
    print("Loop stopped.")


if __name__ == "__main__":
    main()
