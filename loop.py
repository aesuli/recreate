from __future__ import annotations

import argparse
import re
import signal
import shutil
import sys
from pathlib import Path

import torch

import describe
import create


def find_start_index(work_dir: Path) -> int:
    numbered_items = []
    for path in work_dir.iterdir():
        match = re.fullmatch(r"(?:prompt|image)_(\d+)\.(?:txt|png)", path.name)
        if match and path.is_file():
            numbered_items.append(int(match.group(1)))
    return max(numbered_items, default=0)


def copy_start_file(start_file: Path, work_dir: Path) -> Path:
    if not start_file.is_file():
        raise ValueError(f"start file is not a file: {start_file}")
    if any(work_dir.iterdir()):
        raise ValueError("--start-file requires an empty loop directory")

    suffix = start_file.suffix.lower()
    if suffix == ".txt":
        destination = work_dir / "prompt_0.txt"
    elif suffix in describe.SUPPORTED_IMAGE_SUFFIXES:
        destination = work_dir / "image_0.png"
    else:
        supported = ", ".join(sorted({".txt", *describe.SUPPORTED_IMAGE_SUFFIXES}))
        raise ValueError(f"unsupported start file type {start_file.suffix!r}; use one of: {supported}")

    shutil.copy2(start_file, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infinite loop: prompt -> image -> next prompt -> next image, until killed."
    )
    parser.add_argument("directory", type=Path, help="Directory containing prompt_N.txt files.")
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--start-from",
        type=int,
        default=None,
        help="Index of the first prompt to read (default: highest numbered prompt or image).",
    )
    start_group.add_argument(
        "--start-file",
        type=Path,
        help="Copy a .txt or image file into an empty directory as prompt_0.txt or image_0.png.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device. Default: auto",
    )
    parser.add_argument(
        "--image-model",
        default=create.DEFAULT_IMAGE_MODEL,
        help="Diffusers model for text-to-image generation.",
    )
    parser.add_argument(
        "--vision-model",
        default=describe.DEFAULT_VISION_MODEL,
        help="Vision model name for image-to-prompt generation.",
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

    image_model = args.image_model
    vision_model = args.vision_model

    work_dir = args.directory.resolve()
    if not work_dir.is_dir():
        print(f"error: not a directory: {work_dir}", file=sys.stderr)
        sys.exit(1)

    if args.start_file is not None:
        try:
            start_path = copy_start_file(args.start_file.resolve(), work_dir)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Copied {args.start_file} to {start_path}")
        idx = 0
    else:
        idx = args.start_from if args.start_from is not None else find_start_index(work_dir)

    running = True

    def handle_signal(signum: int, frame: object) -> None:
        nonlocal running
        if running:
            print("\nReceived signal, shutting down after current iteration...")
            running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    device = args.device
    remaining_steps = args.steps

    print("Loading vision model (describe)...")
    vlm_device, vlm_dtype = describe.resolve_device_and_dtype(device)
    vlm = describe.load_vlm_pipeline(
        model_id=vision_model,
        device=vlm_device,
        dtype=vlm_dtype,
    )
    print(f"Vision model loaded: {vision_model} on {vlm_device}")

    print("Loading image model (create)...")
    create_device, create_dtype = create.resolve_device_and_dtype(device, image_model)

    pipe = create.load_text2image_pipeline(
        model_id=image_model,
        device=create_device,
        dtype=create_dtype,
    )
    print(f"Image model loaded: {image_model} on {create_device}")

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
                steps=create.DEFAULT_STEPS,
                guidance_scale=create.DEFAULT_GUIDANCE_SCALE,
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
