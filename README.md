# Recreate

Recreate is a local image-reconstruction workflow built around two scripts:

- `describe.py`: generate a text prompt that describes an input image.
- `create.py`: generate an image from a prompt file, stdin, or interactive input.

This software does not use an image-to-image process.
The model generating the image only have the text as the reference to generate the image.

Both steps run with local Hugging Face models and do not require a remote API.

## Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The first real run downloads selected model weights. Defaults:

- Vision model: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Image model: `stabilityai/sdxl-turbo`

For NVIDIA GPUs, install the PyTorch build that matches your CUDA version first.

## 1) Describe: image -> prompt

```bash
python describe.py <path_to_input_image>
```

Default output prompt path:

- `<input_stem>_recreated.prompt.txt`

Example:

```bash
python describe.py input.png --vision-model-preset 2
```

Useful options:

- `--output PATH`
- `--print` (print to stdout instead of saving a prompt file)
- `--force`
- `--vision-model MODEL`
- `--vision-model-preset {1,2,3,4}`
- `--max-size INT` (default `512`)
- `--device auto|cuda|mps|cpu`

## 2) Create: prompt -> image

```bash
python create.py <path_to_prompt_file>
```

You can also provide the prompt without a file:

```bash
echo "A cinematic portrait of a fox in snow" | python create.py --stdin
python create.py --ask
python create.py --ask-multi
```

With `--ask-multi`, the script keeps asking for prompts until you submit an empty one.
If an output filename already exists, it asks whether to overwrite (`y/N`); if not, it asks for a new filename.

Default output image path:

- If prompt is `name.prompt.txt`, output is `name.png`
- For `*_recreated.prompt.txt`, output becomes `*_recreated.png`
- For `--stdin` or `--ask` without `--output`, output is `generated.png`
- For `--ask-multi` without `--output`, output names are `generated.png`, `generated_002.png`, `generated_003.png`, ...

Example:

```bash
python create.py input_recreated.prompt.txt --image-model-preset 1 --width 768 --height 512
```

Useful options:

- `--output PATH`
- `--force`
- `--stdin`
- `--ask`
- `--ask-multi`
- `--image-model MODEL`
- `--image-model-preset {1,2,3,4,5}`
- `--seed INT`
- `--steps INT` (default `8`)
- `--guidance-scale FLOAT` (default `6.0`)
- `--width INT` and `--height INT` (must be multiples of `8`)
- `--device auto|cuda|mps|cpu`

Existing files are not overwritten unless `--force` is provided.

## 3) Loop: continuous prompt -> image -> prompt cycle

```bash
python loop.py <directory> [options]
```

This script implements an infinite feedback loop:
1. Reads `prompt_N.txt` files from the directory.
2. Generates `image_N.png` from each prompt.
3. Describes the generated image to create `prompt_(N+1).txt`.
4. Repeats until stopped (Ctrl+C).

Useful for exploring prompt evolution and image generation drift over multiple iterations.

Useful options:

- `--start-from INT` (default: `0`)
- `--steps INT` (number of iterations before exiting; default: run indefinitely)
- `--image-model MODEL` and `--image-model-preset {1,2,3,4,5}`
- `--vision-model MODEL` and `--vision-model-preset {1,2,3,4}`
- `--width INT` and `--height INT`
- `--device auto|cuda|mps|cpu`

## Model Presets

Both `describe.py` and `create.py` support preset model options for quick access to alternative models.

### Vision Models (image-to-prompt)

Use `--vision-model-preset {1,2,3,4}` or `--vision-model MODEL`:

1. `openbmb/MiniCPM-V-4.6` (default) - Image-text-to-text model. Good balance of speed and quality.
2. `HuggingFaceTB/SmolVLM-256M-Instruct` - Smallest and fastest local VLM. Best for low-resource setups.
3. `llava-hf/llava-1.5-7b-hf` - Stronger general-purpose 7B vision-language model. Better prompts than default.
4. `Qwen/Qwen2.5-VL-7B-Instruct` - Largest and strongest option. Most detailed descriptions, highest memory requirement.

### Image Models (text-to-image)

Use `--image-model-preset {1,2,3,4,5}` or `--image-model MODEL`:

1. `Tongyi-MAI/Z-Image-Turbo` (default) - Fastest Z-Image variant. Quick generation with reasonable quality.
2. `RunDiffusion/Juggernaut-Z-Image` - Cinematic, sharp, and balanced. Good for realistic outputs.
3. `stabilityai/sdxl-turbo` - Fastest SDXL option. Quick recreations, slightly lower quality than base SDXL.
4. `stabilityai/stable-diffusion-xl-base-1.0` - Higher-quality SDXL base model. Better results, slower generation.
5. `stabilityai/stable-diffusion-3-medium-diffusers` - Most capable preset. Best quality, heaviest resource usage.

## License

See [LICENSE](LICENSE).
