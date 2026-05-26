# Recreate

This project now uses a two-step local workflow:

- `describe.py`: generate a reconstruction prompt from an input image.
- `create.py`: generate an image from a prompt text file.

Both scripts use local Hugging Face models.

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
python describe.py input.png --vision-model-option 2
```

Useful options:

- `--output PATH`
- `--print` (print to stdout instead of saving a prompt file)
- `--force`
- `--vision-model MODEL`
- `--vision-model-option {1,2,3}`
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
python create.py input_recreated.prompt.txt --image-model-option 1 --width 768 --height 512
```

Useful options:

- `--output PATH`
- `--force`
- `--stdin`
- `--ask`
- `--ask-multi`
- `--image-model MODEL`
- `--image-model-option {1,2,3,4,5}`
- `--seed INT`
- `--steps INT` (default `8`)
- `--guidance-scale FLOAT` (default `6.0`)
- `--width INT` and `--height INT` (must be multiples of `8`)
- `--device auto|cuda|mps|cpu`

## Notes

- There is no img2img mode in this workflow.
- Existing files are not overwritten unless `--force` is provided.
