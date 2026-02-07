# SDXL LoRA Training with Kohya SS

Train Stable Diffusion XL (SDXL) LoRA models locally using [Kohya SS](https://github.com/bmaltais/kohya_ss).

## Prerequisites

- Python 3.10+
- Git
- A supported GPU:
  - **Apple Silicon Mac** -- uses Metal Performance Shaders (MPS)
  - **NVIDIA GPU** -- CUDA-capable, 8GB+ VRAM

## 1. Setup

```bash
bash setup.sh
```

This clones Kohya SS, creates a virtualenv, detects your platform, and installs the right PyTorch + dependencies.

Then activate the environment:

```bash
source venv/bin/activate
```

<details>
<summary>Manual installation (if you prefer not to use the setup script)</summary>

**Apple Silicon Mac:**
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
```

**NVIDIA GPU:**
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install xformers  # optional but recommended
```
</details>

## 2. Download the SDXL Base Model

```bash
pip install huggingface-hub
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir ./models/sdxl-base
```

Or download manually from https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 and place the files in `./models/sdxl-base`.

## 3. Prepare Your Dataset

Place your training images in `datasets/your_dataset/`:

```bash
mkdir -p datasets/your_dataset
# copy your images into datasets/your_dataset/
```

Then generate metadata:

```bash
python scripts/prepare_dataset.py \
    --input_dir datasets/your_dataset \
    --create_metadata \
    --analyze
```

This validates images and creates a `metadata.jsonl` file. Each line maps an image to its caption:

```json
{"file_name": "image1.jpg", "text": "a photo of my_subject"}
{"file_name": "image2.jpg", "text": "my_subject in a different setting"}
```

Edit the captions as needed. Use a consistent **trigger word** (e.g. `my_subject`) so the LoRA learns to associate it with your subject.

## 4. Train

Set your paths and run the training script:

```bash
export MODEL_PATH="./models/sdxl-base"
export DATASET_DIR="./datasets/your_dataset"
export OUTPUT_DIR="./outputs/models"

bash scripts/train_sdxl_lora.sh
```

The script auto-detects Mac vs NVIDIA and sets appropriate flags.

### Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Resolution | 1024x1024 | SDXL standard |
| Batch size | 1 | Increase if you have enough VRAM/RAM |
| Learning rate | 1e-4 | Lower (5e-5) if overfitting |
| Steps | 2000 | 1000-5000 depending on dataset size |
| LoRA rank | 16 | 4-8 simple concepts, 32+ complex ones |

These can be adjusted in `configs/training_config.json` or `configs/sdxl_lora_config.toml`.

## 5. Use Your LoRA

Your trained model is saved at `outputs/models/sdxl_lora.safetensors`, with checkpoints at regular intervals.

Always include your **trigger word** in prompts.

### Automatic1111 WebUI

1. Copy `sdxl_lora.safetensors` to `stable-diffusion-webui/models/Lora/`
2. Load the SDXL base model
3. Click "Show extra networks" > "Lora" tab > select your LoRA
4. Use your trigger word in the prompt

### ComfyUI

1. Copy `sdxl_lora.safetensors` to `ComfyUI/models/loras/`
2. Add a "Load LoRA" node, select your file, set strength (0.8-1.0)
3. Use your trigger word in the prompt

### Diffusers (Python)

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)
pipe.load_lora_weights("outputs/models/sdxl_lora.safetensors")

image = pipe("my_subject, professional photography, high quality", num_inference_steps=30).images[0]
image.save("output.png")
```

### LoRA weight tips

- `0.5` -- moderate effect
- `0.8` -- slightly less intense
- `1.0` -- full strength (default)
- `>1.2` -- may cause artifacts

## Troubleshooting

### Out of memory

- Set `--train_batch_size=1`
- Add `--gradient_checkpointing`
- Enable `--cache_latents` and `--cache_latents_to_disk`

### Slow training

- Ensure `--cache_latents` is enabled
- Use `--mixed_precision=fp16`
- **Mac**: xformers is not available; MPS is used automatically
- **NVIDIA**: install xformers (`pip install xformers`)
- Disable gradient checkpointing if you have RAM to spare

### Poor results

- Increase training steps
- Lower the learning rate
- Improve dataset quality and diversity
- Increase LoRA rank

### Mac-specific

- **MPS errors**: reinstall PyTorch (`pip install torch torchvision`)
- **Optimizer errors**: use `AdamW` instead of `AdamW8bit` (bitsandbytes is not available on Mac)

## Project Structure

```
sdxl-lora/
├── README.md
├── setup.sh
├── requirements.txt
├── configs/
│   ├── sdxl_lora_config.toml
│   └── training_config.json
├── scripts/
│   ├── train_sdxl_lora.sh
│   └── prepare_dataset.py
├── datasets/
│   └── your_dataset/
├── models/
│   └── sdxl-base/
└── outputs/
    └── models/
```

## Resources

- [Kohya SS](https://github.com/bmaltais/kohya_ss)
- [SDXL Paper](https://arxiv.org/abs/2307.01952)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## License

Follow the licenses of:
- Kohya SS (Apache 2.0)
- Stable Diffusion XL (CreativeML Open RAIL-M)
- Your training data
