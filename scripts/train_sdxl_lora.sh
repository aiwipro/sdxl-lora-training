#!/bin/bash

# SDXL LoRA Training Script
# This script trains a LoRA model using Kohya SS

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/python" ]; then
    # Use venv python directly
    export PATH="$PROJECT_ROOT/venv/bin:$PATH"
fi

# Configuration
KOHYA_SS_DIR="${KOHYA_SS_DIR:-$PROJECT_ROOT/kohya_ss}"  # Path to kohya_ss repository
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/sdxl-base}"  # Path to SDXL base model
# For kohya_ss, train_data_dir should point to the parent directory containing dataset folders
# So if your dataset is in datasets/your_dataset/, point to datasets/
DATASET_PARENT="${DATASET_PARENT:-$PROJECT_ROOT/datasets}"  # Parent directory containing dataset folders
DATASET_NAME="${DATASET_NAME:-your_dataset}"  # Name of your dataset folder
DATASET_DIR="$DATASET_PARENT/$DATASET_NAME"  # Full path to dataset
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/models}"  # Output directory for trained models
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/training_config.json}"  # Training config

# Training parameters (override with environment variables)
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_STEPS="${MAX_STEPS:-2000}"
NETWORK_DIM="${NETWORK_DIM:-16}"
NETWORK_ALPHA="${NETWORK_ALPHA:-8}"
RESOLUTION="${RESOLUTION:-1024,1024}"

# Check if kohya_ss directory exists
if [ ! -d "$KOHYA_SS_DIR" ]; then
    echo "Error: kohya_ss directory not found at $KOHYA_SS_DIR"
    echo "Please set KOHYA_SS_DIR environment variable or clone kohya_ss:"
    echo "  git clone https://github.com/bmaltais/kohya_ss.git"
    exit 1
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model path not found: $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable to your SDXL model"
fi

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    echo "Please set DATASET_DIR environment variable to your dataset path"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check for sdxl_train_network.py in sd-scripts (SDXL-specific script)
TRAIN_SCRIPT_DIR="$KOHYA_SS_DIR/sd-scripts"
TRAIN_SCRIPT="$TRAIN_SCRIPT_DIR/sdxl_train_network.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: sdxl_train_network.py not found at $TRAIN_SCRIPT"
    echo "Please ensure kohya_ss and sd-scripts are properly set up"
    exit 1
fi

# Change to sd-scripts directory where sdxl_train_network.py is located
cd "$TRAIN_SCRIPT_DIR"

# Run training
echo "Starting SDXL LoRA training..."
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Max steps: $MAX_STEPS"
echo ""

# Detect platform for Mac-specific optimizations
PLATFORM=$(uname -m)
OS_TYPE=$(uname -s)
IS_MAC=false

if [[ "$PLATFORM" == "arm64" ]] && [[ "$OS_TYPE" == "Darwin" ]]; then
    IS_MAC=true
    echo "Apple Silicon detected - using MPS backend"
elif [[ "$PLATFORM" == "x86_64" ]] && [[ "$OS_TYPE" == "Darwin" ]]; then
    IS_MAC=true
    echo "Intel Mac detected"
fi

# Use venv python if available
if [ -f "$PROJECT_ROOT/venv/bin/python" ]; then
    PYTHON_CMD="$PROJECT_ROOT/venv/bin/python"
else
    PYTHON_CMD="python"
fi

# Build accelerate command
if [ "$IS_MAC" = true ]; then
    NUM_CPU_THREADS=$(sysctl -n hw.ncpu)
    echo "Using $NUM_CPU_THREADS CPU threads"
else
    NUM_CPU_THREADS=4
fi

# Build training command arguments
TRAIN_ARGS=(
    --pretrained_model_name_or_path="$MODEL_PATH"
    --train_data_dir="$DATASET_PARENT"
    --in_json="$DATASET_DIR/metadata.json"
    --output_dir="$OUTPUT_DIR"
    --output_name="sdxl_lora"
    --save_model_as="safetensors"
    --save_precision="fp16"
    --seed=42
    --resolution="$RESOLUTION"
    --enable_bucket
    --min_bucket_reso=256
    --max_bucket_reso=2048
    --train_batch_size="$BATCH_SIZE"
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-4}"
    --max_train_steps="$MAX_STEPS"
    --learning_rate="$LEARNING_RATE"
    --lr_scheduler="cosine_with_restarts"
    --lr_warmup_steps=100
    --mixed_precision="fp16"
    --save_every_n_steps=500
    --save_last_n_steps=3
    --network_module="networks.lora"
    --network_dim="$NETWORK_DIM"
    --network_alpha="$NETWORK_ALPHA"
    --network_args="enable_lycoris=false"
    --network_args="conv_dim=1"
    --network_args="conv_alpha=1"
    --network_args="algo=lora"
    --cache_latents
    --cache_latents_to_disk
    --clip_skip=2
    --max_token_length=225
    --persistent_data_loader_workers
    --max_data_loader_n_workers=2
)

# Mac-specific: Use AdamW instead of AdamW8bit (bitsandbytes not available on Mac)
if [ "$IS_MAC" = true ]; then
    TRAIN_ARGS+=(--optimizer_type="AdamW")
    echo "Using AdamW optimizer (AdamW8bit not available on Mac)"
else
    TRAIN_ARGS+=(--optimizer_type="AdamW8bit")
fi

# Only add xformers flag on non-Mac systems
if [ "$IS_MAC" = false ]; then
    TRAIN_ARGS+=(--xformers)
fi

# Use venv python if available
if [ -f "$PROJECT_ROOT/venv/bin/python" ]; then
    PYTHON_CMD="$PROJECT_ROOT/venv/bin/python"
else
    PYTHON_CMD="python"
fi

# Run training (sdxl_train_network.py is in current directory now)
$PYTHON_CMD -m accelerate.commands.launch --num_cpu_threads_per_process=$NUM_CPU_THREADS sdxl_train_network.py "${TRAIN_ARGS[@]}"

echo ""
echo "Training completed! Check outputs in: $OUTPUT_DIR"
