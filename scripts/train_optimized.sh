#!/bin/bash

# Optimized training configuration for M2 Max
# Based on testing: batch_size=1 is actually faster on Mac MPS

# Settings optimized for Mac MPS backend
export BATCH_SIZE=1              # Keep at 1 for Mac MPS efficiency
export MAX_STEPS=1500            # Reduce steps (sufficient for LoRA)
export LEARNING_RATE=1e-4
export NETWORK_DIM=16
export NETWORK_ALPHA=8
export GRADIENT_ACCUMULATION_STEPS=4  # Keep higher to maintain effective batch size

echo "Optimized training configuration for Mac MPS:"
echo "  Batch size: $BATCH_SIZE (optimal for MPS)"
echo "  Max steps: $MAX_STEPS"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  No gradient checkpointing (faster)"
echo ""

# Run training
bash scripts/train_sdxl_lora.sh
