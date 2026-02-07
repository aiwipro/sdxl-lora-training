#!/bin/bash

# Setup script for SDXL LoRA training environment

set -e

echo "SDXL LoRA Training Setup"
echo "========================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if kohya_ss is cloned
if [ ! -d "kohya_ss" ]; then
    echo ""
    echo "Kohya SS not found. Cloning repository..."
    git clone https://github.com/bmaltais/kohya_ss.git
    echo "Kohya SS cloned successfully!"
else
    echo "Kohya SS directory found."
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created!"
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Detect platform
PLATFORM=$(uname -m)
OS_TYPE=$(uname -s)

echo ""
echo "Detected platform: $OS_TYPE $PLATFORM"

# Install PyTorch based on platform
if [[ "$PLATFORM" == "arm64" ]] && [[ "$OS_TYPE" == "Darwin" ]]; then
    echo ""
    echo "Apple Silicon detected! Installing PyTorch with MPS (Metal) support..."
    pip install torch torchvision
    echo "PyTorch with MPS support installed!"
elif [[ "$PLATFORM" == "x86_64" ]] && [[ "$OS_TYPE" == "Darwin" ]]; then
    echo ""
    echo "Intel Mac detected! Installing PyTorch..."
    pip install torch torchvision
else
    echo ""
    echo "Installing PyTorch with CUDA 11.8 support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Install kohya_ss requirements if it exists
if [ -d "kohya_ss" ] && [ -f "kohya_ss/requirements.txt" ]; then
    echo ""
    echo "Installing kohya_ss requirements..."
    
    # First, ensure sd-scripts submodule is initialized
    if [ ! -d "kohya_ss/sd-scripts" ]; then
        echo "Initializing sd-scripts submodule..."
        cd kohya_ss
        git submodule update --init --recursive
        cd ..
    fi
    
    # Install scipy separately first (may have version conflicts)
    echo "Installing scipy (may use newer version if exact version unavailable)..."
    pip install scipy || echo "Note: scipy installation had issues, continuing..."
    
    # Install requirements from kohya_ss directory so relative paths work
    echo "Installing kohya_ss requirements (excluding scipy to avoid conflicts)..."
    cd kohya_ss
    # Install all requirements except scipy (we'll use a compatible version)
    grep -v "^scipy==" requirements.txt | grep -v "^#.*scipy" | pip install -r /dev/stdin || {
        echo "Warning: Some kohya_ss requirements failed to install."
        echo "This may be okay - core training functionality should still work."
    }
    cd ..
fi

# Note about xformers on Mac
if [[ "$PLATFORM" == "arm64" ]] && [[ "$OS_TYPE" == "Darwin" ]]; then
    echo ""
    echo "Note: xformers is not available on Apple Silicon. Using PyTorch's native attention instead."
elif [[ "$PLATFORM" == "x86_64" ]] && [[ "$OS_TYPE" == "Darwin" ]]; then
    echo ""
    echo "Note: xformers may not work on Intel Mac. Using PyTorch's native attention instead."
else
    echo ""
    echo "Attempting to install xformers (for memory efficiency)..."
    pip install xformers || echo "Warning: xformers installation failed. You can install it later if needed."
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Download SDXL base model and set MODEL_PATH in your config"
echo "3. Prepare your dataset in datasets/your_dataset/"
echo "4. Run: python scripts/prepare_dataset.py --input_dir datasets/your_dataset --create_metadata --analyze"
echo "5. Start training: bash scripts/train_sdxl_lora.sh"
