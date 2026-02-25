#!/bin/bash
# Wrapper script to run GLM on GPU 1 with proper CUDA isolation

# Set CUDA device BEFORE any Python imports
export CUDA_VISIBLE_DEVICES=1

# Set GCC for flashinfer compilation
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13

# Clear any inherited CUDA state
unset CUDA_DEVICE_ORDER

# Change to project directory
cd /home/iiserk/sbs/llm_sycophancy

# Activate virtual environment
source .venv/bin/activate

# Run GLM with all cases
python scripts/llm_sycophancy_speculative_variants_glm_india.py --all 2>&1 | tee glm_full_gpu1_isolated.log
