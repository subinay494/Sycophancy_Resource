#!/usr/bin/env python3
"""Quick test to verify GPU 1 CUDA functionality."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    torch.cuda.init()
    print(f"CUDA init OK")
    t = torch.zeros(1, device="cuda")
    print(f"Tensor on GPU: {t.device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA NOT AVAILABLE - this is the problem!")
