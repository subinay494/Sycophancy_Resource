#!/usr/bin/env python3
"""
Simple model downloader - downloads model files without loading into memory.
Just uses huggingface_hub to download files.
"""

__author__ = "Shuvam Banerji Seal"

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
PROJECT_ROOT = Path(__file__).parent.parent
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"📄 Loaded environment from {env_file}")

# Get HF token
HF_TOKEN = os.getenv('HF_TOKEN')

# Model configurations
MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "glm-4.6v-flash": "zai-org/GLM-4.6V-Flash",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}

# Target directory
DOWNLOAD_DIR = PROJECT_ROOT / "downloaded_models"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def download_model(model_name: str, model_id: str):
    """Download a model using huggingface_hub snapshot_download."""
    try:
        from huggingface_hub import snapshot_download
        
        print(f"\n{'='*70}")
        print(f"📥 Downloading: {model_name}")
        print(f"   Model ID: {model_id}")
        print(f"   Target: {DOWNLOAD_DIR / model_name}")
        print(f"{'='*70}\n")
        
        # Download the entire model repository
        snapshot_download(
            repo_id=model_id,
            local_dir=str(DOWNLOAD_DIR / model_name),
            local_dir_use_symlinks=False,
            token=HF_TOKEN if HF_TOKEN else None,
            resume_download=True,
        )
        
        print(f"\n✅ Successfully downloaded {model_name}!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {model_name}: {str(e)}\n")
        return False


def main():
    print("\n" + "="*70)
    print("🚀 Starting Model Downloads")
    print("="*70)
    
    if HF_TOKEN:
        print(f"✓ HF_TOKEN found")
    else:
        print(f"⚠️  Warning: HF_TOKEN not set (required for gated models like Gemma)")
    
    print(f"📁 Download directory: {DOWNLOAD_DIR}")
    print(f"📦 Models to download: {len(MODELS)}")
    print()
    
    results = {}
    for model_name, model_id in MODELS.items():
        results[model_name] = download_model(model_name, model_id)
    
    # Summary
    print("\n" + "="*70)
    print("📊 Download Summary")
    print("="*70)
    
    successful = sum(1 for v in results.values() if v)
    print(f"\n✓ {successful}/{len(results)} models downloaded successfully\n")
    
    for model_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {model_name}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
