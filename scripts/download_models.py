#!/usr/bin/env python3
"""
Model downloader script for master_interp_analysis.
Downloads and caches LLM models locally.
"""

__author__ = "Shuvam Banerji Seal"

import os
import sys
import gc
import argparse
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load .env file if it exists
from dotenv import load_dotenv
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"📄 Loaded environment from {env_file}")

from transformers import AutoTokenizer, AutoModelForCausalLM
from master_interp_analysis.experiment_runner import EXPERIMENT_MODELS, LOCAL_MODELS_DIR


def download_model(model_key: str, cache_dir: Path = None, force: bool = False) -> bool:
    """
    Download a model and its tokenizer.
    
    Args:
        model_key: Model key from EXPERIMENT_MODELS (e.g., 'gpt2', 'gemma-2-2b')
        cache_dir: Directory to cache models
        force: Force re-download even if already exists
        
    Returns:
        True if successful, False otherwise
    """
    if model_key not in EXPERIMENT_MODELS:
        print(f"❌ Model '{model_key}' not found.")
        print(f"Available models: {', '.join(EXPERIMENT_MODELS.keys())}")
        return False
    
    config = EXPERIMENT_MODELS[model_key]
    model_id = config["model_id"]
    requires_auth = config.get("requires_auth", False)
    
    cache_dir = cache_dir or LOCAL_MODELS_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading {model_key} ({model_id})...")
    print(f"   Target: {cache_dir}")
    
    try:
        # Get token if needed
        token = None
        if requires_auth:
            token = os.getenv('HF_TOKEN')
            if not token:
                print(f"⚠️  Warning: {model_key} requires auth but HF_TOKEN not set")
                print(f"   Set it with: export HF_TOKEN=your_token")
                return False
        
        # Download tokenizer
        print(f"  📝 Tokenizer...", end='', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
            token=token
        )
        print(" ✓")
        
        # Download model
        print(f"  🧠 Model weights...", end='', flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
            token=token,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        print(" ✓")
        
        # Clean up to free memory
        del model
        del tokenizer
        gc.collect()
        
        print(f"✅ {model_key} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {model_key}: {str(e)}")
        return False


def download_all(cache_dir: Path = None, only_remote: bool = False):
    """Download all models (or only remote ones)."""
    cache_dir = cache_dir or LOCAL_MODELS_DIR
    
    models_to_download = []
    for key, config in EXPERIMENT_MODELS.items():
        if only_remote and config.get("local", False):
            continue  # Skip already local models
        models_to_download.append(key)
    
    print(f"📦 Downloading {len(models_to_download)} models...")
    results = {}
    
    for model_key in models_to_download:
        success = download_model(model_key, cache_dir)
        results[model_key] = success
    
    print("\n" + "=" * 50)
    print("Summary:")
    successful = sum(1 for v in results.values() if v)
    print(f"  ✓ {successful}/{len(results)} models downloaded successfully")
    
    for model_key, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {model_key}")


def list_models():
    """List all available models."""
    print("\n📋 Available Models:")
    print("-" * 70)
    
    print("\n📁 LOCAL (already downloaded):")
    for key, config in EXPERIMENT_MODELS.items():
        if config.get("local", False):
            print(f"  {key:20} - {config['model_id']}")
    
    print("\n🌐 REMOTE (need to download):")
    for key, config in EXPERIMENT_MODELS.items():
        if not config.get("local", False):
            auth = "🔐" if config.get("requires_auth", False) else "  "
            print(f"  {auth}{key:18} - {config['model_id']}")
    
    print("\n🔐 = Requires HF_TOKEN authentication")


def main():
    parser = argparse.ArgumentParser(
        description="Download LLM models for master_interp_analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py --model gpt2
  python download_models.py --model qwen3 
  python download_models.py --all
  python download_models.py --remote-only
  python download_models.py --list
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Model key to download (e.g., gpt2, qwen3, gemma-2-2b)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all models'
    )
    parser.add_argument(
        '--remote-only',
        action='store_true',
        help='Download only remote models (skip already local ones)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=str(LOCAL_MODELS_DIR),
        help=f'Directory to cache models (default: {LOCAL_MODELS_DIR})'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models'
    )
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    
    # List models
    if args.list:
        list_models()
        return
    
    # Download all
    if args.all:
        download_all(cache_dir, only_remote=False)
        return
    
    # Download only remote
    if args.remote_only:
        download_all(cache_dir, only_remote=True)
        return
    
    # Download specific model
    if args.model:
        download_model(args.model, cache_dir)
        return
    
    # No action specified
    parser.print_help()


if __name__ == '__main__':
    main()

