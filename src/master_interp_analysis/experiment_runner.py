"""
Experiment runner configuration for LLM sycophancy analysis.
Defines models and configurations.
"""

__author__ = "Shuvam Banerji Seal"

from pathlib import Path

# Local models directory
LOCAL_MODELS_DIR = Path(__file__).parent.parent.parent / "downloaded_models"

# Model configurations
EXPERIMENT_MODELS = {
    # Qwen Models
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "requires_auth": False,
        "local": False,
    },
    
    # Gemma Models
    "gemma-3-12b-it": {
        "model_id": "google/gemma-3-12b-it",
        "requires_auth": True,  # Gemma models typically require auth
        "local": False,
    },
    
    # GLM Models
    "glm-4.6v-flash": {
        "model_id": "zai-org/GLM-4.6V-Flash",
        "requires_auth": False,
        "local": False,
    },
    
    # GPT-OSS Models
    "gpt-oss-20b": {
        "model_id": "openai/gpt-oss-20b",
        "requires_auth": False,
        "local": False,
    },
}
