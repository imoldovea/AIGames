# gpt_factiry.py

from deepseek_model import DeepSeekModel
from ollama_model import OllamaModel

from gpt_model import GPTModel
from openai_model import OpenAIModel


# Step 4: Create a Factory for Generating GPT Models
class GPTFactory:
    @staticmethod
    def create_gpt_model(provider: str) -> GPTModel:
        if provider == "ChatGPT":
            return OpenAIModel()
        elif provider == "Ollama":
            return OllamaModel()
        elif provider == "DeepSeek":  # Add support for DeepSeek
            return DeepSeekModel()
        else:
            raise ValueError(f"Unknown provider: {provider}")
