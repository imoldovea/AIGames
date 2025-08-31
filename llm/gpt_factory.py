# gpt_factiry.py

# from llm.deepseek_model import DeepSeekModel
# from llm.ollama_model import OllamaModel
from llm.gpt_model import GPTModel
from llm.openai_model import OpenAIModel


# Step 4: Create a Factory for Generating GPT Models
class GPTFactory:
    @staticmethod
    def create_gpt_model(provider: str, system_prompt: str) -> GPTModel:
        if provider == "ChatGPT":
            return OpenAIModel(system_prompt)
        elif provider == "Ollama":
            return OllamaModel(system_prompt)
        elif provider == "DeepSeek":  # Add support for DeepSeek
            return DeepSeekModel(system_prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")
