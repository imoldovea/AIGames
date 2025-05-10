# openai_model.py
# OpenAIModel

import configparser
import logging

import ollama

from llm.gpt_model import GPTModel

PARAMETERS_FILE = "config.properties"
SECRETS_FILE = "secrets.properties"

config = configparser.ConfigParser()
config.read(PARAMETERS_FILE)

secrets = configparser.ConfigParser()
secrets.read(SECRETS_FILE)


class OllamaAIModel(GPTModel):
    def __init__(self, system_prompt, model: str):
        super().__init__(system_prompt)

        logging.info("Initializing OpenAIModel")
        self.provider = "OLlama"
        self.name = "OLlama"
        self.model = model if model else config.get("LLM", "model_name")

        logging.info("OllamaAIModel initialized with system prompt:\n%s", self.system_prompt)

    def generate_response(self, prompt: str) -> str:
        try:
            # Send the system prompt along with the actual user prompt
            chat_input = [
                {"role": "system", "content": f"{self.system_prompt}"},
                {"role": "assistant", "content": prompt}
            ]
            response = ollama.chat(model=self.model, messages=chat_input)
            response = response['message']['content']
            assistant_content = response.choices[0].message.content.strip()
            return assistant_content
        except Exception as e:
            return f"A model error occurred: {e}"

    def get_provider(self) -> str:
        return self.provider

    def get_model(self) -> str:
        return self.model

    def get_provider_model(self) -> str:
        return f"{self.provider}/{self.model}"

    def get_prompt(self) -> str:
        return self.system_prompt

    def get_name(self) -> str:
        return self.name


# Example usage
if __name__ == "__main__":
    prompt = '{"local_context": {"north": "wall", "south": "open", "east": "open", "west": "wall"}, "exit_reached": false}'

    chat_model = OpenAIModel()
    response = chat_model.generate_response(prompt)
    print(f"Response: {response}")
