# openai_model.py
# OpenAIModel

import configparser
import logging

import openai

from gpt_model import GPTModel

PARAMETERS_FILE = "config.properties"
SECRETS_FILE = 'secrets.properties'

config = configparser.ConfigParser()
config.read(PARAMETERS_FILE)

secrets = configparser.ConfigParser()
secrets.read(SECRETS_FILE)


class OpenAIModel(GPTModel):
    def __init__(self, name: str):
        logging.info("Initializing OpeAIModel")
        self.provider = "OpenAI"
        self.name = name

        self.api_key = secrets.get('ChatGPT', 'api_key')
        self.model = model
        self.model = secrets.get('LLM', 'model_name')
        self.temperature = config.getfloat('LLM', 'temperature')
        self.max_tokens = config.getint('LLM', 'max_tokens')
        self.system_prompt = config.get('LLM', 'system_prompt')
        openai.api_key = self.api_key

    def generate_response(self, prompt) -> str:
        try:
            # Send the system prompt along with the actual user prompt
            response = openai.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": f"{self.system_prompt}"},
                    {"role": "user", "content": prompt}
                ]
            )
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
    models = ["gpt-3.5-turbo", "gpt-4o-mini"]
    prompt = '{"local_context": {"north": "wall", "south": "open", "east": "open", "west": "wall"}, "exit_reached": false}'

    chat_model = OpenAIModel()
    response = chat_model.generate_response(prompt)
    logging.info(f"Response: {response}")
