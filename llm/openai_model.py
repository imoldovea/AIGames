# openai_model.py
# OpenAIModel

import configparser
import logging

import openai

from gpt_model import GPTModel

PARAMETERS_FILE = "config.properties"
SECRETS_FILE = "secrets.properties"

config = configparser.ConfigParser()
config.read(PARAMETERS_FILE)

secrets = configparser.ConfigParser()
secrets.read(SECRETS_FILE)

class OpenAIModel(GPTModel):
    def __init__(self):
        super().__init__()

        logging.info("Initializing OpenAIModel")
        self.provider = "OpenAI"
        self.name = "OpenAI"

        # Same section name as in the secrets.properties file
        self.api_key = secrets.get('ChatGPT', 'api_key')
        openai.api_key = self.api_key

    def generate_response(self, prompt: str) -> str:
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
    prompt = '{"local_context": {"north": "wall", "south": "open", "east": "open", "west": "wall"}, "exit_reached": false}'

    chat_model = OpenAIModel()
    response = chat_model.generate_response(prompt)
    print(f"Response: {response}")