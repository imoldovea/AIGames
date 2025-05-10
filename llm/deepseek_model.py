# deepseek_model.py

import configparser
import logging

import https
from openai import OpenAI

from gpt_model import GPTModel  # Assuming gpt_model.py is in Python's search path or same dir

API_URL = "https://api.deepseek.com"
PARAMETERS_FILE = "config.properties"
SECRETS_FILE = "secrets.properties"

config = configparser.ConfigParser()
config.read(PARAMETERS_FILE)

secrets = configparser.ConfigParser()
secrets.read(SECRETS_FILE)


# It's good practice for gpt_model.py to handle these configurations.
# class GPTModel:
#     def __init__(self, system_prompt):
#         self.system_prompt = system_prompt
#         self.model = config.get('LLM', 'model_name', fallback="deepseek-chat")
#         self.max_tokens = config.getint('LLM', 'max_tokens', fallback=1024)
#         self.temperature = config.getfloat('LLM', 'temperature', fallback=0.7)

class DeepSeekAIModel(GPTModel):
    def __init__(self, system_prompt):
        super().__init__(system_prompt)

        logging.info("Initializing DeepSeekModel")
        self.provider = "DeepSeek"
        self.name = "DeepSeek"
        self.model = "deepseek-chat"
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

        try:
            # Same section name as in the secrets.properties file
            self.api_key = secrets.get('DeepSeek', 'api_key')
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            logging.error(
                f"API key for DeepSeek not found in {SECRETS_FILE}. Section 'DeepSeek', option 'api_key'. Error: {e}")
            # Re-raise as a ValueError to indicate a critical configuration issue
            raise ValueError(f"API key for DeepSeek not found in {SECRETS_FILE}. Please check configuration.") from e

        self.client = OpenAI(api_key=self.api_key, base_url=API_URL)

        logging.info("DeepSeek Model initialized with system prompt:\n%s", self.system_prompt)

    def generate_response(self, prompt: str) -> str:
        try:
            # Send the system prompt along with the actual user prompt
            response = self.client.chat.completions.create(
                model=self.model,  # This should be set by GPTModel's __init__
                max_tokens=self.max_tokens,  # This should be set by GPTModel's __init__
                temperature=self.temperature,  # This should be set by GPTModel's __init__
                messages=[
                    {"role": "system", "content": f"{self.system_prompt}"},
                    {"role": "user", "content": f"{prompt}"}
                ]
            )
            assistant_content = response.choices[0].message.content.strip()
            return assistant_content
        except Exception as e:
            logging.error(f"A model error occurred while generating response: {e}", exc_info=True)
            return f"A model error occurred: {e}"  # Consider re-raising or returning a more specific error object

    def get_provider(self) -> str:
        return self.provider

    def get_model(self) -> str:
        return self.model  # This should reflect the actual model name used (e.g., "deepseek-chat")

    def get_provider_model(self) -> str:
        return f"{self.provider}/{self.model}"

    def get_prompt(self) -> str:
        return self.system_prompt

    def get_name(self) -> str:
        return self.name


# Example usage
if __name__ == "__main__":
    # Configure basic logging for the example to see output from the class
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define a system prompt for the example
    example_system_prompt = "You are a helpful AI assistant that provides concise answers."
    # The user prompt for the example
    user_query_prompt = '{"local_context": {"north": "wall", "south": "open", "east": "open", "west": "wall"}, "exit_reached": false}'
    # A more general user prompt for testing:
    # user_query_prompt = "What is the capital of France?"

    try:
        # Pass the required system_prompt argument
        chat_model = DeepSeekAIModel(system_prompt=example_system_prompt)
        response = chat_model.generate_response(user_query_prompt)
        print(f"\nResponse from DeepSeek Model:\n{response}")
    except ValueError as ve:  # Catch the ValueError from missing API key or other init issues
        logging.error(f"Initialization Error: {ve}")
        print(f"Error during initialization: {ve}")
    except Exception as e:  # Catch any other unexpected errors during the example run
        logging.error(f"An unexpected error occurred in example usage: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
