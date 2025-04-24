# gpt_model.py

import configparser
from abc import ABC, abstractmethod


class GPTModel(ABC):
    def __init__(self, system_prompt):
        config = configparser.ConfigParser()
        config.read('config.properties')

        # self.system_prompt = config.get('LLM', 'system_prompt')
        self.system_prompt = system_prompt
        self.algorithm = config.get('LLM', 'algorithm')
        self.system_prompt = self.system_prompt.replace("{algorithm}", self.algorithm)

        self.model = config.get('LLM', 'model_name')
        self.temperature = config.getfloat('LLM', 'temperature')
        self.max_tokens = config.getint('LLM', 'max_tokens')
        self.provider = config.get('LLM', 'provider')

    @abstractmethod
    def generate_response(self, prompt) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_provider(self) -> str:
        pass

    @abstractmethod
    def get_model(self) -> str:
        pass

    @abstractmethod
    def get_provider_model(self) -> str:
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        pass
