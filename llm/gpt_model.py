# gpt_model.py

from abc import ABC, abstractmethod


class GPTModel(ABC):
    @abstractmethod
    def generate_response(self, prompt, use_history: bool) -> str:
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
