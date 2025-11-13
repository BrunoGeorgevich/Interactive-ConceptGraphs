from abc import ABC, abstractmethod


class IInferenceStrategy(ABC):
    """
    Base interface for inference strategies, inspired by AIModule.py from AIMSM.
    Defines the core contract for loading, unloading, and querying model state.
    """

    @abstractmethod
    def load_model(self):
        """
        Loads the model into memory (CPU/GPU).
        """
        pass

    @abstractmethod
    def unload_model(self):
        """
        Releases the model from memory.
        """
        pass

    @abstractmethod
    def get_type(self) -> str:
        """
        Returns the inference type.

        :return: Either 'local' or 'remote'.
        :rtype: str
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Checks if the model is currently loaded.

        :return: True if the model is loaded, False otherwise.
        :rtype: bool
        """
        pass
