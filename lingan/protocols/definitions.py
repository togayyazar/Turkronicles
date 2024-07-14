from abc import abstractmethod
from typing import Protocol


class Serializable(Protocol):
    @abstractmethod
    def save(cls, path: str):
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        ...


class JSONSerializable(Serializable, Protocol):
    @abstractmethod
    def json(self):
        raise NotImplementedError
