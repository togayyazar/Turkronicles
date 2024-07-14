from typing import Protocol
from abc import abstractmethod
from ..containers.components import DiachronicCorpus, Corpus


class Operation(Protocol):
    @abstractmethod
    def on_diachronic(self, d: DiachronicCorpus):
        ...

    @abstractmethod
    def on_synchronic(self, c: Corpus):
        ...
