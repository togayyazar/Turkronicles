from typing import Protocol
from abc import abstractmethod
from ..containers.components import DiachronicCorpus, Corpus


class Operation(Protocol):
    def on_diachronic(self, d: DiachronicCorpus):
        ...

    def on_synchronic(self, c: Corpus):
        ...
