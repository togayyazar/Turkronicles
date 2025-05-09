import abc
from typing import Protocol, overload, Union, Tuple
from ..containers.components import DiachronicCorpus, Corpus


class OperationBase(Protocol):
    ...


class SynchronicOperation(OperationBase):
    def on_synchronic(self, c: Corpus):
        ...


class DiachronicOperation(OperationBase):
    def __init__(self, time_range: Union[slice, None] = None, t1: Union[Tuple[int, int]] = None,
                 t2: Union[Tuple[int, int]] = None):
        if time_range is None:
            time_range = slice(None, None, None)
        self.time_range = time_range
        self.t1 = t1
        self.t2 = t2

    def on_diachronic(self, d: DiachronicCorpus):
        ...


class Operation(SynchronicOperation, DiachronicOperation):
    def __init__(self, time_range: Union[slice, None] = None, t1: Union[Tuple[int, int]] = None,
                 t2: Union[Tuple[int, int]] = None):
        super().__init__(time_range=time_range, t1=t1, t2=t2)

