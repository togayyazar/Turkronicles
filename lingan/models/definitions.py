from __future__ import annotations

from typing import Protocol

DATA_TYPES = {}


class DataMeta(type(Protocol)):
    def __new__(cls, clsname, bases, attrs):
        DATA_TYPES[clsname] = cls
        return super().__new__(cls, clsname, bases, attrs)


class Data(Protocol, metaclass=DataMeta):
    ...


class VData(Data, Protocol):
    vocabulary: "Vocabulary"
