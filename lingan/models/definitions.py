from __future__ import annotations

from typing import Protocol

DATA_TYPES = {}


class DataMeta(type):
    def __new__(cls, clsname, bases, attrs):
        DATA_TYPES[clsname] = cls

        return super().__new__(cls, clsname, bases, attrs)


class DataProtocol(type(Protocol), type(DataMeta)):
    pass


class Data(Protocol, metaclass=DataProtocol):
    value: object


class VData(Data, Protocol):
    vocabulary: "Vocabulary"
