from typing import Protocol


class Adapter(Protocol):
    def load(self, path: str = None):
        ...

    def save(self, path: str = None):
        ...
