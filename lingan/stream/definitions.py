from abc import abstractmethod
from typing import Protocol

from typing import Iterable, Iterator, Callable, Protocol, TypeVar, Optional, List, Generic


class StreamP[T](Protocol):
    @abstractmethod
    def map[R](self, mapper: Callable[[T], R]) -> 'StreamP[R]': ...

    @abstractmethod
    def filter(self, predicate: Callable[[T], bool]) -> 'StreamP[T]': ...

    @abstractmethod
    def limit(self, n: int) -> 'StreamP[T]': ...

    @abstractmethod
    def for_each(self, action: Callable[[T], None]) -> None: ...

    @abstractmethod
    def to_list(self) -> List[T]: ...

    @abstractmethod
    def close(self): ...
