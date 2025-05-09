from typing import Iterable, Callable, List
from .definitions import StreamP


class Stream[T](StreamP[T]):

    def __init__(self, elements: Iterable[T]):
        self._elements = iter(elements)

    def map[R](self, mapper: Callable[[T], R]) -> 'StreamP[R]':
        return Stream(map(mapper, self._elements))

    def filter(self, predicate: Callable[[T], bool]) -> 'StreamP[T]':
        return Stream(filter(predicate, self._elements))

    def limit(self, n: int) -> 'StreamP[T]':
        return Stream(x for i, x in zip(range(n), self._elements))

    def for_each(self, op: Callable[[T], None]) -> None:
        for element in self._elements:
            op(element)
        self.close()

    def to_list(self) -> List[T]:
        self.close()
        return list(self._elements)

    def close(self):
        pass

    def __next__(self):
        return next(self._elements)

    def next(self):
        try:
            return next(self._elements)
        except StopIteration:
            return self.close()


class LineStream(Stream[str]):
    def __init__(self, path: str, language="english"):
        self.path = path
        self.language = language
        self.descriptor = open(self.path)
        super().__init__(self._sentences())

    def _sentences(self) -> Iterable[str]:
        self.descriptor = open(self.path)
        return self.descriptor
