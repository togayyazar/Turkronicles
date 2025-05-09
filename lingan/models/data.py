import pickle
from typing import Dict, Iterable, Optional, Tuple, List, Hashable, Sequence
import numpy as np

from .definitions import Data, VData
from scipy.sparse import dok_matrix


class Vocabulary[T:Hashable](Data):

    def __init__(self, keys: Iterable[T] = None, max_size: Optional[int] = None,
                 min_frequency: Optional[int] = None):
        self._frequency_dist: dict[T, int] = dict()
        self._key2idx: Dict[T, int] = dict()
        self._idx2key: list[T] = list()
        self._index = 0
        self.min_frequency = 0 if not min_frequency else min_frequency
        self.max_size = max_size if max_size else np.inf

        if keys:
            for w in keys:
                self.add(w)

    def exist(self, word: str):
        return word in self._key2idx

    @property
    def key2idx(self):
        return self._key2idx

    @property
    def idx2key(self):
        return self._idx2key

    @property
    def frequency_dist(self) -> Dict[str, int]:
        return self._frequency_dist

    @property
    def size(self):
        return len(self.key2idx)

    def token_count(self):
        return sum(self.frequency_dist.values())

    def add(self, key: str, freq=None):
        if not freq:
            freq = 1

        if key not in self.frequency_dist:
            self.frequency_dist[key] = freq
            self.key2idx[key] = self._index
            self._idx2key.append(key)
            self._index += 1
        else:
            self.frequency_dist[key] += freq

    def frequency(self, key: str):
        if not self.exist(key):
            return 0
        return self.frequency_dist[key]

    def index(self, key):
        if not self.exist(key):
            return -1

        return self.key2idx[key]

    def key(self, index):
        return self.idx2key[index]

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        return vocab

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __contains__(self, word):
        return word in self._key2idx

    def __len__(self):
        return len(self._idx2key)

    def __iter__(self):
        return iter(self._idx2key)

    def intersection(self, v_other: 'Vocabulary') -> Tuple['Vocabulary', 'Vocabulary']:
        common_words = list(filter(self.__contains__, v_other))
        v = Vocabulary()
        v_o = Vocabulary()
        for word in common_words:
            v.add(word, self.frequency(word))
            v_o.add(word, v.frequency(word))

        return v, v_o


class Embeddings(VData):

    def __init__(self, W: np.ndarray = None, vocabulary: Vocabulary = None):
        self.W = W
        self._dimension = None
        self.vocabulary = vocabulary

        if W is not None:
            self._dimension = W.shape[1]

    def vector(self, word: str) -> np.ndarray:
        idx = self.vocabulary.key2idx[word]
        return self.W[idx]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @property
    def dimension(self):
        return self._dimension

    @property
    def N(self):
        return self.W.shape[0]

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            e = pickle.load(f)
        return e

    def intersection(self, E: 'Embeddings') -> Tuple['Embeddings', 'Embeddings']:
        v, v_other = self.vocabulary.intersection(E.vocabulary)
        v_length = len(v)
        dim = self.dimension
        W = np.zeros(shape=(v_length, dim), dtype=self.W.dtype)
        W_o = np.zeros(shape=(v_length, dim), dtype=self.W.dtype)
        for word, index in v.key2idx:
            W[index] = self.vector(word)
            W_o[index] = E.vector(word)

        emb = Embeddings(vocabulary=v)
        emb_o = Embeddings(vocabulary=v_other)
        emb.W = W
        emb_o.W = W_o
        return emb, emb_o


class CooccurenceMatrix(Data):
    def __init__(self, word2idx: Dict[str, int], idx2word: List[str], F: dok_matrix = None):
        self.word2idx = word2idx
        self.idx2word = idx2word
        if F is None:
            F: dok_matrix = dok_matrix((len(idx2word), len(idx2word)), dtype=np.int32)
        self.F = F

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            e = pickle.load(f)
        return e

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def increase(self, word1: str, word2: str, freq: int = 1):
        i = self.word2idx[word1]
        j = self.word2idx[word2]

        self.F[i, j] += freq

    def decrease(self, word1: str, word2: str, freq: int = 1):
        i = self.word2idx[word1]
        j = self.word2idx[word2]

        self.F[i, j] -= freq


class NGram[T:Hashable](VData):
    def __init__(self, n: int = None, ngrams: Iterable[Sequence[T]] = None, vocabulary: Vocabulary[T] = None):
        if not n and not ngrams:
            raise ValueError("n and ngrams cannot be None at the same time!")
        if ngrams and vocabulary:
            raise ValueError("Only one of the arguments (vocabulary or ngrams) should be provided!")

        if not n and ngrams:
            n = len(next(iter(ngrams)))

        if not vocabulary:
            vocabulary = Vocabulary[T]()

        self.n = n
        self.vocabulary = vocabulary
        self.collocations: dict[T, dict[T, dict]] = dict()

        if ngrams:
            for ngram in ngrams:
                self.add(ngram)

    def add(self, ngram: Sequence[T]):
        if len(ngram) != self.n:
            raise ValueError(f'Expected ngram length {self.n} but {len(ngram)} given')

        self.vocabulary.add(ngram)
        for i, key in enumerate(ngram):
            if key not in self.collocations:
                self.collocations[key] = dict()

            for j, other_key in enumerate(ngram):
                if i == j:
                    continue
                if other_key not in self.collocations[key]:
                    self.collocations[key][other_key] = {'frequency': 1}
                else:
                    new_frequency = self.collocations[key][other_key]['frequency'] + 1
                    self.collocations[key][other_key]['frequency'] = new_frequency

    def exist(self, ngram: Sequence[T]) -> bool:
        return self.vocabulary.exist(ngram)

    def get_ngram(self, index: int) -> Sequence[T]:
        return self.vocabulary.key(index)

    def frequency(self, ngram: Sequence[T]) -> int:
        return self.vocabulary.frequency(ngram)

    def co_frequency(self, key: T, other_key: T) -> int:
        return self.collocations[key][other_key].get('frequency', 0)

    def index(self, ngram: Sequence[T]) -> int:
        return self.vocabulary.index(ngram)

    def collocates(self, key: [T]) -> List[T]:
        return list(self.collocations[key].keys())

    @classmethod
    def load(cls, path: str) -> 'NGram':
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        return vocab

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self) -> int:
        return len(self.vocabulary)

    def __iter__(self):
        return iter(self.vocabulary)

    def __repr__(self) -> str:
        return f"NGram(n={self.n})"


class ContextualEmbeddings(VData):
    pass
