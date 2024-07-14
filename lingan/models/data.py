import pickle
from typing import Dict, Iterable, Optional, Union, Tuple
import numpy as np

from .definitions import Data, VData


class Vocabulary(Data):

    def __init__(self, words: Iterable[str] = None, max_size: Optional[int] = None,
                 min_frequency: Optional[int] = None):
        self._frequency_dist: Optional[dict[str, int]] = dict()
        self._word2idx: Optional[Dict[str, int]] = dict()
        self._idx2word: Optional[list] = list()
        self._index = 0
        self.min_frequency = 0 if not min_frequency else min_frequency
        self.max_size = max_size if max_size else np.inf

        if words:
            for w in words:
                self.add(w)

    def exist(self, word: str):
        return word in self._word2idx

    @property
    def word2idx(self):
        return self._word2idx

    @property
    def idx2word(self):
        return self._idx2word

    @property
    def frequency_dist(self):
        return self._frequency_dist

    @property
    def size(self):
        return len(self.word2idx)

    @property
    def token_count(self):
        return sum(self.frequency_dist.values())

    def add(self, word: str, freq=None):
        if not freq:
            freq = 1

        if word not in self.frequency_dist:
            self.frequency_dist[word] = freq
            self.word2idx[word] = self._index
            self._idx2word.append(word)
            self._index += 1
        else:
            self.frequency_dist[word] += freq

    def frequency(self, word: str):
        if not self.exist(word):
            return 0
        return self.frequency_dist[word]

    def index(self, word):
        if not self.exist(word):
            return -1

        return self.word2idx[word]

    def word(self, index):
        return self.idx2word[index]

    def json(self):
        j = {
            "frequency_dist": self.frequency_dist,
            "word2idx": self._word2idx,
            "idx2word": self._idx2word
        }
        return j

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        return vocab

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __contains__(self, word):
        return word in self._word2idx

    def __len__(self):
        return len(self._idx2word)

    def __iter__(self):
        return iter(self.idx2word)

    def intersection(self, v_other: 'Vocabulary') -> Tuple['Vocabulary', 'Vocabulary']:
        common_words = list(filter(self.__contains__, v_other))
        v = Vocabulary()
        v_o = Vocabulary()
        for word in common_words:
            v.add(word, self.frequency(word))
            v_o.add(word, v.frequency(word))

        return v, v_o


class Embeddings(VData):

    def __init__(self, W: np.ndarray = None, vocabulary: Union[Vocabulary, str] = None):
        self.W = W
        self._dimension = None
        self.vocabulary = vocabulary

        if W is not None:
            self._dimension = W.shape[1]

        if isinstance(vocabulary, str):
            self.vocabulary = Vocabulary.load(vocabulary)

    def vector(self, word: str) -> np.ndarray:
        idx = self.vocabulary.word2idx[word]
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
        for word, index in v.word2idx:
            W[index] = self.vector(word)
            W_o[index] = E.vector(word)

        emb = Embeddings(vocabulary=v)
        emb_o = Embeddings(vocabulary=v_other)
        emb.W = W
        emb_o.W = W_o
        return emb, emb_o

    def normalize(self, in_place=True):
        W = self.W - self.W.mean(0)
        if in_place:
            self.W = W
        else:
            return W

    class Ngrams(VData):
        pass
