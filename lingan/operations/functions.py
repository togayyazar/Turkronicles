from typing import Tuple, Union

from scipy.linalg import orthogonal_procrustes

from lingan.containers import DiachronicCorpus, Corpus
from lingan.models import Embeddings, Vocabulary
from lingan.operations.definitions import Operation


class AlignmentMatrix(Operation):
    def __init__(self, base: Tuple[str, str], target: Tuple[str, str], normalize=True):
        self.base_period = base
        self.target_period = target
        self.normalize = normalize
        self.cache = dict()

    def on_diachronic(self, d: DiachronicCorpus):
        if (self.base_period, self.target_period) in self.cache:
            R = self.cache[(self.base_period, self.target_period)]
        else:
            corpus_target: Corpus = d[self.target_period]
            corpus_base: Corpus = d[self.base_period]
            E_target: Embeddings = corpus_target.data
            E_base: Embeddings = corpus_base.data

            E_b_int, E_t_int = E_base.intersection(E_target)

            W_b = E_b_int.W
            W_t = E_t_int.W

            if self.normalize:
                W_b = W_b - W_b.mean(axis=0)
                W_t = W_t - W_t.mean(axis=0)

            R, _ = orthogonal_procrustes(W_t, W_b)
            self.cache[(self.base_period, self.target_period)] = R

        return R


class AlignEmbeddings(Operation):

    def __init__(self, base: Tuple[str, str], target: Tuple[str, str], in_place=True, normalize=True, use_cache=True):
        self.base_period = base
        self.target_period = target
        self.normalize = normalize
        self.in_place = in_place

    def on_diachronic(self, d: DiachronicCorpus):
        corpus_target: Corpus = d[self.target_period]
        E_target: Embeddings = corpus_target.data
        R = AlignEmbeddings(self.base_period, self.target_period)
        aligned_embeddings = E_target.W @ R
        if self.in_place:
            E_target.W = aligned_embeddings
        else:
            new_E = Embeddings(W=aligned_embeddings, vocabulary=E_target.vocabulary)
            return new_E

    def set_target_period(self, target: Tuple[str, str]):
        self.target_period = target

    def set_base_period(self, base: Tuple[str, str]):
        self.base_period = base


class Exists(Operation):
    def __init__(self, word: str, time_range: Union[slice, Tuple] = None):
        self.time_range = time_range
        self.word = word

    def on_diachronic(self, d: DiachronicCorpus):
        queue: list = d[self.time_range]
        for cont in iter(queue):
            if isinstance(cont, Corpus):
                if cont.data.exist(self.word):
                    return True
            else:
                queue.extend(cont.corpora)

        return False

    def on_synchronic(self, c: Corpus):
        data: Vocabulary = c.data
        return data.exist(self.word)


class Frequency(Operation):
    def __init__(self, word: str, time_range: Union[slice, Tuple] = None):
        self.time_range = time_range
        self.word = word

    def on_diachronic(self, d: DiachronicCorpus):
        time_series = {}
        for c in d.corpus_iterator(self.time_range):
            time_series[(c.beginning, c.end)] += c.data.frequency(self.word)

        return sum(time_series.values()),time_series

    def on_synchronic(self, c: Corpus):
        return c.data.frequency(self.word)
