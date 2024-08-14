from typing import Tuple, Union
from scipy.linalg import orthogonal_procrustes
from lingan.containers import DiachronicCorpus, Corpus
from lingan.models import Embeddings, Vocabulary
from lingan.operations.definitions import Operation
from lingan.math.functions import cosine_similarity
import numpy as np


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

    def on_synchronic(self, c: Corpus):
        pass


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

    def on_synchronic(self, c: Corpus):
        pass


class MostSimilar(Operation):
    def __init__(self, word: str, k: int, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice, Tuple] = None):
        self.word = word
        self.k = k
        self.target_period = target_period
        self.time_range = time_range

    def on_diachronic(self, d: DiachronicCorpus) -> Tuple[list[str], list[float], list[Tuple[int, int]]]:
        if self.target_period:
            corpus = d[self.target_period]
            return self.on_synchronic(corpus)

        words_across_time = []
        similarities = []
        corpora = [c for c in d.corpus_iterator(self.time_range)]
        for c in corpora:
            similar_words, sims = self.on_synchronic(c)
            words_across_time.append(similar_words)
            similarities.append(sims)

        return words_across_time, similarities, list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        embeddings: Embeddings = c.data
        word_vector = embeddings.vector(self.word)
        affinity_matrix = cosine_similarity(embeddings.W, word_vector)
        affinity_matrix[embeddings.vocabulary.index(self.word), 0] = -1
        indices = np.argsort(affinity_matrix[:, 0])[::-1][:self.k]
        similar_words = list(map(embeddings.vocabulary.word, indices))
        return similar_words, affinity_matrix[:, 0][indices].tolist()


class Similarity(Operation):
    def __init__(self, u: str, v: str, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice, Tuple] = None):
        self.u = u
        self.v = v
        self.target_period = target_period
        self.time_range = time_range

    def on_diachronic(self, d: DiachronicCorpus):
        if self.target_period:
            corpus = d[self.target_period]
            return self.on_synchronic(corpus)

        similarities = []
        corpora = [c for c in d.corpus_iterator(self.time_range)]
        for c in corpora:
            similar_words, sims = self.on_synchronic(c)
            similarities.append(sims)

        return similarities, list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        embeddings: Embeddings = c.data
        u_vector = embeddings.vector(self.u)
        v_vector = embeddings.vector(self.v)
        return cosine_similarity(u_vector, v_vector)
