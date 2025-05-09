from typing import Tuple, Union, Optional
from scipy.linalg import orthogonal_procrustes
from lingan.containers import DiachronicCorpus, Corpus
from lingan.models import Embeddings, Vocabulary
from lingan.operations.definitions import DiachronicOperation, Operation
from lingan.math.functions import cosine_similarity
import numpy as np

__all__ = ["OrthogonalAlignmentMatrix",
           "ProcrustesAlignment",
           "MostSimilar",
           "Similarity",
           "SecondOrderEmbedding",
           "AlignedMostSimilar",
           "AlignedVector"]


class LinearAlignmentMatrix(DiachronicOperation):
    pass


class IntersectionEmbeddings(DiachronicOperation):
    pass


class OrthogonalAlignmentMatrix(DiachronicOperation):

    def __init__(self, t1: Optional[Tuple[int, int]] = None, t2: Optional[Tuple[int, int]] = None, threshold=None):
        super().__init__(None, t1, t2)
        if not threshold:
            threshold = 0
        self.threshold = threshold
        self.cache = dict()

    def intersection_embeddings(self, c1: Corpus[Embeddings], c2: Corpus[Embeddings]) -> Tuple[Embeddings, Embeddings]:
        E_1: Embeddings = c1.data
        E_2: Embeddings = c2.data

        common_words = list(
            filter(lambda w: E_1.vocabulary.frequency(w) >= self.threshold and
                             E_2.vocabulary.frequency(w) >= self.threshold,
                   filter(E_1.vocabulary.__contains__, E_2.vocabulary))
        )

        v1 = Vocabulary()
        v2 = Vocabulary()
        W_1 = np.zeros(shape=(len(common_words), E_1.dimension))
        W_2 = np.zeros(shape=(len(common_words), E_2.dimension))

        for i, word in enumerate(common_words):
            v1.add(word, E_1.vocabulary.frequency(word))
            W_1[i] = E_1.vector(word)

            v2.add(word, E_2.vocabulary.frequency(word))
            W_2[i] = E_2.vector(word)

        E_1_C = Embeddings(W=W_1, vocabulary=v1)
        E_2_C = Embeddings(W=W_2, vocabulary=v2)

        return E_1_C, E_2_C

    def on_diachronic(self, d: DiachronicCorpus):
        if R := self.cache.get((self.t1, self.t2)):
            return R
        E1_c, E2_c = self.intersection_embeddings(d[self.t1], d[self.t2])
        R, _ = orthogonal_procrustes(E2_c.W, E1_c.W)
        self.cache[(self.t1, self.t2)] = R
        return R


class LinearAlignment(DiachronicOperation):
    def __init__(self, t1: Optional[Tuple[int, int]] = None, t2: Optional[Tuple[int, int]] = None, threshold=None):
        super().__init__(None, t1, t2)
        if not threshold:
            threshold = 0
        self.threshold = threshold
        self.cache = dict()

    def on_diachronic(self, d: DiachronicCorpus):
        pass


class ProcrustesAlignment(DiachronicOperation):
    def __init__(self, t1: Tuple[int, int], t2: Tuple[int, int], time_range=None):
        super().__init__(time_range, t1, t2)

    def on_diachronic(self, d: DiachronicCorpus):
        R = OrthogonalAlignmentMatrix(t1=self.t1, t2=self.t2).on_diachronic(d)
        E_2: Embeddings = d[self.t2].data
        W_2_aligned = E_2.W @ R
        E_aligned = Embeddings(W=W_2_aligned, vocabulary=E_2.vocabulary)
        corpus_aligned = Corpus(beginning=self.t2[0], end=self.t2[1], data=E_aligned)
        return corpus_aligned


class MostSimilar(Operation):
    def __init__(self, word: str, k: int, time_range: slice = None):
        super().__init__(time_range)
        self.word = word
        self.k = k

    def on_diachronic(self, d: DiachronicCorpus):
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
        affinity_matrix[embeddings.vocabulary.index(self.word), 0] = -2
        indices = np.argsort(affinity_matrix[:, 0])[::-1][:self.k]
        similar_words = list(map(embeddings.vocabulary.word, indices))
        return similar_words, affinity_matrix[:, 0][indices].tolist()


class AlignedVector(DiachronicOperation):
    def __init__(self, word: str, t1: Tuple[int, int], t2: Tuple[int, int], time_range=None):
        super().__init__(time_range, t1, t2)
        self.word = word

    def on_diachronic(self, d: DiachronicCorpus):
        R: Corpus[Embeddings] = OrthogonalAlignmentMatrix(self.t1, self.t2, 0).on_diachronic(d)
        word_vector = d[self.t2].data.vector(self.word)
        return word_vector @ R


class AlignedMostSimilar(DiachronicOperation):
    def __init__(self, word: str, k: int, t1: Tuple[int, int], t2: Tuple[int, int]):
        super().__init__(None, t1, t2)
        self.word = word
        self.k = k

    def on_diachronic(self, d: DiachronicCorpus[Embeddings]):
        embeddings: Embeddings = d[self.t1].data
        aligned_vector = AlignedVector(self.word, self.t1, self.t2, self.time_range).on_diachronic(d)
        affinity_matrix = cosine_similarity(embeddings.W, aligned_vector)
        indices = np.argsort(affinity_matrix[:, 0])[::-1][:self.k]
        similar_words = list(map(embeddings.vocabulary.word, indices))
        return similar_words, affinity_matrix[:, 0][indices].tolist()


class Similarity(Operation):
    def __init__(self, w1: str, w2: str, time_range: Union[slice, Tuple] = None):
        super().__init__(time_range)
        self.u = w1
        self.v = w2

    def on_diachronic(self, d: DiachronicCorpus):
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


class SecondOrderEmbedding(Operation):
    def __init__(self, word: str, k=10, time_range: slice = None):
        super().__init__(time_range)
        self.k = k
        self.word = word

    def on_diachronic(self, d: DiachronicCorpus[Embeddings]):
        vecs = []
        corpora = [c for c in d.corpus_iterator(self.time_range)]

        N = set([c.perform(MostSimilar(self.word, self.k)) for c in corpora])
        for c in corpora:
            neighborhood_t, = c.perform(MostSimilar(self.word, self.k))
            N = N.union(neighborhood_t)

        for c in corpora:
            v_c = []
            for n in N:
                if c.data.vocabulary.exist(n):
                    sim = cosine_similarity(c.data.vector(self.word), c.data.vector(n))
                else:
                    t2 = list(filter(lambda x: n in x.vocabulary, corpora))[0].period()
                    aligned_n = AlignedVector(self.word, c.period(), t2, self.time_range).on_diachronic(d)
                    sim = cosine_similarity(c.data.vector(self.word), aligned_n)
                v_c.append(sim)
            vecs.append(np.array(v_c))

        return vecs


class SemanticDrift(DiachronicOperation):
    def __init__(self, word: str, time_range: slice = None):
        super().__init__(time_range)
        self.word = word

    def on_diachronic(self, d: DiachronicCorpus[Embeddings]):
        dist = []
        corpora = [c for c in d.corpus_iterator(self.time_range)]

        base_period = corpora[0].period()
        base_vec = corpora[0].data.vector(self.word)

        corpora = corpora[1:]

        for c in corpora:
            period = c.period()
            v = AlignedVector(self.word, base_period, period, self.time_range).on_diachronic(d)
            distance = 1 - cosine_similarity(base_vec, v)
            dist.append(distance)

        time_periods = list(map(lambda c: (c.beginning, c.end), corpora))
        return np.array(dist), time_periods


class ChangePoint(DiachronicOperation):
    def __init__(self, word: str, B_times: int = 1000, z_threshold: float = 1.75, time_range: slice = None):
        super().__init__(time_range)
        self.word = word
        self.B_times = B_times
        self.z_threshold = z_threshold

    @staticmethod
    def mean_shift(arr):
        res = np.zeros(shape=(arr.shape[0] - 1,))
        for j in range(res.shape[0]):
            mean_shift = np.sum(arr[j + 1:]).mean() - np.sum(arr[:j + 1]).mean()
            res[j] = mean_shift

        return res

    def on_diachronic(self, d: DiachronicCorpus[Embeddings]):
        dist_series, periods = d.perform(SemanticDrift(self.word, self.time_range))
        z = (dist_series - np.mean(dist_series) / np.std(dist_series))

        mean_shift_orig = ChangePoint.mean_shift(z)

        B = np.zeros(shape=(self.B_times, mean_shift_orig.shape[0]), dtype=float)
        for i in range(self.B_times):
            z_perm = np.random.permutation(dist_series)
            B[i] = ChangePoint.mean_shift(z_perm)

        p_values = np.zeros_like(mean_shift_orig)
        for i in range(B.shape[1]):
            K_i = B[i]
            count = K_i[K_i >= mean_shift_orig[i]].size
            p_values[i] = (1 / self.B_times) * count

        min_p = np.finfo(np.float64).max
        min_period = None
        for i, z_val in enumerate(z[:-1]):
            if z_val > self.z_threshold:
                if p_values[i] < min_p:
                    min_p = p_values[i]
                    min_period = periods[i]

        return min_p, min_period


class CumulativeSimilarity(DiachronicOperation):
    pass


class Analogy(Operation):
    pass
