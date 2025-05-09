import itertools
from typing import Tuple, Union, Dict, Optional
import numpy as np
from scipy.optimize import curve_fit

from lingan.containers import DiachronicCorpus, Corpus
from lingan.models import Vocabulary
from lingan.operations.definitions import OperationBase, DiachronicOperation, Operation, SynchronicOperation
from scipy.spatial.distance import jensenshannon


class Exists(Operation):
    def __init__(self, word: str, time_range: slice = None):
        super().__init__(time_range)
        self.word = word

    def on_diachronic(self, d: DiachronicCorpus[Vocabulary]):
        corpora = [c for c in d.corpus_iterator(self.time_range)]
        time_series = np.zeros((len(corpora),), dtype=bool)
        for i, c in enumerate(corpora):
            time_series[i] = self.on_synchronic(c)

        return time_series

    def on_synchronic(self, c: Corpus[Vocabulary]):
        data: Vocabulary = c.data
        return data.exist(self.word)


class Frequency(Operation):
    def __init__(self, word: str, time_range: Union[slice, Tuple] = None, normalized=False):
        super().__init__(time_range=time_range)
        self.word = word
        self.normalized = normalized

    def on_diachronic(self, d: DiachronicCorpus):
        corpora = [c for c in d.corpus_iterator(self.time_range)]
        time_series = [self.on_synchronic(c) for c in corpora]

        return time_series, list(map(lambda x: (x.beginning, x.end), corpora))

    def on_synchronic(self, c: Corpus):
        data: Vocabulary = c.data
        if self.normalized:
            return data.frequency(self.word) / sum(data.frequency_dist.values())
        return data.frequency(self.word)


class MergeVocabulary(DiachronicOperation):

    def __init__(self, time_range: Union[slice] = None):
        super().__init__(time_range)

    def on_diachronic(self, d: DiachronicCorpus[Vocabulary]):
        unified_vocab = Vocabulary()
        for c in d.corpus_iterator(self.time_range):
            vocab = c.data
            for word in vocab.idx2key:
                unified_vocab.add(word, vocab.frequency(word))
        return unified_vocab


class FilterFrequency(OperationBase):

    def __init__(self, threshold: int, time_range: Union[slice]):
        self.threshold = threshold
        self.time_range = time_range

    def on_diachronic(self, d: DiachronicCorpus):
        for c in d.corpus_iterator(self.time_range):
            v: Vocabulary = c.data
            new_v = Vocabulary()
            for word in v:
                if freq := v.frequency(word) > self.threshold:
                    new_v.add(word, freq)
            c.data = new_v

    def on_synchronic(self, c: Corpus):
        v: Vocabulary = c.data
        new_v = Vocabulary()
        for word in v:
            if freq := v.frequency(word) > self.threshold:
                new_v.add(word, freq)
        c.data = new_v


class Similarity(DiachronicOperation):

    def __init__(self, time_range: Optional[Union[slice]] = None):
        super().__init__(time_range)

    @staticmethod
    def sim(words1, words2):
        w1 = set(words1)
        w2 = set(words2)
        return len(w1.intersection(w2)) / len(w1.union(w2))

    def on_diachronic(self, d: DiachronicCorpus):
        corpora = [c for c in d.corpus_iterator(self.time_range)]
        affinity_matrix = np.zeros((len(corpora), len(corpora)))
        for i, c1 in enumerate(corpora):
            v1: Vocabulary = c1.data
            words1 = v1.idx2key
            for j, c2 in enumerate(corpora):
                v2: Vocabulary = c2.data
                words2 = v2.idx2key
                affinity_matrix[i, j] = Similarity.sim(words1, words2)

        return affinity_matrix, list(map(lambda x: (x.beginning, x.end), corpora))


class Distance(DiachronicOperation):
    def __init__(self, t1: Union[Tuple[int, int]] = None, t2: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None, weighted=False):
        super().__init__(time_range)
        self.t1 = t1
        self.t2 = t2
        self.weighted = weighted

    @staticmethod
    def divergence(word_freq1: Dict[str, int], word_freq2: Dict[str, int]):

        common_vocab = list(set(word_freq1.keys()).union(word_freq2.keys()))
        P = {w: word_freq1.get(w, 0) for w in common_vocab}
        Q = {w: word_freq2.get(w, 0) for w in common_vocab}

        P_dist = np.array(list(P.values()))
        Q_dist = np.array(list(Q.values()))

        return jensenshannon(P_dist, Q_dist, 2)

    def on_diachronic(self, d: DiachronicCorpus):
        if self.t1 and self.t2:
            t1_vocab: Vocabulary = d[self.t1].data
            t2_vocab: Vocabulary = d[self.t2].data
            return Distance.divergence(t1_vocab.frequency_dist, t2_vocab.frequency_dist)

        corpora = [c for c in d.corpus_iterator(self.time_range)]
        distance_matrix = np.zeros((len(corpora), len(corpora)))
        for i, c1 in enumerate(corpora):
            v1: Vocabulary = c1.data
            words_freq1 = v1.frequency_dist
            for j, c2 in enumerate(corpora):
                v2: Vocabulary = c2.data
                words_freq2 = v2.frequency_dist
                distance_matrix[i, j] = Distance.divergence(words_freq1, words_freq2)

        return distance_matrix, list(map(lambda x: (x.beginning, x.end), corpora))


class SubwordFrequency(Operation):
    def __init__(self, subword: str, normalized=False, time_range: Optional[slice] = None):
        super().__init__(time_range)
        self.subword = subword
        self.normalized = normalized

    def on_diachronic(self, d: DiachronicCorpus):
        corpora = [c for c in d.corpus_iterator(self.time_range)]
        time_series = np.zeros(shape=(len(corpora),), dtype=np.float32)
        for i, c in enumerate(corpora):
            time_series[i] = self.on_synchronic(c)

        return time_series, list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        v: Vocabulary = c.data
        total = 0
        for word, freq in v.frequency_dist.items():
            if self.subword in word:
                total += freq
        if self.normalized:
            total /= sum(v.frequency_dist.values())
        return total


class WordsWithSubword(Operation):
    def __init__(self, subword: str, time_range: slice = None):
        super().__init__(time_range)
        self.subword = subword

    def on_diachronic(self, d: DiachronicCorpus[Vocabulary]):
        corpora = [c for c in d.corpus_iterator(self.time_range)]
        words_c = []
        for c in corpora:
            words_c.append(self.on_synchronic(c))
        return words_c, list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus[Vocabulary]):
        words = list(filter(lambda w: self.subword in w, c.data))
        return words


class WordsEndWith(Operation):
    def __init__(self, suffix: str, time_range: slice = None):
        super().__init__(time_range)
        self.suffix = suffix

    def on_diachronic(self, d: DiachronicCorpus):
        corpora = [c for c in d.corpus_iterator(self.time_range)]
        words_c = []
        for c in corpora:
            words_c.append(self.on_synchronic(c))
        return words_c, list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        words = list(filter(lambda w: w.endswith(self.suffix), c.data))
        return words


class WordsStartsWith(OperationBase):
    def __init__(self, prefix: str, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None):
        super().__init__(time_range)
        self.target_period = target_period
        self.prefix = prefix

    def on_diachronic(self, d: DiachronicCorpus):
        if self.target_period:
            corpus: Corpus = d[self.target_period]
            v: Vocabulary = corpus.data
            words = list(filter(lambda w: w.startswith(self.prefix), v))
            return words

        corpora = [c for c in d.corpus_iterator(self.time_range)]
        words_c = []
        for c in corpora:
            v: Vocabulary = c.data
            words = list(filter(lambda w: w.startswith(self.prefix), v))
            words_c.append(words)
        return words_c, list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        v: Vocabulary = c.data
        words = list(filter(lambda w: w.startswith(self.prefix), v))
        return words


class UniqueWordCount(Operation):

    def __init__(self, time_range: Union[slice] = None):
        super().__init__(time_range=time_range)

    def on_diachronic(self, d: DiachronicCorpus):
        corpora = [c for c in d.corpus_iterator(self.time_range)]
        return [self.on_synchronic(c) for c in corpora], list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        v: Vocabulary = c.data
        return len(v)


class CommonWords(OperationBase):

    def __init__(self, base_period: Union[Tuple[int, int]] = None, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None):
        self.time_range = time_range
        self.base_period = base_period
        self.target_period = target_period

    def on_diachronic(self, d: DiachronicCorpus):
        if self.target_period:
            corpus_base: Corpus = d[self.target_period]
            corpus_target: Corpus = d[self.base_period]
            v_base: Vocabulary = corpus_base.data
            v_target: Vocabulary = corpus_target.data
            return set.intersection(set(v_base.idx2key), set(v_target.idx2key))

        corpora = [c for c in d.corpus_iterator(self.time_range)]
        return set.intersection(*map(lambda c: set(c.data.idx2key), corpora))


class AverageWordLength(Operation):

    def __init__(self, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None):
        super().__init__(time_range)
        self.target_period = target_period

    def on_diachronic(self, d: DiachronicCorpus):
        if self.target_period:
            corpus: Corpus = d[self.target_period]
            v: Vocabulary = corpus.data
            return sum(len(word) for word in v) / len(v)

        corpora = [c for c in d.corpus_iterator(self.time_range)]
        time_series = []
        for c in corpora:
            v: Vocabulary = c.data
            time_series.append(sum(len(word) for word in v) / len(v))
        return time_series, list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        v: Vocabulary = c.data
        return sum(len(word) for word in v) / len(v)


class TokenCount(Operation):
    def __init__(self, time_range: Union[slice] = None):
        super().__init__(time_range)

    def on_diachronic(self, d: DiachronicCorpus):
        corpora = [c for c in d.corpus_iterator(self.time_range)]
        return [self.on_synchronic(c) for c in corpora], list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        v: Vocabulary = c.data
        return v.token_count()


class TypeOverlap(DiachronicOperation):
    def __init__(self, time_range: slice = None, ):
        super().__init__(time_range)

    def on_diachronic(self, d: DiachronicCorpus[Vocabulary]):
        corpora = [c for c in d.corpus_iterator(self.time_range)]
        base_c = corpora[0]
        return [len(set(base_c.data.idx2key).intersection(c.data.idx2key)) for c in corpora], list(
            map(lambda c: (c.beginning, c.end), corpora))


class DivergentWords(DiachronicOperation):
    def __init__(self, t1: Tuple[int, int], t2: Tuple[int, int], n=None):
        super().__init__(t1=t1, t2=t2)
        self.n = n

    def on_diachronic(self, d: DiachronicCorpus):
        c1_vocab: Vocabulary = d[self.t1].data
        c2_vocab: Vocabulary = d[self.t2].data

        c1_freq_dist = c1_vocab.frequency_dist
        c2_freq_dist = c2_vocab.frequency_dist

        common_vocab = np.array(list(set(c1_freq_dist.keys()).union(c2_freq_dist.keys())))
        P = np.array([c1_freq_dist.get(w, 0) for w in common_vocab])
        Q = np.array([c2_freq_dist.get(w, 0) for w in common_vocab])

        P_dist = P / np.sum(P)
        Q_dist = Q / np.sum(Q)

        M = 0.5 * (P_dist + Q_dist)

        scores = np.zeros((len(common_vocab),))

        for i, w in enumerate(common_vocab):
            a = -M[i] * np.log2(M[i])
            b = 0 if P_dist[i] == 0 else P_dist[i] * np.log2(P_dist[i])
            c = 0 if Q_dist[i] == 0 else Q_dist[i] * np.log2(Q_dist[i])
            scores[i] = a + 0.5 * (b + c)

        sorted_indices = np.argsort(-scores)
        sorted_words = common_vocab[sorted_indices]
        if not self.n:
            self.n = len(sorted_words)
        return sorted_words[:self.n], scores[sorted_indices][:self.n]


class TypeTokenRatio(Operation):
    def __init__(self, time_range: Union[slice] = None):
        super().__init__(time_range=time_range)

    def on_diachronic(self, d: DiachronicCorpus[Vocabulary]):
        TTR_series = [self.on_synchronic(c) for c in d.corpus_iterator(time_range=self.time_range)]
        time_periods = list(map(lambda c: (c.beginning, c.end), d.corpus_iterator(time_range=self.time_range)))
        return TTR_series, time_periods

    def on_synchronic(self, c: Corpus[Vocabulary]):
        token_count = c.perform(TokenCount())
        unique_word_count = c.perform(UniqueWordCount())
        return (unique_word_count / token_count) * 100


class HeapsFit(Operation):
    pass


class ZipfsFit(Operation):
    def __init__(self, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None):
        super().__init__(time_range)
        self.target_period = target_period

    @staticmethod
    def zipf_law(x, s, k):
        return k / np.power(x, s)

    def on_diachronic(self, d: DiachronicCorpus[Vocabulary]):
        pass

    def on_synchronic(self, c: Corpus[Vocabulary]):
        sorted_freq = sorted(c.data.frequency_dist.items(), key=lambda x: x[1], reverse=True)
        frequencies = np.array(list(zip(*sorted_freq))[1])

        ranks = np.arange(1, len(frequencies) + 1)

        initial = [1.0, max(frequencies)]

        try:
            params = curve_fit(ZipfsFit.zipf_law, ranks, frequencies, p0=initial, maxfev=10000)
            return params  # (s, A)
        except RuntimeError:
            print("Optimization failed. Try adjusting initial parameters or input data.")
            return None
