import itertools
from typing import Tuple, Union, Dict
import numpy as np
from lingan.containers import DiachronicCorpus, Corpus
from lingan.models import Vocabulary
from lingan.operations.definitions import Operation
from scipy.spatial.distance import jensenshannon


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

        return sum(time_series.values()), time_series


class MergeVocabulary(Operation):

    def __init__(self, base_period: Union[Tuple[int, int]], target_period: Union[Tuple[int, int]]):
        self.base_period = base_period
        self.target_period = target_period

    def on_diachronic(self, d: DiachronicCorpus):
        v_base: Vocabulary = d[self.base_period].data
        v_target: Vocabulary = d[self.target_period].data
        union_vocab = set(v_base.idx2word).union(v_target.idx2word)
        new_vocab = Vocabulary()

        for word in union_vocab:
            freq = v_base.frequency(word) + v_target.frequency(word)
            new_vocab.add(word, freq)

        return new_vocab


class FilterFrequency(Operation):

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


class Similarity(Operation):

    def __init__(self, time_range: Union[slice]):
        self.time_range = time_range

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
            words1 = v1.idx2word
            for j, c2 in enumerate(corpora):
                v2: Vocabulary = c2.data
                words2 = v2.idx2word
                affinity_matrix[i, j] = Similarity.sim(words1, words2)

        return affinity_matrix, list(map(lambda x: (x.beginning, x.end), corpora))


class Distance(Operation):
    def __init__(self, base_period: Union[Tuple[int, int]] = None, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None):
        self.time_range = time_range
        self.base_period = base_period
        self.target_period = target_period

    @staticmethod
    def divergence(word_freq1: Dict[str, int], word_freq2: Dict[str, int], weighted=None):
        w1 = 0.5
        w2 = 0.5
        if weighted:
            w1 = len(word_freq1) / (len(word_freq1) + len(word_freq2))
            w2 = len(word_freq2) / (len(word_freq1) + len(word_freq2))

        P = {}
        Q = {}
        words = set(itertools.chain(word_freq1.keys(), word_freq2.keys()))
        for word in words:
            P[word] = P.get(word, 0) + word_freq1.get(word, 0)
            Q[word] = Q.get(word, 0) + word_freq2.get(word, 0)

        P_dist = np.array(list(P.values()))
        Q_dist = np.array(list(Q.values()))

        return np.power(jensenshannon(w1 * P_dist, w2 * Q_dist, 2), 1 / 2)

    def on_diachronic(self, d: DiachronicCorpus):
        if self.base_period:
            base_corpus: Corpus = d[self.base_period]
            base_vocab: Vocabulary = base_corpus.data

            target_corpus: Corpus = d[self.target_period]
            target_vocab: Vocabulary = target_corpus.data

            return Distance.divergence(base_vocab.frequency_dist, target_vocab.frequency_dist)

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


class MorphemeFrequency(Operation):
    """
    Initializes the MorphemeFrequency class.

    Parameters:
        morpheme (str): The morpheme to analyze.
        target_period (Union[Tuple[int, int], None]): Beginning and end points of a corpus being considered in
        which the morpheme is searched.
        time_range (Union[slice, None]): The range of time to consider for analysis.

    Returns:
        None
    """

    def __init__(self, morpheme: str, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None):
        self.time_range = time_range
        self.target_period = target_period
        self.morpheme = morpheme

    def on_diachronic(self, d: DiachronicCorpus):
        if self.target_period:
            corpus: Corpus = d[self.target_period]
            v: Vocabulary = corpus.data
            total = 0
            for word, freq in v.frequency_dist.items():
                if self.morpheme in word:
                    total += freq
            return total

        corpora = [c for c in d.corpus_iterator(self.time_range)]
        time_series = []
        for c in corpora:
            v: Vocabulary = c.data
            total = 0
            for word, freq in v.frequency_dist.items():
                if self.morpheme in word:
                    total += freq
            time_series.append(total)
        return np.array(time_series), list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        v: Vocabulary = c.data
        total = 0
        for word, freq in v.frequency_dist.items():
            if self.morpheme in word:
                total += freq
        return total


class WordsWithMorpheme(Operation):
    def __init__(self, morpheme: str, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None):
        self.time_range = time_range
        self.target_period = target_period
        self.morpheme = morpheme

    def on_diachronic(self, d: DiachronicCorpus):
        if self.target_period:
            corpus: Corpus = d[self.target_period]
            v: Vocabulary = corpus.data
            words = list(filter(lambda w: self.morpheme in w, v))

            return words

        corpora = [c for c in d.corpus_iterator(self.time_range)]
        words_c = []
        for c in corpora:
            v: Vocabulary = c.data
            words = list(filter(lambda w: self.morpheme in w, v))
            words_c.append(words)
        return words_c, list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        v: Vocabulary = c.data
        words = list(filter(lambda w: self.morpheme in w, v))
        return words


class WordsEndsWith(Operation):
    def __init__(self, suffix: str, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None):
        self.time_range = time_range
        self.target_period = target_period
        self.suffix = suffix

    def on_diachronic(self, d: DiachronicCorpus):
        if self.target_period:
            corpus: Corpus = d[self.target_period]
            v: Vocabulary = corpus.data
            words = list(filter(lambda w: w.endswith(self.suffix), v))
            return words

        corpora = [c for c in d.corpus_iterator(self.time_range)]
        words_c = []
        for c in corpora:
            v: Vocabulary = c.data
            words = list(filter(lambda w: w.endswith(self.suffix), v))
            words_c.append(words)
        return words_c, list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        v: Vocabulary = c.data
        words = list(filter(lambda w: w.endswith(self.suffix), v))
        return words


class WordsStartsWith(Operation):
    def __init__(self, prefix: str, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None):
        self.time_range = time_range
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

    def __init__(self, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None):
        self.target_period = target_period
        self.time_range = time_range

    def on_diachronic(self, d: DiachronicCorpus):
        if self.target_period:
            corpus: Corpus = d[self.target_period]
            v: Vocabulary = corpus.data
            return len(v)

        corpora = [c for c in d.corpus_iterator(self.time_range)]
        time_series = []
        for c in corpora:
            v: Vocabulary = c.data
            time_series.append(len(v))
        return time_series, list(map(lambda c: (c.beginning, c.end), corpora))

    def on_synchronic(self, c: Corpus):
        v: Vocabulary = c.data
        return len(v)


class CommonWords(Operation):

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
            return set.intersection(set(v_base.idx2word), set(v_target.idx2word))

        corpora = [c for c in d.corpus_iterator(self.time_range)]
        return set.intersection(*map(lambda c: set(c.data.idx2word), corpora))


class AverageWordLength(Operation):

    def __init__(self, target_period: Union[Tuple[int, int]] = None,
                 time_range: Union[slice] = None):
        self.time_range = time_range
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
