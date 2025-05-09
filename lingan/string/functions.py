from typing import Union, Dict, List
import numpy as np
from typing_extensions import Tuple

from lingan.math.functions import jaccard_similarity
from itertools import product


def levenshtein_distance(source: str, target: str, d_cost: float = 1.0, i_cost: float = 1.0,
                         s_cost: Union[np.array, float] = 2.0, alphabet2idx: Dict[str, int] = None,
                         align=False) -> Union[float, Tuple[float, List]]:
    """
    :param source: str Source string
    :param target: str Target string
    :param d_cost: float deletion cost
    :param i_cost: float insertion cost
    :param s_cost: Union[np.array, float] substitution cost; either a cost matrix or a scalar
    :param alphabet2idx: Dict[str, int] mapping from character to index. Can only be used when s_cost is a cost matrix
    :param align: bool whether to return aligned strings
    :return: float levenshtein distance
    """

    if isinstance(s_cost, float):
        alphabet = sorted(list(set(source).union(set(target))))
        alphabet2idx = {c: i for i, c in enumerate(alphabet)}
        s_cost_matrix = np.zeros((len(alphabet2idx), len(alphabet2idx)), dtype=np.float32)
        s_cost_matrix.fill(s_cost)
        np.fill_diagonal(s_cost_matrix, 0.0)
    else:
        s_cost_matrix = s_cost

    n = len(source)
    m = len(target)
    D = np.zeros((n + 1, m + 1), dtype=np.float32)
    trace = dict.fromkeys(product(range(n + 1), range(m + 1)))
    for key in trace.keys():
        trace[key] = []

    D[1:, 0] = np.arange(d_cost, n * d_cost + d_cost, d_cost)
    D[0, 1:] = np.arange(i_cost, m * i_cost + i_cost, i_cost)

    for i in range(1, n + 1):
        trace[(i, 0)].append('d')

    for j in range(1, m + 1):
        trace[(0, j)].append('i')

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(
                D[i - 1, j] + d_cost,
                D[i, j - 1] + i_cost,
                D[i - 1, j - 1] + s_cost_matrix[alphabet2idx[source[i - 1]], alphabet2idx[target[j - 1]]],
            )

            if D[i - 1, j] + d_cost <= D[i, j]:
                trace[(i, j)].append('d')
            if D[i, j - 1] + i_cost <= D[i, j]:
                trace[(i, j)].append('i')
            if D[i - 1, j - 1] + s_cost_matrix[alphabet2idx[source[i - 1]], alphabet2idx[target[j - 1]]] <= D[i, j]:
                trace[(i, j)].append('s')

    if align:
        i, j = n, m
        alignment_ops = []
        while i > 0 or j > 0:
            if 's' in trace[(i, j)]:
                alignment_ops.append('s')
                i -= 1
                j -= 1
            elif 'i' in trace[(i, j)]:
                alignment_ops.append('i')
                j -= 1
            else:
                alignment_ops.append('d')
                i -= 1
        alignment_ops.reverse()

        src_a = []
        target_a = []
        i, j = 0, 0
        for op in alignment_ops:
            if op == 'd':
                src_a.append(source[i])
                target_a.append('*')
                i += 1
            elif op == 's':
                src_a.append(source[i])
                target_a.append(target[j])
                i += 1
                j += 1
            else:
                src_a.append('*')
                target_a.append(target[j])
                j += 1



        alignment_string = ' '.join(src_a) + '\n' + ' '.join(len(alignment_ops) * ['|']) + '\n' + ' '.join(target_a)
        return D[n, m], alignment_string, alignment_ops

    return D[n, m]


def damerau_levenshtein_distance(source: str, target: str, d_cost: float = 1.0, i_cost: float = 1.0,
                                 t_cost: float = 1.0,
                                 s_cost: Union[np.array, float] = 2.0, alphabet2idx: Dict[str, int] = None) -> float:
    if isinstance(s_cost, float):
        alphabet = sorted(list(set(source).union(set(target))))
        alphabet2idx = {c: i for i, c in enumerate(alphabet)}
        s_cost_matrix = np.zeros((len(alphabet2idx), len(alphabet2idx)), dtype=np.float32)
        s_cost_matrix.fill(s_cost)
        np.fill_diagonal(s_cost_matrix, 0.0)
    else:
        s_cost_matrix = s_cost

    n = len(source)
    m = len(target)
    D = np.zeros((n + 1, m + 1), dtype=np.float32)
    D[1:, 0] = np.arange(d_cost, n * d_cost + d_cost, d_cost)
    D[0, 1:] = np.arange(i_cost, n * i_cost + i_cost, i_cost)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(
                D[i - 1, j] + d_cost,
                D[i, j - 1] + i_cost,
                D[i - 1, j - 1] + s_cost_matrix[alphabet2idx[source[i - 1]], alphabet2idx[target[j - 1]]],
                D[i - 2, j - 2] + t_cost if source[i - 1] == target[j - 2]
                                            and source[i - 2] == target[j - 1] else np.inf
            )

    return D[n, m]


def jaro_distance(source: str, target: str) -> float:
    match_distance = min(len(source), len(target)) // 2

    matched_s = [False] * len(source)
    matched_t = [False] * len(target)

    s = []
    t = []

    for i in range(len(source)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(target))

        for j in range(start, end):
            if source[i] != target[j]:
                continue
            if matched_t[j]:
                continue

            s.append(source[i])
            matched_t[j] = True
            break

    for i in range(len(target)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(source))

        for j in range(start, end):
            if target[i] != source[j]:
                continue
            if matched_s[j]:
                continue

            t.append(target[i])
            matched_s[j] = True
            break

    transpositions = (len(list(filter(lambda i: s[i] != t[i], range(min(len(s), len(t)))))) + abs(len(s) - len(t))) / 2

    return 1 - (1 / 3 * (len(s) / len(source) + len(t) / len(target) + ((len(s) - transpositions) / len(s))))


def jaro_winkler_distance(source: str, target: str, weight=5) -> float:
    prefix_length = 0
    for i in range(len(source)):
        if source[i] == target[i]:
            prefix_length += 1
        else:
            break
    return jaro_distance(source, target) + max(prefix_length, weight) * 0.1 * (
            1 - jaro_distance(source, target))


def hamming_distance(source: str, target: str) -> float:
    if len(source) != len(target):
        raise ValueError('Hamming distance is defined for sequences of equal length')
    if len(source) == 0 and len(target) == 0:
        return 0.0
    return sum(c1 != c2 for c1, c2 in zip(source, target))


def lcs_distance(source: str, target: str) -> float:
    if len(source) == 0 and len(target) == 0:
        return 0.0

    if source[-1] == target[-1]:
        return lcs_distance(source[:-1], target[:-1])

    return 1 + min(lcs_distance(source, target[:-1]), lcs_distance(source[:-1], target))


def jaccard_distance(source: str, target: str, q: int = 2) -> float:
    if q > min(len(source), len(target)):
        raise ValueError('Jaccard distance is defined for q <= min(|source|, |target|)')

    if q == 0 and len(source) + len(target) > 0:
        raise ValueError('Jaccard distance is defined for q > 0 when one of the source or target string is not empty')

    if len(source) == 0 and len(target) == 0:
        return 0.0

    Q_source = [source[i:i + q] for i in range(0, len(source) - 1 - q + 1)]
    Q_target = [target[i:i + q] for i in range(0, len(target) - 1 - q + 1)]
    return 1 - jaccard_similarity(Q_source, Q_target)


def q_distance(source: str, target: str, q: int = 2) -> float:
    if q > min(len(source), len(target)):
        raise ValueError('Q distance is defined for q <= min(|source|, |target|)')

    if q == 0 and len(source) + len(target) > 0:
        raise ValueError('Q distance is defined for q > 0 when one of the source or target string is not empty')

    if len(source) == 0 and len(target) == 0:
        return 0.0

    Q_source = [source[i:i + q] for i in range(0, len(source) - q)]
    Q_target = [target[i:i + q] for i in range(0, len(target) - q)]

    vec_dims = set.union(set(Q_source), set(Q_target))
    s_vec = np.array([Q_source.count(d) for d in vec_dims])
    t_vec = np.array([Q_target.count(d) for d in vec_dims])

    return np.linalg.norm((s_vec - t_vec), ord=1)


def cosine_distance(source: str, target: str, q: int = 2) -> float:
    if q > min(len(source), len(target)):
        raise ValueError('Cosine distance is defined for q <= min(|source|, |target|)')

    if q == 0 and len(source) + len(target) > 0:
        raise ValueError('Cosine distance is defined for q > 0 when one of the source or target string is not empty')

    if len(source) == 0 and len(target) == 0:
        return 0.0

    Q_source = [source[i:i + q] for i in range(0, len(source) - q)]
    Q_target = [target[i:i + q] for i in range(0, len(target) - q)]

    vec_dims = set.union(set(Q_source), set(Q_target))
    s_vec = np.array([Q_source.count(d) for d in vec_dims])
    t_vec = np.array([Q_target.count(d) for d in vec_dims])

    return 1 - np.dot(s_vec, t_vec) / (np.linalg.norm(s_vec) * np.linalg.norm(t_vec))
