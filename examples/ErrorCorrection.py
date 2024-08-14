from typing import List
import pymongo
from lingan.operations.embeddings import MostSimilar
from lingan.containers import Corpus
from lingan.string import levenshtein_distance
from lingan.adapters.embeddings import GensimAdapter
import pickle
import numpy as np

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["gts"]
entries = db["maddeler"]

with open("../absent.pkl", "rb") as f:
    absent_words: List = pickle.load(f)
    absent_words.remove(None)

with open("../found.pkl", "rb") as f:
    found: List = pickle.load(f)

all_docs = list(entries.find({}))
lookup = set()

for doc in all_docs:
    lookup.add(doc["madde"])

ALPHABET = [
    'a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş',
    't', 'u', 'ü', 'v', 'y', 'z', 'â', 'î', 'û'
]

a2idx = {a: i for i, a in enumerate(ALPHABET)}

potential_errors = {
    "ü": ["û"],
    "a": ["â"],
    "i": ["î", "l"],
    "t": ['d'],
    "p": ['b'],
    "ç": ['c'],
    'ı': ['i', "r", "l"],
    'c': ['e', "ç"],
    'u': ["ü", "û"],
    "ğ": ["g"],
    "ö": ["o"],
    "f": ["t"],
    "j": ["i", "l"],
    "y": ["v"],
    "m": ["n"],
    "s": ["ş"],
    "b": ["h"]

}

substitution_matrix = np.ones((len(ALPHABET), len(ALPHABET)), dtype=float)
for a, l in potential_errors.items():
    for b in l:
        substitution_matrix[a2idx[a], a2idx[b]] = 0.5
        substitution_matrix[a2idx[b], a2idx[a]] = 0.5

np.fill_diagonal(substitution_matrix, 0)

FAST_TEXT_PATH = "/home/togay/PycharmProjects/WordCompetition/models/8/1930.mdl"
emb_fast = GensimAdapter(FAST_TEXT_PATH).load()
corpus_fast = Corpus()
corpus_fast.data = emb_fast

problematic_words = []
mapping = dict()

for word in emb_fast.vocabulary:
    if word is None or len(word) <= 2 or len(word) > 5:
        continue

    if (word not in lookup
            and word + "mek" not in lookup
            and word + "mak" not in lookup
            and word not in found):
        continue

    try:
        similar_words_fast, sims = MostSimilar(word, 10).on_synchronic(corpus_fast)
        candidate_words = [w for w in similar_words_fast if w not in lookup]
        d = {}
        for c in candidate_words:
            distance = levenshtein_distance(word, c, s_cost=substitution_matrix, alphabet2idx=a2idx)
            d[c] = distance

        d = dict(sorted(filter(lambda x: x[1] <= 1.5, d.items()), key=lambda item: item[1]))
        for key in d.keys():
            if mapping.get(key) is None:
                mapping[key] = [word]
            else:
                mapping[key].append(word)

    except Exception as e:
        problematic_words.append(word)

with open('mapping_sgns_10.pkl', 'wb') as f:
    pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('problematic_words_sgns_5.pkl', 'wb') as f:
    pickle.dump(problematic_words, f, protocol=pickle.HIGHEST_PROTOCOL)
