from lingan.adapters.protocols import Adapter
from gensim.models import Word2Vec, FastText

from lingan.models import Embeddings, Vocabulary


class GensimAdapter(Adapter):

    def __init__(self, model_path=None, model_type: str = 'Word2Vec'):
        self.model_path = model_path
        self.model_type = model_type

    def _load_model(self, path: str):
        if self.model_type == 'Word2Vec':
            model = Word2Vec.load(path)
        elif self.model_type == 'FastText':
            model = FastText.load(path)
        else:
            model = None
        return model

    def load(self, path: str = None):
        if not path:
            path = self.model_path

        model = self._load_model(path)

        vocabulary = Vocabulary()

        for word in model.wv.index_to_key:
            vocabulary.add(word, model.wv.get_vecattr(word, "count"))

        emb = Embeddings(W=model.wv.vectors, vocabulary=vocabulary)
        return emb

    def save(self, path: str = None):
        pass
