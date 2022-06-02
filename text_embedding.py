from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

from constant import FILE_PATH_TRAIN, FILE_PATH_TEST
from load_squad import load_squad_v2_label

stop_words = stopwords.words("english")


class TextCleaner:
    def __init__(self):
        pass

    def run(self, contexts):
        return [self.run_one(context) for context in contexts]

    def run_one(self, context):
        context = context.lower()
        context = re.sub("\w*\d\w*", "", context)
        context = re.sub("\n", " ", context)
        context = re.sub(r"http\S+", "", context)
        context = re.sub("[^a-z]", " ", context)
        return context


class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def run(self, contexts):
        return [self.run_one(context) for context in contexts]

    def run_one(self, context):
        return " ".join(
            [self.lemmatizer.lemmatize(word) for word in word_tokenize(context)]
        )


class Embedding:
    def __init__(self):
        self.file_path_train = FILE_PATH_TRAIN
        self.preprocessings = [TextCleaner(), Lemmatizer()]
        self.vectorizer = None

    def load_vectorizer(self, vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def save(self, vectorizer_path):
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)


class TfIdfEmbedding(Embedding):
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)

    def build_embedding(self, vectorizer_path=None):
        questions, contexts, _ = load_squad_v2_label(self.file_path_train)
        corpus = contexts + questions
        for p in self.preprocessings:
            corpus = p.run(corpus)
        self.vectorizer.fit(corpus)
        if vectorizer_path:
            self.save(vectorizer_path=vectorizer_path)

    def transform(self, texts: List[str], normalize: bool = True):
        for p in self.preprocessings:
            texts = p.run(texts)
        embeddings = self.vectorizer.transform(texts)
        if normalize:
            norms = np.linalg.norm(embeddings.toarray(), axis=1)
            return embeddings / (norms.reshape(-1, 1) + 1e-8)
        else:
            return embeddings


class BM25Embedding(Embedding):
    def __init__(self):
        super().__init__()
        self.vectorizer = None

    def build_embedding(self, vectorizer_path=None):
        questions, contexts, _ = load_squad_v2_label(FILE_PATH_TEST)
        for p in self.preprocessings:
            contexts = p.run(contexts)
        tokenized_corpus = [doc.split(" ") for doc in contexts]
        self.vectorizer = BM25Okapi(tokenized_corpus)
        if vectorizer_path:
            self.save(vectorizer_path=vectorizer_path)

    def transform(self, texts: List[str]):
        for p in self.preprocessings:
            texts = p.run(texts)
        return texts
