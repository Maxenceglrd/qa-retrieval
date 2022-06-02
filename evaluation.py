import pickle

import numpy as np

from constant import FILE_PATH_TEST
from load_squad import load_squad_v2_label
from text_embedding import TfIdfEmbedding, BM25Embedding


def cosine_score(queries, contexts):
    return queries @ contexts.T


class Evaluation:
    def __init__(self):
        pass

    @staticmethod
    def evaluate_accuracy_top_k(scores, true_contexts, top_k: int = 10):
        best_contexts = []
        for i in range(scores.shape[0]):
            best_context = list(
                np.ravel(np.argpartition(scores[i, :], -top_k))[-top_k:]
            )
            best_contexts.append(best_context)
        matching = []
        good_questions = []
        for true_context, best_context in zip(true_contexts, best_contexts):
            matching.append((true_context, best_context))
        n_correct = 0
        for i, (true_index, proposed_index) in enumerate(matching):
            if true_index in proposed_index:
                good_questions.append(i)
                n_correct += 1
        return n_correct / len(matching) * 100, good_questions


class RetrievingEvaluationTFIDF(Evaluation):
    def __init__(self, embedding: TfIdfEmbedding):
        super().__init__()
        self.evaluated_embedding = embedding

    def evaluate_all(self, precomputed: bool = True, tok_k: int = 10):
        if precomputed:
            _, _, true_contexts = load_squad_v2_label(FILE_PATH_TEST)
            with open("models/question_test_embedding.pk", "rb") as f:
                questions_test = pickle.load(f)
            with open("models/contexts_test_embedding.pk", "rb") as f:
                contexts_test = pickle.load(f)
        else:
            questions_test, contexts_test, true_contexts = load_squad_v2_label(
                FILE_PATH_TEST
            )
            questions_test = self.evaluated_embedding.transform(
                questions_test, normalize=True
            )
            contexts_test = self.evaluated_embedding.transform(
                contexts_test, normalize=True
            )
            with open("models/question_test_embedding.pk", "wb") as f:
                pickle.dump(questions_test, f)
            with open("models/contexts_test_embedding.pk", "wb") as f:
                pickle.dump(contexts_test, f)
        scores = cosine_score(questions_test, contexts_test)
        test_accuracy = self.evaluate_accuracy_top_k(scores, true_contexts, top_k=tok_k)
        return test_accuracy

    def get_one_query_result(self, query: str):
        with open("models/contexts_test_embedding.pk", "rb") as f:
            contexts_test = pickle.load(f)
        questions, contexts, true_contexts = load_squad_v2_label(FILE_PATH_TEST)
        query_embedding = self.evaluated_embedding.transform([query], normalize=True)
        scores = cosine_score(query_embedding, contexts_test)
        predicted_context = contexts[np.argmax(scores)]
        if contexts[true_contexts[questions.index(query)]] == predicted_context:
            result = True
        else:
            result = False
        return contexts[np.argmax(scores)], result


class RetrievingEvaluationBM25(Evaluation):
    def __init__(self, embedding: BM25Embedding):
        super().__init__()
        self.evaluated_embedding = embedding

    def evaluate_all(self, precomputed: bool = True, tok_k: int = 10):
        questions_test, _, true_contexts = load_squad_v2_label(FILE_PATH_TEST)
        for p in self.evaluated_embedding.preprocessings:
            questions_test = p.run(questions_test)
        questions_test = [query.split(" ") for query in questions_test]
        scores = np.array(
            [self.evaluated_embedding.vectorizer.get_scores(q) for q in questions_test]
        )
        test_accuracy = self.evaluate_accuracy_top_k(scores, true_contexts, top_k=tok_k)
        return test_accuracy

    def get_one_query_result(self, query: str):
        questions, contexts, true_contexts = load_squad_v2_label(FILE_PATH_TEST)
        split_query = query.split(" ")
        scores = self.evaluated_embedding.vectorizer.get_scores(split_query)
        predicted_context = contexts[np.argmax(scores)]
        if contexts[true_contexts[questions.index(query)]] == predicted_context:
            result = True
        else:
            result = False
        return contexts[np.argmax(scores)], result
