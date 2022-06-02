import argparse
import pickle

import matplotlib.pyplot as plt

from evaluation import RetrievingEvaluationTFIDF, RetrievingEvaluationBM25
from text_embedding import TfIdfEmbedding, BM25Embedding

if __name__ == "__main__":
    test_context_embedding_path = "test_context_embedding.pk"

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_model", dest="train_model", action="store_true")
    parser.add_argument("--evaluate_all", dest="evaluate_all", action="store_true")
    parser.add_argument("--recompute_test", dest="recompute_test", action="store_true")
    parser.add_argument("--save_metrics", dest="save_metrics", action="store_true")
    parser.add_argument("--algo", dest="model_name", type=str, default="tfidf")
    parser.add_argument("--top_k", dest="top_k", type=int, default=10)
    parser.add_argument("--query", dest="query", type=str)
    args = parser.parse_args()

    model_name = args.model_name

    models_to_settings = {
        "tfidf": {
            "embedding": TfIdfEmbedding,
            "evaluator": RetrievingEvaluationTFIDF,
            "vectorizer_path": "models/vectorizer.pk",
        },
        "bm25": {
            "embedding": BM25Embedding,
            "evaluator": RetrievingEvaluationBM25,
            "vectorizer_path": "models/vectorizer_bm25.pk",
        },
    }

    vectorizer_path = models_to_settings[model_name]["vectorizer_path"]
    embedding = models_to_settings[model_name]["embedding"]()

    if args.train_model:
        embedding.build_embedding(vectorizer_path=vectorizer_path)
    else:
        embedding.load_vectorizer(vectorizer_path)
        evaluator = models_to_settings[model_name]["evaluator"](embedding)
        if args.evaluate_all:
            accuracy, good_questions = evaluator.evaluate_all(
                tok_k=args.top_k, precomputed=not args.recompute_test
            )
            print(f"Test accuracy on top_k = {args.top_k} : {accuracy} %")
        elif args.save_metrics:
            accuracies = []
            for k in range(1, 11):
                accuracy = evaluator.evaluate_all(
                    tok_k=k, precomputed=not args.recompute_test
                )
                accuracies.append(accuracy)
            fig = plt.figure()
            plt.plot(range(1, 11), accuracies)
            fig.savefig(f"assets/topk_accuracy_{model_name}.png")
            with open(f"assets/topk_accuracy_{model_name}.pk", "wb") as f:
                pickle.dump(accuracies, f)
        else:
            context, result = evaluator.get_one_query_result(args.query)
            print("\n")
            print("Best context for question:")
            print(args.query)

            print("\n")
            print(context)

            print("\n")
            print(f"Correct ? {result}")
