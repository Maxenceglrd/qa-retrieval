import json
from typing import Tuple, List


def load_squad_v2_label(
    path: str, n_context: int = 20000
) -> Tuple[List[str], List[str], List[int]]:
    with open(path, encoding="utf8") as f:
        data = json.load(f)["data"]
    questions = []
    contexts = []
    true_contexts = []
    for article in data:
        if len(contexts) <= n_context:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    questions.append(qa["question"])
                    true_contexts.append(len(contexts))
                contexts.append(paragraph["context"])
    return questions, contexts, true_contexts
