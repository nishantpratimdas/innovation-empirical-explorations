import os
from customer_reviews_exp.config.config import CORPUS_PATH, SIMILARITY_THRESHOLD, CORPUS_EMBEDDING_PATH, GENERATED_DATA_PATH, RATES_PATH, INFERENCE_CORPUS_SIZE
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

INDEX = 0

with open(CORPUS_PATH, "r") as f:
    corpus = set(line.strip() for line in f)

innovation_counts = {}
semantic_innovation_counts = {}

with open(CORPUS_EMBEDDING_PATH, "rb") as f:
    training_embeddings = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

for filepath in sorted(os.listdir(GENERATED_DATA_PATH)):
    if "_gram_" in filepath:
        with open(f"{GENERATED_DATA_PATH}/{filepath}", "r") as f:
            generated_lines = [line.strip() for line in f]

        num_not_in_corpus = 0
        for line in generated_lines:
            if line not in corpus:
                num_not_in_corpus += 1

        generated_embeddings = model.encode(generated_lines)
        similarity_matrix = (
            model.similarity(generated_embeddings,
                             training_embeddings).cpu().numpy()
        )
        semantic_similar_count = np.sum(
            similarity_matrix.max(axis=1) >= SIMILARITY_THRESHOLD
        )

        file_index = int(filepath[0])
        innovation_counts[file_index] = num_not_in_corpus
        semantic_innovation_counts[file_index] = INFERENCE_CORPUS_SIZE - semantic_similar_count


folder_path = RATES_PATH
os.makedirs(folder_path, exist_ok=True)

with open(f"{folder_path}/innovation_counts.pkl", "wb") as f:
    pickle.dump(innovation_counts, f)

with open(f"{folder_path}/semantic_innovation_counts.pkl", "wb") as f:
    pickle.dump(semantic_innovation_counts, f)

print(innovation_counts, semantic_innovation_counts)
