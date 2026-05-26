from sentence_transformers import SentenceTransformer
from customer_reviews_exp.config.config import CORPUS_PATH, CORPUS_EMBEDDING_PATH
import pickle

with open(CORPUS_PATH, "r") as f:
    training_corpus = [line.strip() for line in f]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(training_corpus)
with open(CORPUS_EMBEDDING_PATH, "wb") as f:
    pickle.dump(embeddings, f)
