import pickle
import random
from collections import defaultdict
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
import csv

import tqdm

def get_sentence(generated_sentence):
    sentence = []
    for word in generated_sentence:
        if word == "<s>":
            continue
        if word == "</s>":
            return sentence
        sentence.append(word)

n_min = 2
n_max = 5
training_corpus_size = 5000
inference_corpus_size = 500
max_tries = inference_corpus_size*100
random.seed(0)

with open("tuple_exp/tuple_dataset.csv", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    facts = [row for row in reader if row]

innovation_rates = defaultdict(list)
hallucination_rates = defaultdict(list)

training_corpus = random.choices(facts, k=training_corpus_size)

with open(f"tuple_exp/training_corpus.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(training_corpus)

for i in range(n_min, n_max + 1):
    train_data, vocab = padded_everygram_pipeline(i, training_corpus)
    model = Laplace(i)
    model.fit(train_data, vocab)
    print(f"{i}-gram model trained, now generating")
    inference_dataset = []
    tries = 0

    # Generate length 7 tuples, as this was the length of the tuples in the training data
    pbar = tqdm.tqdm(total=inference_corpus_size)
    while len(inference_dataset) < inference_corpus_size and tries < max_tries :
        tries += 1
        generated_text = model.generate(num_words=15, random_seed=tries)
        generated_sentence = get_sentence(generated_text)
        if generated_sentence is not None and len(generated_sentence) == 7:
            inference_dataset.append(generated_sentence)
            pbar.update(1)

    pbar.close()

    if len(inference_dataset) != inference_corpus_size:
        raise Exception("Failed")

    with open(f"tuple_exp/generated_text/corpus_{i}.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(inference_dataset)

    innovated_count = 0
    hallucinated_count = 0

    for line in inference_dataset:
        if line not in facts:
            hallucinated_count += 1

        if line not in training_corpus:
            innovated_count += 1

    innovation_rates[i] = innovated_count / inference_corpus_size
    hallucination_rates[i] = hallucinated_count / inference_corpus_size

print("Innovation rates: ", innovation_rates)
print("Hallucination rates: ", hallucination_rates)

results = {
    "innovation": dict(innovation_rates),
    "hallucination": dict(hallucination_rates),
    "n_min": n_min,
    "n_max": n_max,
}

with open("tuple_exp/results.pkl", "wb") as f:
    pickle.dump(results, f)

print("Saved results to tuple_exp/results.pkl")
