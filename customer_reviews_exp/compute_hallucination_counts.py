import os
import pickle
from customer_reviews_exp.config.config import INFERENCE_CORPUS_SIZE, RATES_PATH, CORPUS_PATH, GENERATED_DATA_PATH
from collections import defaultdict
import csv

INDEX = 0

training_examples = set()
with open(CORPUS_PATH, "r") as f:
    for line in f:
        training_examples.update(line.strip() for line in f)


def get_hallucination_count(generated_lines, annotations):
    hallucinations = 0
    for line in generated_lines:
        if line in training_examples:
            continue
        annotation = annotations.get(line)
        if annotation is not None:
            if annotation == "0":
                hallucinations += 1
        else:
            raise Exception(f"Line {line} is not annotated")
    return hallucinations


hallucination_count = defaultdict(dict)

folder_path = RATES_PATH
os.makedirs(folder_path, exist_ok=True)

SUFFIX = "_annotations_aggregated.csv"

for annotations_file in os.listdir(f"{GENERATED_DATA_PATH}/annotated_corpus"):
    if SUFFIX not in annotations_file:
        continue
    with open(f"{GENERATED_DATA_PATH}/annotated_corpus/{annotations_file}", "r") as f:
        reader = csv.DictReader(f)
        annotated_corpus = {}
        for row in reader:
            annotated_corpus[row["statement"]] = row["consensus"]

    for generations_file in sorted(os.listdir(GENERATED_DATA_PATH)):
        if "_gram_" in generations_file:
            with open(f"{GENERATED_DATA_PATH}/{generations_file}", "r") as f:
                generated_lines = [line.strip() for line in f]

            hallucination_count[
                annotations_file.replace(SUFFIX, "")
            ][int(generations_file[0])] = get_hallucination_count(
                generated_lines, annotated_corpus
            )

with open(f"{folder_path}/hallucination_counts.pkl", "wb") as f:
    pickle.dump(hallucination_count, f)

print(hallucination_count)
