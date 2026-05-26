from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
import os
from customer_reviews_exp.config.config import CORPUS_PATH, INFERENCE_CORPUS_SIZE, GENERATED_DATA_PATH

import tqdm

def get_sentence(generated_sentence):
    sentence = []
    for word in generated_sentence:
        if word == "<s>":
            continue
        if word == "</s>":
            return sentence
        sentence.append(word)


def inference_dataset_generator(model, corpus_size,
                                seed=0, max_len=25,
                                max_tries=1000):
    tries = 0
    corpus = []
    pbar = tqdm.tqdm(total=corpus_size)
    while len(corpus) < corpus_size:
        tries = tries + 1
        if tries > max_tries:
            pbar.close()
            raise Exception(f"max tries reached. current max tries={max_tries}")

        # no need to put text seed as "<s>" as that is already the default.
        generated_sentence = model.generate(
            max_len,
            random_seed=seed*max_tries + tries
        )
        if "</s>" not in generated_sentence:
            continue
        else:
            inference = get_sentence(generated_sentence)
            if inference is not None and len(inference) > 0:
                corpus.append(inference)
                pbar.update(1)
    pbar.close()
    return corpus


n_min = 2
n_max = 7
inference_datasets = []
failed = False
inference_corpus_size = INFERENCE_CORPUS_SIZE

with open(CORPUS_PATH, "r") as f:
    lines = [line.strip() for line in f if line.strip()]
tokenized = [line.split() for line in lines]

for i in range(n_min, n_max + 1):
    print(f"Training a {i}-gram model")
    train_data, vocab = padded_everygram_pipeline(i, tokenized)
    model = Laplace(i)
    model.fit(train_data, vocab)
    print(f"Generating with a {i}-gram model")
    try:
        inference_datasets.append(
            inference_dataset_generator(model, inference_corpus_size)
        )

    except Exception as e:
        print("Generation failed:", e)
        failed = True

if not failed:
    folder_path = GENERATED_DATA_PATH
    os.makedirs(folder_path, exist_ok=True)

    unique_lines = set()
    for (i, dataset) in enumerate(inference_datasets):
        with open(f"{GENERATED_DATA_PATH}/{i + n_min}_gram_corpus.txt", "w") as f:
            for sentence in dataset:
                line = " ".join(sentence)
                f.write(line + "\n")
                unique_lines.add(line)

    unique_lines = sorted(unique_lines)
    with open(f"{GENERATED_DATA_PATH}/merged_everygram.txt", "w") as out:
        for line in unique_lines:
            out.write(line + "\n")
