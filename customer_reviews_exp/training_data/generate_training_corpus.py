import os

import pandas as pd

from customer_reviews_exp.config.config import CORPUS_PATH, CORPUS_DATA_PATH, LENGTH_LIMIT

dfs = []

path = CORPUS_DATA_PATH
for file in sorted(os.listdir(path)):
    if file.endswith("labelled.txt"):
        file_path = os.path.join(path, file)
        _ = pd.read_csv(file_path, sep="\t", names=["text", "bool"])
        dfs.append(_)

df = pd.concat(dfs, ignore_index=True)
df = df.dropna()

df = df[df["text"].str.split().str.len() <= LENGTH_LIMIT]

df["text"] = df["text"].str.replace(r"[^a-zA-Z\s]", "", regex=True)
df["text"] = df["text"].str.lower()


corpus = df["text"]
corpus.to_csv(CORPUS_PATH, index=False, header=False)
