from customer_reviews_exp.config.config import GENERATED_DATA_PATH, CORPUS_PATH, ANNOTATION_TODO_CORPUS_PATH

INPUT_PATH = f"{GENERATED_DATA_PATH}/merged_everygram.txt"
FILTER_PATH = CORPUS_PATH
OUTPUT_PATH = ANNOTATION_TODO_CORPUS_PATH


with open(FILTER_PATH, "r") as f:
    filter_set = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(filter_set)} lines to filter")


kept = 0
removed = 0

with open(INPUT_PATH, "r") as fin, open(OUTPUT_PATH, "w") as fout:
    for line in fin:
        clean = line.strip()
        if not clean:
            continue

        if clean in filter_set:
            removed += 1
        else:
            fout.write(clean + "\n")
            kept += 1

print(f"Kept: {kept}")
print(f"Already in training set, so no annotations needed: {removed}")
