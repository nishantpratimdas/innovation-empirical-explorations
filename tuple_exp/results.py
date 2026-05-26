import csv

# Smallest and largest order of n-grams
n_max = 5
n_min = 2

with open("tuple_exp/tuple_dataset.csv", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    facts = [row for row in reader if row]

with open("tuple_exp/training_corpus.csv", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    corpus = [row for row in reader if row]

flag = 1
for i in range(n_min, n_max + 1):
    count = 0
    inn_count = 0
    with open(f"tuple_exp/generated_text/corpus_{i}.csv", newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row not in corpus:
                inn_count += 1
                if row not in facts:
                    continue
                else:
                    count += 1

    print(f"{i}-gram: {count} instances of innovation without hallucination")
    print(f"{i}-gram: {inn_count} instances of innovation")
    if count > 0:
        flag = 0
        break

if flag == 1:
    print("Whenever the model innovates, it hallucinates.")
