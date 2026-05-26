import argparse
import csv
import requests
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from customer_reviews_exp.config.config import GENERATED_DATA_PATH, ANNOTATION_TODO_CORPUS_PATH

load_dotenv()

# the annotations were generated using the following models:
#   1. anthropic/claude-sonnet-4.6
#   2. deepseek/deepseek-v3.2
#   3. google/gemini-3-flash-preview
#   4. openai/gpt-5.4

# replace the model name in the following line to generate annotations for
# the corresponding model.

MODEL_LIST = [
    "anthropic/claude-sonnet-4.6",
    "deepseek/deepseek-v3.2",
    "google/gemini-3-flash-preview",
    "openai/gpt-5.4"
]

parser = argparse.ArgumentParser(
    prog='Annotation generator',
    description='Generate annotations')

parser.add_argument(
    'model', type=str,
    default=MODEL_LIST[0],
    help=f"Should be one of {MODEL_LIST}"
)

parser.add_argument(
    "-l", "--limit", type=int,
    default=None,
    help="Limit to annotating a small number of texts"
)


args = parser.parse_args()
model = args.model
if model not in MODEL_LIST:
    raise Exception(f"Model should be one of {MODEL_LIST}")

model_file_name = model.replace("/", "-")

# This endpoint works on openrouter for all the models we use
url = "https://openrouter.ai/api/v1/messages"

folder_path = f"{GENERATED_DATA_PATH}/annotated_corpus"
os.makedirs(folder_path, exist_ok=True)

SAVE_PATH = f"{folder_path}/{model_file_name}_annotations.csv"

# Generated strings that were not present in the training data, so that we need
# to check if they are hallucinations
CORPUS_PATH = ANNOTATION_TODO_CORPUS_PATH

MAX_WORKERS = 50

# Ask each model odd number times for each statement, so that we can take majority
NUM_SAMPLES = 11

def ask_model(text, max_retries=5):
    base_prompt = f"""Is the following text a review?
Respond with a 1 if it is, or with a 0 if it isn't.
<BEGIN TEXT>
{text}
<END TEXT>"""

    retry_prompt = "\nIMPORTANT: Respond with ONLY a single character: 1 or 0. No explanation."

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            # Emphasize the 0-1 output requirement more as the number of retries
            # increases
            prompt = base_prompt + (retry_prompt * attempt)

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16, # since some models refuse to respond with fewer than 16
            }

            response = requests.post(url, json=payload, headers=headers)
            result = response.json()

            content = result.get("content") or []
            text_out = None

            if isinstance(content, list):
                for item in content:
                    if "text" in item:
                        text_out = item["text"].strip()
                        break

            if text_out in {"0", "1"}:
                return text_out

        except Exception as e:
            print(f"Retry {attempt+1} failed: {text[:30]}... -> {e}")

    return "ERROR"


def process(i, statement, k):
    result = ask_model(statement)
    return i, statement, k, result


with open(CORPUS_PATH, "r") as f:
    corpus_rows = [line.strip() for line in f if line.strip()]

completed = set()

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            try:
                i = int(row[0])
                k = int(row[2])
                completed.add((i, k))
            except:
                continue

    print(f"Loaded {len(completed)} completed samples")
else:
    print("Starting fresh")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []
    submitted_count = 0
    for i, statement in enumerate(corpus_rows):
        for k in range(NUM_SAMPLES):
            if (i, k) in completed:
                continue

            futures.append(executor.submit(process, i, statement, k))
            submitted_count += 1
            if args.limit is not None and submitted_count >= args.limit:
                break

        if args.limit is not None and submitted_count >= args.limit:
            print(f"Completed submitting {args.limit} annotations.")
            break

    with open(SAVE_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not os.path.exists(SAVE_PATH):
            writer.writerow(["index", "statement", "sample_id", model])

        for future in tqdm(as_completed(futures), total=len(futures)):
            i, statement, k, result = future.result()
            writer.writerow([i, statement, k, result])
