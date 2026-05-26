import pickle
from pathlib import Path

from matplotlib import pyplot as plt
from customer_reviews_exp.config.config import RATES_PATH, INFERENCE_CORPUS_SIZE, PLOTS_PATH
import numpy as np
from scipy.stats import binomtest

with open(f"{RATES_PATH}/innovation_counts.pkl", "rb") as f:
    innovation_counts = pickle.load(f)

with open(f"{RATES_PATH}/semantic_innovation_counts.pkl", "rb") as f:
    semantic_innovation_counts = pickle.load(f)

with open(f"{RATES_PATH}/hallucination_counts.pkl", "rb") as f:
    hallucination_counts = pickle.load(f)

print(f"Making sure the plots directory {PLOTS_PATH} exists.")
Path(PLOTS_PATH).mkdir(exist_ok=True)

nvals = list(innovation_counts.keys())

def plot_with_error_bars(n_data, label, fmt='', show_errors = True, **kwargs):
    ns = list(n_data.keys())
    counts = [n_data[n] for n in ns]
    cis = [binomtest(v, INFERENCE_CORPUS_SIZE).proportion_ci(confidence_level=0.95) for v in counts]
    lows = np.array([ci.low for ci in cis])
    highs = np.array([ci.high for ci in cis])
    vals = np.array([c/INFERENCE_CORPUS_SIZE for c in counts])
    if not show_errors:
        errors = None
    else:
        errors = [vals - lows, highs - vals]
    plt.ylim(bottom=0)
    plt.errorbar(
        ns,
        vals,
        errors,
        fmt=fmt,
        label=label,
        alpha=0.5,
        **kwargs
    )


plot_with_error_bars(innovation_counts, "Innovation rate", marker="o")
plot_with_error_bars(semantic_innovation_counts, "Semantic innovation rate", marker="o")

for judge, results in hallucination_counts.items():
    plot_with_error_bars(
        results,
        label=f"Hallucinations judged by {judge}",
        fmt=".--",
    )

plt.xlabel(r"Order $n$ of $n$-gram generator")
plt.ylabel(r"Rate (with Clopper-Pearson error-bars)")

plt.legend(loc=(0.5, 0.8), prop={'size': 6}, framealpha=1)
plt.savefig(f"{PLOTS_PATH}/innovation-and-hallucination.pdf")


plt.clf()

plot_with_error_bars(innovation_counts, "Innovation rate", show_errors=False, marker="o")
plot_with_error_bars(semantic_innovation_counts, "Semantic innovation rate", show_errors=False, marker="o")

for judge, results in hallucination_counts.items():
    plot_with_error_bars(
        results,
        show_errors=False,
        label=f"Hallucinations judged by {judge}",
        fmt=".--",
    )

plt.xlabel(r"Order $n$ of $n$-gram generator")
plt.ylabel(r"Rate")

plt.legend(loc=(0.5, 0.8), prop={'size': 6}, framealpha=1)
plt.savefig(f"{PLOTS_PATH}/innovation-and-hallucination-no-error-bars.pdf")

plt.clf()
innovation_vals = [innovation_counts[n]/INFERENCE_CORPUS_SIZE for n in nvals]

# remove top and right axis

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

for judge, results in hallucination_counts.items():
    plt.plot(
        innovation_vals,
        [results[n]/INFERENCE_CORPUS_SIZE for n in nvals],
        ".--",
        label=f"Hallucinations judged by {judge}",
        alpha=0.5
    )

plt.xlabel(r"Innovation rate")
plt.ylabel(r"Hallucination rate")
plt.legend(loc=(0.02, 0.7), prop={'size': 10}, framealpha=0.5)
plt.savefig(f"{PLOTS_PATH}/innovation-vs-hallucination.pdf")


print(f"Plots recreated in {PLOTS_PATH}.")
