import pandas as pd
import glob

from customer_reviews_exp.config.config import GENERATED_DATA_PATH

PATH = f"{GENERATED_DATA_PATH}/annotated_corpus"

for filepath in glob.glob(f"{PATH}/*_annotations.csv"):
    print(filepath)
    df = pd.read_csv(filepath)
    model_name = df.columns.tolist()[-1]
    print(model_name)
    df["response"] = pd.to_numeric(df[model_name])

    results = []

    for idx, group in df.groupby("index"):
        values = group["response"].astype(int)
        num_values = len(values)

        if num_values == 0:
            consensus = "ERROR"
            confidence = 0.0
        else:
            count_1 = (values == 1).sum()
            count_0 = (values == 0).sum()

            if count_1 >= count_0:
                consensus = 1
                confidence = count_1 / num_values
            else:
                consensus = 0
                confidence = count_0 / num_values

        statement = group["statement"].iloc[0]

        results.append({
            "index": idx,
            "statement": statement,
            "consensus": consensus,
            "confidence": confidence,
            "num_samples": num_values,
        })

    out_df = pd.DataFrame(results)
    outpath = f'{filepath.replace(".csv", "")}_aggregated.csv'
    print(outpath)
    out_df.to_csv(outpath, index=False)
