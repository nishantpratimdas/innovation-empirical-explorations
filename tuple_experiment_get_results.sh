# A toy tuple generation experiment inspired by Miao and Kearns (PNAS, 2026)

tbu_training_data () {
    echo "Creating training data for n-gram models from the TBU dataset of Miao and Kearns"
    uv run python -m tuple_exp.tbu_training_data
}

train_and_generate_text () {
    echo "Training and generating text"
    uv run python -m tuple_exp.train_and_generate_text
}

analyze_results () {
    echo "Checking if ever it is the case that the model innovates without hallucinating"
    uv run python -m tuple_exp.results
}

# Uncomment next line to generate the training data. The TBU dataset of Miao and
# Kearns can be generated using their publicly available code at
# https://github.com/mmiao2/Hallucination/blob/main/data/biography_data_pipeline.ipynb

# tbu_training_data

# Uncomment next line to redo n-gram training and text generation

# train_and_generate_text

analyze_results

