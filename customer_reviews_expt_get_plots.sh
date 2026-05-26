# Customer reviews experiment

preprocess_training_data () {

    echo "Generating training corpus from the Kotzias et al. dataset"
    uv run python3 -m customer_reviews_exp.training_data.generate_training_corpus

    echo "Generating training corpus embeddings using sentence-transformers"
    uv run python3 -m customer_reviews_exp.training_data.generate_training_corpus_embeddings

}

train_and_generate_text () {
    echo "Training and generating text"
    uv run python -m customer_reviews_exp.train_and_generate_text

}

create_annotation_todo () {
    echo "Create list of texts to be annotated for whether they are hallucinations"
    uv run python -m customer_reviews_exp.annotation_todo_creator
}

get_ai_annotations () {
    echo "Get AI annotations"
    uv run python -m customer_reviews_exp.ai_annotation_generator "anthropic/claude-sonnet-4.6"
    uv run python -m customer_reviews_exp.ai_annotation_generator "deepseek/deepseek-v3.2"
    uv run python -m customer_reviews_exp.ai_annotation_generator "google/gemini-3-flash-preview"
    uv run python -m customer_reviews_exp.ai_annotation_generator "openai/gpt-5.4"
}

aggregate_ai_annotations () {
    echo "Aggregating AI annotations"
    uv run python -m customer_reviews_exp.annotations_aggregator
}

compute_rates () {
    echo "Computing innovation counts"
    uv run python -m customer_reviews_exp.compute_innovation_counts
    echo "Computing hallucination counts"
    uv run python -m customer_reviews_exp.compute_hallucination_counts
}

plot_data () {
    echo "Plotting data"
    uv run python -m customer_reviews_exp.plot_innovation_hallucination
}

# Uncomment next line to redo pre-processing of input dataset

# preprocess_training_data

# Uncomment next line to redo n-gram training and text generation

# train_and_generate_text

# Uncomment next line to redo generation of list of texts that are to be annotated

# create_annotation_todo

# Uncomment the next line to get AI annotations
#
# NOTE: This requires valid Openrouter API key in an .env file, as described in
# README.md. For API usage policy reasons, the API key used in our experiments
# is not included.

# get_ai_annotations

# Uncomment this to recompute the majority vote out of the 11 run for each statement for each AI

# aggregate_ai_annotations

# Uncommment to compute innovation and hallucination counts

# compute_rates

# Uncomment to generate plots

plot_data
