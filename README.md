We explore the connection between innovation and hallucination studied
theoretically in our work in **two** empirical settings.

1. We start with a setting due to Miao and Kearns (PNAS, 2026), who considered
   empirically exploring the Kalai Vempala framework in a controlled setting
   where $n$-grams models (their results mostly consider $n=2$) are trained to
   generate tuples of a fixed size (an example in their paper considers 6-tuples
   while their data and code seem to consider 7-tuples), denoting relationships
   derived from biographical data.  Their publicly available `TBU` dataset we
   consider consists of tuples of the form (Name, Date of Birth, Hometown, Field
   of Study, College/University, Job, Employer).  They subsample a subset of
   such samples from a database as the training data to train the $n$-gram.

   Given the lack of structure, it seems plausible that whenever a model trained
   on such a data innovated by producing a 7-tuple outside its training data, it
   would likely hallucinate (i.e. produce a tuple that is not in the database).
   This is what we observe for this setup: the innovations and hallucination
   rates are equal for small values of $n$ (in addition to the $n=2$ setting of
   Miao and Kearns, we also consider $n$ up to 5).  The code for analyzing this
   setting is available in the `tuple_exp` directory.



2. We then turn to a setting where more semantic information might be available,
   while still trying to remain within a similarly controlled setting.  The
   directory `customer_reviews_exp` contains the data and code for analyzing
   innovation vs. hallucination when $n$-gram models are trained using the
   Kotzias et al, 2015 dataset to generate short (up to 20 words) pieces of text
   that look like Amazon, Yelp or IMDB reviews.

   Innovation in this setting can be defined in a manner similar to the paper,
   by asking whether a generated piece of text appears in the training corpus.

   We also consider, in addition, a more **semantic** notion of innovation in
   which a generated piece of text is said to be an innovation if its similarity
   score, as computed by a sentence transformer model (`all-MiniLM-L6-v2`), is
   less than a high threshold (say 0.95) against every example in the training
   set.

   Checking whether a generated piece of text is a hallucination is also tricky
   in this situation. We examine both manual human labelling, and then, for
   scalability and bias reduction, using foundations models (claude-sonnet-4.6,
   deepseek/deepseek-v3.2, google/gemini-3-flash-preview, and openai/gpt-5.4 )
   to judge each generated text via a prompt of the following form:

   > Is the following text a review? Respond with a 1 if it is, or with a 0 if it isn't.
   > <BEGIN TEXT>
   > {text}
   > <END TEXT>

   As might be expected, the models differ in how strict they are in treating
   texts as hallucinations.  However, the general directional trend shown in
   innovation vs hallucination plot is similar for all the "judges".



To reproduce the results, please run the `bash` scripts
`tuple_experiment_get_results.sh` and `customer_reviews_expt_get_plots.sh`
respectively from the directory in which this scripts are located. By
uncommenting the marked lines in these scripts, the whole pipeline of the
experiments can be reproduced. One caveat is that running the script that
obtains the hallucination judgements in the `customer_reviews` experiments
requires an OpenRouter API key to be present in a `.env` file (that should be
located so that it can be found by the `load_dotenv` function of the
`python-dotenv` package).
