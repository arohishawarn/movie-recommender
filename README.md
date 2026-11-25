# Content-Based Movie Recommender

A MovieLens-based movie recommendation project built in Python. The notebook explores movie representation, content similarity, user-level train/test splitting, and several ways to build a user preference profile from movie ratings.

## Project overview

The project currently covers two related approaches:

1. **Movie-to-movie recommendations** — represent movies using their genres and user-generated tags, then find similar movies with cosine similarity.
2. **User profiling** — combine the content vectors of movies a user has rated to create a representation of that user's preferences.

The notebook is an exploratory, step-by-step implementation. A complete personalized ranking and recommendation evaluation pipeline is not yet included.

## Dataset

The notebook attempts to load the **MovieLens Latest Small** dataset from GroupLens:

- `movies.csv` — movie IDs, titles, and genres
- `ratings.csv` — user IDs, movie IDs, ratings, and timestamps
- `tags.csv` — user-applied tags for movies

If the MovieLens download or loading process fails, the notebook falls back to a small synthetic dataset so that the workflow can still be explored offline. The synthetic data is generated for demonstration and is not real MovieLens data.

## Feature engineering

### Genre one-hot encoding

Movie genres are separated using the pipe (`|`) delimiter and converted into binary genre columns. The resulting matrix is stored in `H`.

### Genre and tag text

Tags are lowercased and grouped by `movieId`. They are then joined with each movie's genres to create a combined text field named `doc`.

### TF-IDF representation

Scikit-learn's `TfidfVectorizer` converts each movie document into a sparse TF-IDF feature vector. The resulting matrix is stored in `X`.

The notebook also extracts the release year from the movie title and fills missing years with the median year. The current movie-similarity function uses the TF-IDF matrix; the one-hot matrix is prepared for feature exploration.

## Movie similarity and recommendations

The notebook includes:

- `find_movie(query)` — finds the first movie title containing the supplied text (case-insensitive).
- `top_cosine_similar(query, n=5)` — calculates cosine similarity between the selected movie and all movie vectors, excludes the selected movie itself, and returns the top `n` matches with their titles and genres.

Example:

```python
top_cosine_similar("toy", n=5)
```

## User profiling

Ratings are split **per user**, rather than by randomly splitting the complete ratings table:

- Users with fewer than 10 ratings are skipped by default.
- 80% of each eligible user's ratings are assigned to training.
- 20% are assigned to testing.
- The split uses a fixed random state for reproducibility.

The test set is used to identify relevant movies: movies rated **4 or higher** are treated as relevant. `eval_users` contains users who have at least one relevant test movie and also appear in the training data.

The notebook demonstrates four user-profile strategies for a selected user:

| Profile | How it is constructed |
| --- | --- |
| Average profile | Uses only movies rated 4 or higher, with equal weights. |
| Rating-weighted profile | Uses all training ratings as weights, so higher ratings contribute more. |
| Centered profile | Uses each rating minus that user's mean rating, allowing above-average and below-average preferences to contribute with opposite signs. |
| Time-decay profile | Applies exponential time decay to the centered rating weights, giving more recent ratings greater influence. |

The time-decay calculation uses the user's most recent rating timestamp as the reference point and a configurable `decay_lambda` value.

The notebook compares the highest-weighted TF-IDF terms from these profiles and also inspects positive and negative terms for the time-decay profile.

## Tech stack

- Python
- NumPy
- pandas
- scikit-learn
- Jupyter Notebook

## Getting started

1. Clone the repository:

   ```bash
   git clone https://github.com/arohishawarn/movie-recommender.git
   cd movie-recommender
   ```

2. Install the required packages:

   ```bash
   pip install numpy pandas scikit-learn jupyter
   ```

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open **Content Based Recommendation System.ipynb** and run the cells from top to bottom.

The notebook downloads and extracts MovieLens Latest Small automatically when it is not already present in the working directory and a download is available.

## Current scope and next steps

The current notebook establishes movie content features, movie-to-movie cosine similarity, a per-user data split, relevance labels, and four user-profile construction methods.

The following parts are not yet implemented in the notebook:

- Generating and ranking personalized candidate recommendations from a user profile
- Filtering movies the user has already rated
- Evaluating personalized recommendations with ranking metrics such as Precision@K or Recall@K

These are natural next stages for extending the project.

## Repository structure

```text
movie-recommender/
├── Content Based Recommendation System.ipynb
└── README.md
```

## Notes

- The notebook currently demonstrates profiling with a selected user ID (`u = 4`); change this value to inspect another eligible user.
- The synthetic fallback is useful for testing the notebook structure, but results from it should not be interpreted as real user preferences.
- Run notebook cells in order because later sections depend on variables created earlier.
