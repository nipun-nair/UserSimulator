import pandas as pd
import json
from pathlib import Path
from task_eval import SimulatorEvaluator
import math
import pandas as pd
from scipy.stats import entropy, pearsonr

def compute_entropy(values, normalize=True):
    """
    Compute Shannon entropy for a list of discrete values.

    Args:
        values (list): List of discrete values (strings, categories, etc.)
        normalize (bool): If True, returns normalized entropy [0,1].

    Returns:
        float: Entropy value
    """
    if not values:
        return 0.0

    counts = pd.Series(values).value_counts()
    probs = counts / counts.sum()
    ent = entropy(probs, base=2) 
    return ent


def entropy_per_column_csv(csv_file, columns=None, normalize=True):
    """
    Compute entropy for each column in a CSV file.

    Args:
        csv_file (str): Path to CSV file
        columns (list or None): List of column names to compute entropy. If None, compute all columns.
        normalize (bool): Whether to normalize entropy [0,1]

    Returns:
        dict: {column_name: entropy_value}
    """
    df = pd.read_csv(csv_file)
    if columns is None:
        columns = df.columns.tolist()

    entropies = {}
    for col in columns:
        entropies[col] = compute_entropy(df[col].dropna().tolist(), normalize=normalize)
    
    return entropies

def evaluate_binpref(values, n_responses=100):
    pos_rates = []
    for entry in values:
        if pd.isna(entry):  
            continue
        if isinstance(entry, str):
            entry_list = [x.strip() for x in entry.split(',')]
        else:
            entry_list = entry
        yes_count = sum(str(resp).lower().rstrip('.') == "yes" for resp in entry_list[:n_responses])
        pos_rate = yes_count / n_responses
        pos_rates.append(pos_rate)
    return pos_rates

def evaluate_binpref_col(sim_csv, human_csv):
    sim_data = pd.read_csv(sim_csv)
    human_data = pd.read_csv(human_csv)
    columns = sim_data.columns.tolist()
    binpref = {}
    for col in columns:
        sim_rates = evaluate_binpref(sim_data[col])
        human_ratings = human_data['avg_rating'][:len(sim_rates)]
        if len(sim_rates) > 1:
            binpref[col] = pearsonr(human_ratings, sim_rates)[0]
        else:
            binpref[col] = None
    return binpref

def evaluate_openpref_columns(csv_file):
    """
    Evaluate OpenPref (Task3) for all columns in a CSV.
    Assumes each cell contains comma-separated aspects or aspect|sentiment pairs.
    
    Returns a dictionary:
        {column_name: {'num_aspects': int, 'aspect_entropy': float, 'sentiment_entropy': float}}
    """
    df = pd.read_csv(csv_file)
    results = {}

    for col in df.columns:
        all_aspects = []
        all_sentiments = []

        for cell in df[col].dropna():
            if not str(cell).strip():  # Skip empty strings or cells with only whitespace
                continue
            # If the cell is a comma-separated list
            parts = [x.strip() for x in str(cell).split(',')]
            for part in parts:
                if '|' in part:  # aspect|sentiment format
                    aspect, sentiment = part.split('|', 1)
                    all_aspects.append(aspect.strip())
                    all_sentiments.append(sentiment.strip())
                else:  # only aspect
                    all_aspects.append(part)
        
        results[col] = {
            'num_aspects': len(set(all_aspects)),
            'aspect_entropy': compute_entropy(all_aspects),
            'sentiment_entropy': compute_entropy(all_sentiments) if all_sentiments else 0.0
        }
    
    return results    

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer

# --- Word Diversity (TTR) ---
def word_diversity(items):
    words = []
    for cell in items:
        for word in str(cell).split():
            words.append(word.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

# --- Word Embedding Diversity (W2V) ---
def w2v_diversity(items, w2v_model):
    vecs = []
    for cell in items:
        for word in str(cell).split():
            if word in w2v_model:
                vecs.append(w2v_model[word])
    if len(vecs) < 2:
        return 0.0
    sims = cosine_similarity(vecs)
    upper_tri = sims[np.triu_indices_from(sims, k=1)]
    return 1 - np.mean(upper_tri)  # higher = more diverse

# --- Sentence Embedding Diversity ---
def sentence_embedding_diversity(items, model):
    embeddings = model.encode([str(cell) for cell in items])
    if len(embeddings) < 2:
        return 0.0
    sims = cosine_similarity(embeddings)
    upper_tri = sims[np.triu_indices_from(sims, k=1)]
    return 1 - np.mean(upper_tri)

# --- Main evaluation function ---
def evaluate_recrequest_diversity(csv_file, sentence_model_name='all-MiniLM-L6-v2'):
    df = pd.read_csv(csv_file)
    results = {}

    # Load models if needed
    w2v_model = KeyedVectors.load_word2vec_format("glove.6B.50d.word2vec.txt", binary=False)
    sent_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    for col in df.columns:
        items = df[col].dropna().tolist()
        results[col] = {
            'Word Diversity (TTR)': word_diversity(items),
            'Word Embedding Diversity (W2V)': w2v_diversity(items, w2v_model) if w2v_model else None,
            'Sentence Embedding Diversity': sentence_embedding_diversity(items, sent_model)
        }

    return results
from task_eval import SimulatorEvaluator
from profile_creation import OllamaHelper
def evaluate_negative_feedback(csv_file):
    # Load CSV file into DataFrame
    df = pd.read_csv(csv_file)
    
    # Initialize the evaluator and Ollama model
    evaluator = SimulatorEvaluator()
    evaluator.ollama = OllamaHelper(model="llama3.3:70b")
    
    results = {}
    scores = {i: [] for i in range(len(df.columns))}  # Initialize a dictionary to store scores for each column
    
    # Iterate over each column in the DataFrame
    for i, col in enumerate(df.columns):
        for j, response_text in df[col].items():  # Corrected: using items() to iterate over column data
            eval_prompt = f"""You are an evaluator. Score the simulator's response: {response_text}

                                Evaluate based on these three criteria:
                                A. Clear rejection with persona-justified reasoning (tone, genre, theme preferences)
                                B. Semantically rich, persona-aligned reformulation request without naming specific movies
                                C. Tone and reasoning style matching the stated persona
                                
                                Scoring rubric:
                                0 = Responds "I don't know" when meaningful response possible
                                1-2 = Fails criteria A, B, or C completely, or mentions movies
                                3-4 = Meets only 1 criterion partially, weak execution
                                5-6 = Meets 1-2 criteria but with generic/weak elements
                                7-8 = Meets 2-3 criteria but with minor gaps in persona alignment or semantic richness
                                9 = Meets all criteria A, B, C with minor imperfections
                                10 = Perfect compliance with criteria A, B, and C
                                
                                Return only the numeric score (0-10)."""
        
            llm_output = evaluator.ollama.call_model(eval_prompt)
            
            # Extract the first integer between 0 and 10
            score = None
            for token in llm_output.strip().split():
                if token.isdigit():
                    val = int(token)
                    if 0 <= val <= 10:
                        score = val
                        break

            if score is not None:
                scores[i].append(score)  # Append the score to the list for the respective column
            else: 
                scores[i].append(0)  # Append a default score of 0 if no valid score found

        # Calculate the average score for the column
        avg_score = np.mean(scores[i]) if scores[i] else 0.0  # Use scores[i] to get the list of scores for this column
        print(f"Average score of {col}: {avg_score:.2f}")
        
    return scores

# Example usage
if __name__ == "__main__":
    # Load your CSV
    df1 = pd.read_csv("baseline3/baseline_results_task1.csv") 
    df2 = pd.read_csv("baseline/baseline_results_task2.csv") 
    df3 = pd.read_csv("baseline/baseline_results_task3.csv") 
    df4 = pd.read_csv("baseline/baseline_results_task4.csv") 
    human_csv="data/t2_bin_preference/amazon_reviews/raw_review_Movies_and_TV/all.csv"
    entropies = entropy_per_column_csv("baseline/baseline_results_task1.csv")
    binpref = evaluate_binpref_col("baseline/baseline_results_task2.csv", human_csv)
    print("=== Column-wise Entropy ===")
    for col, ent in entropies.items():
        print(f"{col}: {ent:.3f}")
    print("=== Column-wise BINREF ===")
    for col, bf in binpref.items():
        print(f"{col}: {bf:.3f}")
    
    
    task3_results = evaluate_openpref_columns("baseline/baseline_results_task3.csv")
    print("=== Task 3: OpenPref ===")
    for col, metrics in task3_results.items():
        print(f"{col}: Num Aspects={metrics['num_aspects']}, "
        f"Aspect Entropy={metrics['aspect_entropy']:.3f}, "
        f"Sentiment Entropy={metrics['sentiment_entropy']:.3f}")

    csv_file = "baseline/baseline_results_task4.csv"
    task4_diversity = evaluate_recrequest_diversity(csv_file)

    for col, metrics in task4_diversity.items():
        print(f"{col}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")


    scores = evaluate_negative_feedback("baseline3n/baseline_results_task1.csv")
    print(scores)
    