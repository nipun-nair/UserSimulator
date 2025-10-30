import random
import pandas as pd
from openai import OpenAI
from rank_bm25 import BM25Okapi
import os
import json

os.environ['OPENAI_API_KEY'] = "your_openai_api_key_here"
    
client = OpenAI()

# ========== RETRIEVER ==========

class BM25Retriever:
    def __init__(self, history_docs):
        """
        Args:
            history_docs: list of strings (not DataFrame!)
        """
        self.history = history_docs
        tokenized_corpus = [doc.lower().split() for doc in history_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query, k=5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(scores, self.history), reverse=True)
        return [doc for _, doc in ranked[:k]]


# ========== BASELINES ==========

def truncate_prompt(text, max_chars=2000*4):
    """Rough char-to-token estimate"""
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def run_gpt(query, model="gpt-4"):
    messages = [{"role": "user", "content": query}]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def run_icl_random(query, history_docs, k=5, model="gpt-4"):
    """
    Args:
        history_docs: list of strings
    """
    sampled = random.sample(history_docs, min(k, len(history_docs)))
    prompt = "User history:\n" + "\n".join(sampled) + "\n\nQuestion:\n" + query
    prompt = truncate_prompt(prompt)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def run_rag(query, retriever, k=5, model="gpt-4"):
    topk = retriever.retrieve(query, k)
    prompt = "Relevant history:\n" + "\n".join(topk) + "\n\nQuestion:\n" + query
    prompt = truncate_prompt(prompt)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def run_pag(query, history_docs, retriever, k=5, model="gpt-4"):
    """
    Args:
        history_docs: list of strings
        retriever: BM25Retriever instance
    """
    # Step 1: Generate profile summary (use sample to reduce cost)
    sample_size = min(10, len(history_docs))
    sampled_history = random.sample(history_docs, sample_size)
    summary_prompt = "Summarize the user's behavior patterns from the history:\n" + "\n".join(sampled_history)
    summary_prompt = truncate_prompt(summary_prompt)
    
    summary = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": summary_prompt}]
    ).choices[0].message.content
    
    # Step 2: Retrieve k items
    retrieved = retriever.retrieve(query, k)

    # Step 3: Final query
    prompt = (
        f"User profile summary:\n{summary}\n\n"
        f"Retrieved relevant history:\n" + "\n".join(retrieved) +
        f"\n\nQuestion:\n{query}"
    )
    prompt = truncate_prompt(prompt)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


# ========== EXPERIMENT RUNNER ==========

def run_experiment(queries, history_file, k=3):
    """
    Args:
        queries: list of query strings
        history_file: path to CSV file
        k: number of items to retrieve
    """
    # Load CSV
    df = pd.read_csv(history_file)
    print(f"Loaded {history_file}: {len(df)} rows, {df.shape[1]} columns")
    
    # Convert DataFrame to list of strings for BM25 and other methods
    history_docs = df.astype(str).agg(' | '.join, axis=1).tolist()
    
    # Create retriever with the list of strings
    retriever = BM25Retriever(history_docs)
    
    results = []
    for i, q in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}...")
        res = {
            "query": q,
            "GPT": run_gpt(q),
            "ICL-Random": run_icl_random(q, history_docs, k),
            "RAG": run_rag(q, retriever, k),
            "PAG": run_pag(q, history_docs, retriever, k)
        }
        results.append(res)

    return pd.DataFrame(results)


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    
    # Define user history files
    user_history = [
        "data/t1_items/amazon_reviews/raw_review_Movies_and_TV/processed.csv",
        "data/t2_bin_preference/amazon_reviews/raw_review_Movies_and_TV/all.csv",
        "data/t3_open_preference/amazon_reviews/raw_review_Movies_and_TV/reviews_small.csv",
        "data/t4_requests/amazon_reviews/raw_review_Movies_and_TV/requests.csv"
       "data/t6_feedback/amazon_reviews/raw_review_Movies_and_TV/items.csv"
    ]

    # Load prompts from JSONL files
    prompts = []
    file_paths = [
        "data/t1_items/generated/llama3.3:70b/amazon_reviews/raw_review_Movies_and_TV/responses_0-100_test.jsonl",
        "data/t2_bin_preference/generated/llama3.3:70b/raw_review_Movies_and_TV/responses_0-100_test.jsonl",
        "data/t3_open_preference/generated/llama3.3:70b/raw_review_Movies_and_TV/responses_0-100_test.jsonl",
        "data/t4_requests/generated/llama3.3:70b/raw_review_Movies_and_TV/responses_0-100_test.jsonl"
        "data/t6_feedback/generated/llama3.3:70b/amazon_reviews/raw_review_Movies_and_TV/responses_0-100_limit.jsonl"
    ]
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                if "prompt" in data:
                    prompts.append(data["prompt"])

    # Convert queries into 2D array (tasks × queries)
    num_tasks = len(user_history)
    queries_per_task = 100

    user_queries_2d = [
        prompts[i*queries_per_task:(i+1)*queries_per_task]
        for i in range(num_tasks)
    ]

    # Run experiment for each task and store results
    os.makedirs("baseline4n", exist_ok=True)
    
    for task_idx, history_file in enumerate(user_history):
        print(f"\n{'='*60}")
        print(f"Starting Task {task_idx+1}")
        print(f"{'='*60}")
        
        queries = user_queries_2d[task_idx]
        
        # Run experiment
        df = run_experiment(queries, history_file, k=3)
        
        # Rename columns to indicate task
        df.columns = [f"Task{task_idx+1}_{col}" for col in df.columns]
        
        # Save each task separately
        filename = f"baseline4n/baseline_results_task{task_idx+1}.csv"
        df.to_csv(filename, index=False)
        print(f"\nSaved Task {task_idx+1} results to {filename}")