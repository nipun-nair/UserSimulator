import math
import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
from scipy.stats import entropy, pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import string
#import nltk
#nltk.download('brown')
#nltk.download('punkt_tab')
from tqdm import tqdm
from textblob import TextBlob
from gensim.models import Word2Vec
import os
import re
from collections import defaultdict
from gensim.models import KeyedVectors

from profile_creation import OllamaHelper

glove_model = KeyedVectors.load_word2vec_format("glove.6B.50d.word2vec.txt", binary=False)

class SimulatorEvaluator:
    def __init__(self):
        pass

    # ========== 1. ItemsTalk Task: Entropy of Mentioned Items ==========
    def eval_items_talk(self, sim_jsonl, human_csv):
        def normalize_title(title):
            # Lowercase and strip whitespace and trailing punctuation for consistency
            if not isinstance(title, str):
                return None
            title = title.lower().strip()
            title = re.sub(r'[\s' + string.punctuation + r']+$', '', title)  # remove trailing punctuation
            return title if title else None

        def extract_titles_from_sim(jsonl_path):
            titles = []

            # Regex pattern to find likely movie titles with optional years, e.g. 'Movie Name (2015)', possibly with extra info in parentheses
            movie_pattern = re.compile(
                r'([\w\s\'\-:,\.&\?!]+?\([0-9]{4}[^\)]*\))', 
                flags=re.IGNORECASE
            )

            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    resp = data.get('response', '')

                    # Split the response by newlines and analyze each line
                    for entry in resp.split('\n'):
                        entry = entry.strip()
                        if not entry:
                            continue

                        # Remove leading numbering if present (e.g., "1. ", "2. ")
                        entry = entry.lstrip('0123456789. ').strip()

                        # First try to extract movie titles using the regex pattern
                        matches = movie_pattern.findall(entry)

                        # If we have matches, normalize and add each one
                        if matches:
                            for m in matches:
                                normalized = normalize_title(m)
                                if normalized:
                                    titles.append(normalized)
                            continue  # go to next line after processing matches

                        # If no match, check if the entire line looks like a movie title (fallback)
                        if "(" in entry and ")" in entry and len(entry) < 100:
                            normalized = normalize_title(entry)
                            if normalized:
                                titles.append(normalized)

            return titles
        def extract_titles_from_human(csv_path):
            titles = []
            df = pd.read_csv(csv_path)[:100]
            for row in df.itertuples():
                test_answer = getattr(row, 'test_answer', None)
                if test_answer:
                    try:
                        # Parse the string representation of the list
                        movie_list = ast.literal_eval(test_answer)
                        if not isinstance(movie_list, list):
                            continue
                        for movie in movie_list:
                            normalized = normalize_title(movie)
                            if normalized:
                                titles.append(normalized)
                    except Exception as e:
                        # Optionally log the error: print(f"Failed to parse: {test_answer}, error: {e}")
                        continue
            return titles

        def compute_entropy(titles):
            if not titles:
                # Avoid log(0) scenario, return zero entropy for empty input
                return 0.0
            counts = pd.Series(titles).value_counts()
            probs = counts / counts.sum()
            ent = entropy(probs, base=2)
            max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
            #ent /= max_entropy
            return ent

        sim_titles = extract_titles_from_sim(sim_jsonl)
        human_titles = extract_titles_from_human(human_csv)

        sim_entropy = compute_entropy(sim_titles)
        human_entropy = compute_entropy(human_titles)

        print("\n=== ItemsTalk Task ===")
        print(f"Simulator item entropy: {sim_entropy:.3f}")
        print(f"Human item entropy:     {human_entropy:.3f}")
        return sim_entropy, human_entropy
    # ========== 2. BinPref Task: Pearson Correlation ==========
    def eval_bin_pref(self, sim_jsonl, human_csv):
        with open(sim_jsonl) as f:
            sim_data = [json.loads(line) for line in f]
        sim_df = pd.DataFrame(sim_data)
        human_df = pd.read_csv(human_csv)

        pos_rates = []
        for entry in sim_data:
            responses = entry.get('user_responses', [])
            if not responses:
                continue
            yes_count = sum(str(resp).strip().lower().rstrip('.') == "yes" for resp in entry['user_responses'])
            pos_rate = yes_count / len(entry['user_responses'])
            pos_rates.append({'title': entry['title'], 'sim_pos_rate': pos_rate})
        sim_pos_df = pd.DataFrame(pos_rates)

        # Merge on movie title
        merged = pd.merge(human_df, sim_pos_df, left_on='clean_title', right_on='title', how='inner')
        if len(merged) < 2:
            print("Not enough data for correlation.")
            return None, None

        # Correlate human avg_rating with simulator positive rate
        corr = pearsonr(merged['avg_rating'], merged['sim_pos_rate'])[0]
        print("\n=== BinPref Task ===")
        print(f"Pearson(avg_rating, sim_pos_rate): {corr:.3f}")
        return corr

    def aspect_extractor(self, text):
        """
        Extracts noun phrases as aspects and their sentiment using TextBlob.
        Returns a list of (aspect, sentiment) tuples.
        """
        blob = TextBlob(text)
        aspects = []
        for phrase in blob.noun_phrases:
            for sentence in blob.sentences:
                if phrase in sentence:
                    polarity = sentence.sentiment.polarity
                    if polarity > 0.1:
                        sentiment = 'positive'
                    elif polarity < -0.1:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                    aspects.append((phrase, sentiment))
        return aspects

    def eval_open_pref(self, sim_jsonl, human_csv):
        def get_aspect_sentiments(jsonl_path, is_csv=False):
            aspect_sentiments = {}
            aspect_counts = []
            sentiment_counts = []
            if is_csv:
                df = pd.read_csv(jsonl_path)[:100]
                df = df.dropna(subset=['review'])

                for row in df.itertuples():
                    aspects = self.aspect_extractor(getattr(row, 'review'))
                    aspect_counts.extend([a for a, s in aspects])
                    sentiment_counts.extend([s for a, s in aspects])
                    for aspect, sentiment in aspects:
                        if aspect not in aspect_sentiments:
                            aspect_sentiments[aspect] = []
                        aspect_sentiments[aspect].append(sentiment)
            else:
                with open(jsonl_path) as f:
                    for line in f:
                        data = json.loads(line)
                        aspects = self.aspect_extractor(data['response'])
                        aspect_counts.extend([a for a, s in aspects])
                        sentiment_counts.extend([s for a, s in aspects])
                        for aspect, sentiment in aspects:
                            if aspect not in aspect_sentiments:
                                aspect_sentiments[aspect] = []
                            aspect_sentiments[aspect].append(sentiment)
            return aspect_sentiments, aspect_counts, sentiment_counts

        sim_aspects, sim_aspect_list, sim_sentiment_list = get_aspect_sentiments(sim_jsonl)
        human_aspects, human_aspect_list, human_sentiment_list = get_aspect_sentiments(human_csv, is_csv=True)
        

        # 1. Aspect entropy
        sim_aspect_entropy = entropy(pd.Series(sim_aspect_list).value_counts(normalize=True), base=2)
        human_aspect_entropy = entropy(pd.Series(human_aspect_list).value_counts(normalize=True), base=2)

        # 2. Sentiment entropy
        sim_sentiment_entropy = entropy(pd.Series(sim_sentiment_list).value_counts(normalize=True), base=2)
        human_sentiment_entropy = entropy(pd.Series(human_sentiment_list).value_counts(normalize=True), base=2)

        print("\n=== OpenPref Task ===")
        print(f"Aspect entropy:   Simulator={sim_aspect_entropy:.3f}, Human={human_aspect_entropy:.3f}")
        print(f"Sentiment entropy:Simulator={sim_sentiment_entropy:.3f}, Human={human_sentiment_entropy:.3f}")
        return sim_aspect_entropy, human_aspect_entropy, sim_sentiment_entropy, human_sentiment_entropy

        
    # ========== 4. RecRequest Task: Diversity and Granularity ==========
    def eval_rec_request(self, sim_jsonl, human_csv, max_requests=500):
        # --- Helper: Load requests ---
        def get_requests(jsonl_path, field='response', max_n=None):
            requests = []
            with open(jsonl_path) as f:
                for i, line in enumerate(f):
                    if max_n and i >= max_n:
                        break
                    data = json.loads(line)
                    requests.append(data[field])
            return requests

        # --- Load data (limit to max_requests) ---
        sim_requests = get_requests(sim_jsonl, max_n=max_requests)
        human_df = pd.read_csv(human_csv)
        human_requests = human_df['request'].astype(str).tolist()[:max_requests]

        # --- Word Diversity: Type-Token Ratio (TTR) ---
        def type_token_ratio(texts):
            tokens = ' '.join(texts).split()
            types = set(tokens)
            return len(types) / len(tokens) if len(tokens) > 0 else 0

        # --- Word Embedding Diversity: Word2Vec ---
        def word2vec_diversity(texts, model):
            tokenized = [str(t).split() for t in texts if str(t).strip()]
            vocab = set(word for sent in tokenized for word in sent if word in model)
            if len(vocab) < 2:
                return 0
            vectors = np.array([model[word] for word in vocab])
            sim_matrix = cosine_similarity(vectors)
            n = len(vocab)
            avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))
            return 1 - avg_sim


        # --- Sentence Embedding Diversity: SBERT (batched for memory safety) ---
        def cosine_diversity(texts, model_name='all-MiniLM-L6-v2', batch_size=64):
            texts = [str(t) for t in texts if str(t).strip()]
            if len(texts) < 2:
                return 0
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="SBERT Batching"):
                batch = texts[i:i+batch_size]
                batch_emb = model.encode(batch)
                embeddings.append(batch_emb)
            embeddings = np.vstack(embeddings)
            sim_matrix = cosine_similarity(embeddings)
            n = len(texts)
            avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))
            return 1 - avg_sim

            # --- Calculate metrics ---
        sim_ttr = type_token_ratio(sim_requests)
        human_ttr = type_token_ratio(human_requests)
        sim_w2v_div = word2vec_diversity(sim_requests, glove_model)
        human_w2v_div = word2vec_diversity(human_requests, glove_model)
        sim_sent_div = cosine_diversity(sim_requests)
        human_sent_div = cosine_diversity(human_requests)

        # --- Print results ---
        print("\n=== RecRequest Task ===")
        print(f"Word Diversity (TTR):             Simulator={sim_ttr:.3f}, Human={human_ttr:.3f}")
        print(f"Word Embedding Diversity (W2V):   Simulator={sim_w2v_div:.3f}, Human={human_w2v_div:.3f}")
        print(f"Sentence Embedding Diversity:     Simulator={sim_sent_div:.3f}, Human={human_sent_div:.3f}")

        return {
            'sim_ttr': sim_ttr, 'human_ttr': human_ttr,
            'sim_w2v_div': sim_w2v_div, 'human_w2v_div': human_w2v_div,
            'sim_sent_div': sim_sent_div, 'human_sent_div': human_sent_div
        }
    
    # ========== 6. Negative Feedback Task: Response Evaluation ==========
    def evaluate_negative_feedback(self, jsonl_path):
        scores = []

        # Read JSONL file
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                response_text = data.get("response", "")

                # Build evaluation prompt
                eval_prompt = """You are an evaluator. Score the simulator's response: {response_text}
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
                # Call the LLM evaluator
                self.ollama = OllamaHelper(model="llama3.3:70b")
                llm_output = self.ollama.call_model(eval_prompt)

                # Extract the first integer between 0 and 10
                score = None
                for token in llm_output.strip().split():
                    if token.isdigit():
                        val = int(token)
                        if 0 <= val <= 10:
                            score = val
                            break

                if score is not None:
                    scores.append(score)
                else: 
                    scores.append(0)

        # Calculate average
        avg_score = np.mean(scores) if scores else 0.0
        print(f"Average score: {avg_score:.2f}")
        return avg_score, scores

# ========== Example Usage ==========

if __name__ == "__main__":
    evaluator = SimulatorEvaluator()
    print("Simulator pre_prompt_tuning")
    # 1. ItemsTalk

    pre_prompt_tuning_sim_entropy, human_entropy = evaluator.eval_items_talk(
        sim_jsonl="data/t1_items/generated/llama3.3:70b/amazon_reviews/raw_review_Movies_and_TV/responses_0-100_test.jsonl",
        human_csv="data/t1_items/amazon_reviews/raw_review_Movies_and_TV/processed.csv"
    )
    
    # 2. BinPref
    pre_prompt_tuning_corr = evaluator.eval_bin_pref(
        sim_jsonl="data/t2_bin_preference/generated/llama3.3:70b/raw_review_Movies_and_TV/responses_0-100_test.jsonl",
        human_csv="data/t2_bin_preference/amazon_reviews/raw_review_Movies_and_TV/all.csv"
    )
    
    # 3. OpenPref (uses ABSA for aspect and sentiment extraction)
    pre_prompt_tuning_num_sim_aspects, num_human_aspects,  pre_prompt_tuning_sim_aspect_entropy, human_aspect_entropy, pre_prompt_tuning_sim_sentiment_entropy, human_sentiment_entropy = evaluator.eval_open_pref(
        sim_jsonl="data/t3_open_preference/generated/llama3.3:70b/raw_review_Movies_and_TV/responses_0-100_test.jsonl",
        human_csv="data/t3_open_preference/amazon_reviews/raw_review_Movies_and_TV/reviews_small.csv"
    )
    # 4. RecRequest
    preprompt_metrics = evaluator.eval_rec_request(
        sim_jsonl="data/t4_requests/generated/llama3.3:70b/raw_review_Movies_and_TV/responses_0-100_test.jsonl",
        human_csv="data/t4_requests/amazon_reviews/raw_review_Movies_and_TV/requests.csv"
    )
    preprompt_sim_ttr = preprompt_metrics['sim_ttr']
    human_ttr = preprompt_metrics['human_ttr']
    preprompt_sim_w2v_div = preprompt_metrics['sim_w2v_div']
    human_w2v_div = preprompt_metrics['human_w2v_div']
    preprompt_sim_sent_div = preprompt_metrics['sim_sent_div']
    human_sent_div = preprompt_metrics['human_sent_div']
    
    
    

   
    print("Prompt tuning evaluation")
    
    
    # 1. ItemsTalk
    post_prompt_tuning_sim_entropy, human_entropy = evaluator.eval_items_talk(
        sim_jsonl="data/t1_items/generated/llama3.3:70b/amazon_reviews/raw_review_Movies_and_TV/prompt_tuning/responses_0-100_test.jsonl",
        human_csv="data/t1_items/amazon_reviews/raw_review_Movies_and_TV/processed.csv"
    )
    
    # 2. BinPref
    post_prompt_tuning_corr = evaluator.eval_bin_pref(
        sim_jsonl="data/t2_bin_preference/generated/llama3.3:70b/raw_review_Movies_and_TV/prompt_tuning/responses_0-100_test.jsonl",
        human_csv="data/t2_bin_preference/amazon_reviews/raw_review_Movies_and_TV/all.csv"
    )
    
    # 3. OpenPref (uses ABSA for aspect and sentiment extraction)
    post_prompt_tuning_num_sim_aspects, num_human_aspects,  post_prompt_tuning_sim_aspect_entropy, human_aspect_entropy, post_prompt_tuning_sim_sentiment_entropy, human_sentiment_entropy = evaluator.eval_open_pref(
        sim_jsonl="data/t3_open_preference/generated/llama3.3:70b/raw_review_Movies_and_TV/prompt_tuning/responses_0-100_test.jsonl",
        human_csv="data/t3_open_preference/amazon_reviews/raw_review_Movies_and_TV/reviews_small.csv"
    )
    
    # 4. RecRequest
    post_prompt_metrics = evaluator.eval_rec_request(
        sim_jsonl="data/t4_requests/generated/llama3.3:70b/raw_review_Movies_and_TV/prompt_tuning/responses_0-100_test.jsonl",
        human_csv="data/t4_requests/amazon_reviews/raw_review_Movies_and_TV/requests.csv"
    )
    postprompt_sim_ttr = post_prompt_metrics['sim_ttr']
    human_ttr = post_prompt_metrics['human_ttr']
    postprompt_sim_w2v_div = post_prompt_metrics['sim_w2v_div']
    human_w2v_div = post_prompt_metrics['human_w2v_div']
    postprompt_sim_sent_div = post_prompt_metrics['sim_sent_div']
    human_sent_div = post_prompt_metrics['human_sent_div']
    
    # 6. Negative Feedback
    pre_prompt_tuning_rating, pre_prompt_tuning_scores = evaluator.evaluate_negative_feedback("data/t6_feedback/generated/llama3.3:70b/amazon_reviews/raw_review_Movies_and_TV/responses_0-100_limit.jsonl")

    post_prompt_tuning_rating, post_prompt_tuning_scores = evaluator.evaluate_negative_feedback("data/t6_feedback/generated/llama3.3:70b/amazon_reviews/raw_review_Movies_and_TV/prompt_tuning/responses_0-100_limit.jsonl")
    

    print("Rating before Prompt tuning ", pre_prompt_tuning_rating,"\n")
    print("Rating after Prompt tuning ", post_prompt_tuning_rating,"\n")

    print("Scores before Prompt tuning ", pre_prompt_tuning_scores,"\n")
    print("Scores after Prompt tuning ", post_prompt_tuning_scores,"\n")
