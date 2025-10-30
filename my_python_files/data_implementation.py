import pandas as pd
import json
import random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
import re


class AmazonReviewProcessor:
    def __init__(self, subset='raw_review_Movies_and_TV', dataset_name='McAuley-Lab/Amazon-Reviews-2023'):
        self.subset = subset
        self.dataset_name = dataset_name
        self.df = None

    def load_data(self):
        print(f"Loading {self.subset} from {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, self.subset, trust_remote_code=True)
        self.df = dataset['full'].to_pandas()

        # Select relevant columns and clean strings
        self.df = self.df[['user_id', 'asin', 'rating', 'title', 'text', 'parent_asin']].dropna()
        self.df['title'] = self.df['title'].str.strip()
        self.df['text'] = self.df['text'].str.strip().str.replace(r'\s+', ' ', regex=True)

        # Load raw metadata for title-year mapping
        meta = load_dataset(self.dataset_name, 'raw_meta_Movies_and_TV', split="full")

        # Build mapping: parent_asin -> "Title (Year)" if year found
        asin2title = {}
        for entry in meta:
            asin = entry.get('parent_asin')
            raw_title = entry.get('title')
            title = raw_title.strip() if isinstance(raw_title, str) else ''

            features = entry.get('features')
            if not isinstance(features, list):
                features = []

            year = None
            for f in features:
                if isinstance(f, str):
                    match = re.search(r'(19|20)\d{2}', f)
                    if match:
                        year = match.group()
                        break

            if asin and title and year:
                formatted = f"{title} ({year})"
                asin2title[asin] = formatted

        # Map clean_title in main df using asin2title mapping (based on 'asin' column)
        self.df['clean_title'] = self.df['asin'].map(asin2title)
        self.df = self.df.dropna(subset=['clean_title'])  # Drop rows where mapping failed

        print(f"Mapped {self.df['clean_title'].ne(self.df['asin']).sum()} out of {len(self.df)} rows to clean titles.")
        print(self.df[['clean_title']].sample(10))
        print(f"Loaded {len(self.df)} rows.")

    def preprocess_all(self):
        self.load_data()
        self._preprocess_t1_items()
        self._preprocess_t2_bin_preference()
        self._preprocess_t3_open_preference()
        self._preprocess_t4_requests()
        self._preprocess_t6_feedback()
    

    def _group_by_user(self):
        # Group reviews by user_id, using clean_title for consistent naming
        user_reviews = defaultdict(list)
        for _, row in self.df.iterrows():
            user_reviews[row['user_id']].append({
                "title": row['clean_title'],
                "review": row['text']
            })
        return user_reviews

    def _preprocess_t1_items(self, anchor_count=10, seed=42):
        print("[T1] Processing anchor + test format...")
        save_dir = Path(f'data/t1_items/amazon_reviews/{self.subset}')
        save_dir.mkdir(parents=True, exist_ok=True)

        title2count = defaultdict(int)
        processed = []
        user_reviews = self._group_by_user()
        random.seed(seed)

        for user_id, reviews in tqdm(user_reviews.items(), desc="Processing users for T1"):
            if len(reviews) < anchor_count + 1:
                continue

            reviews_copy = reviews[:]
            random.shuffle(reviews_copy)

            # Anchor: first anchor_count reviews, join title and review text with colon
            anchor_lines = [f"{r['title']}: {r['review']}" for r in reviews_copy[:anchor_count]]
            anchor_str = '\n'.join(anchor_lines)

            test_reviews = reviews_copy[anchor_count:]
            test_answer = [r['title'] for r in test_reviews]
            for t in test_answer:
                title2count[t] += 1

            if not test_answer:
                continue

            # Full history titles from original order (unshuffled); use clean_title
            full_history_titles = [r['title'] for r in reviews]

            processed.append({
                "user_id": user_id,
                "anchor_str": anchor_str,
                "test_answer": json.dumps(test_answer),
                "history_titles": json.dumps(full_history_titles)
            })

        df_out = pd.DataFrame(processed)
        df_out.to_csv(save_dir / 'processed.csv', index=False)

        # Save title count dictionary for analysis
        with open(save_dir / 'title2count.json', 'w') as f:
            json.dump(title2count, f, indent=4)

        print(f"[T1] Done. Processed {len(df_out)} users.")

    def _preprocess_t2_bin_preference(self):
        print("[T2] Processing frequent/infrequent items...")
        save_dir = Path(f'data/t2_bin_preference/amazon_reviews/{self.subset}')
        save_dir.mkdir(parents=True, exist_ok=True)

        grouped = self.df.groupby("clean_title").agg(
            counts=("text", "count"),
            avg_rating=("rating", "mean")
        ).reset_index()
        grouped['avg_rating'] = grouped['avg_rating'].round(2)

        freq = grouped[grouped['counts'] >= 20].sample(n=min(100, len(grouped)), random_state=0)
        infreq = grouped[(grouped['counts'] >= 2) & (grouped['counts'] <= 5)].sample(n=min(100, len(grouped)), random_state=0)
        all_sample = grouped.sample(n=min(200, len(grouped)), random_state=0)

        freq.to_csv(save_dir / 'frequent.csv', index=False)
        infreq.to_csv(save_dir / 'infrequent.csv', index=False)
        all_sample.to_csv(save_dir / 'all.csv', index=False)

    def _preprocess_t3_open_preference(self):
        print("[T3] Generating reviews_large and reviews_small...")
        save_dir = Path(f'data/t3_open_preference/amazon_reviews/{self.subset}')
        save_dir.mkdir(parents=True, exist_ok=True)

        user_reviews = self._group_by_user()
        reviews_large = [r for reviews in user_reviews.values() for r in reviews]
        reviews_small = [random.choice(reviews) for reviews in user_reviews.values() if reviews]

        pd.DataFrame(reviews_large).to_csv(save_dir / 'reviews_large.csv', index=False)
        pd.DataFrame(reviews_small).to_csv(save_dir / 'reviews_small.csv', index=False)

    def _preprocess_t4_requests(self):
        print("[T4] Simulating requests...")
        save_dir = Path(f'data/t4_requests/amazon_reviews/{self.subset}')
        save_dir.mkdir(parents=True, exist_ok=True)

        user_reviews = self._group_by_user()
        rows = []

        for uid, reviews in user_reviews.items():
            titles = list({r["title"] for r in reviews})
            if len(titles) < 2:
                continue
            request = f"I'm looking for movies similar to {titles[0]} and {titles[1]}"
            rows.append({
                "user_id": uid,
                "request": request,
                "extracted_names": titles
            })

        df = pd.DataFrame(rows)
        df.to_csv(save_dir / 'requests.csv', index=False)

    def _preprocess_t5_feedback(self):
        print("[T5] Simulating feedback task...")
        save_dir = Path(f'data/t5_feedback/amazon_reviews/{self.subset}')
        save_dir.mkdir(parents=True, exist_ok=True)

        df_positive = self.df[self.df['rating'] >= 4]
        title_to_texts = df_positive.groupby('clean_title')['text'].apply(list).to_dict()
        all_titles = [t for t, texts in title_to_texts.items() if texts]

        user_reviews = df_positive.groupby('user_id').apply(
            lambda x: x[['clean_title', 'text']].to_dict(orient='records'),
            include_groups=False
        ).to_dict()

        filtered_users = {uid: revs for uid, revs in tqdm(user_reviews.items(), desc="Filtering users with ≥3 positive reviews") if len(revs) >= 3}
        print(f"Users with >=3 positive reviews: {len(filtered_users)}")

        items_data = []
        context_data = []

        for uid, reviews in tqdm(filtered_users.items(), desc="Generating feedback samples"):
            request = f"I enjoyed {reviews[0]['clean_title']} and {reviews[1]['clean_title']}. What should I watch next?"
            positive_title = reviews[2]['clean_title']

            candidate_titles = [t for t in all_titles if t != positive_title]
            if not candidate_titles:
                continue

            random_title = random.choice(candidate_titles)
            random_text = random.choice(title_to_texts[random_title])

            items_data.append({
                "user_id": uid,
                "request": request,
                "first": positive_title,
                "random": random_title
            })

            context_data.append({
                "user_id": uid,
                "request": request,
                "first": reviews[2]['text'],
                "random": random_text
            })

        if items_data and context_data:
            pd.DataFrame(items_data).to_csv(save_dir / 'items.csv', index=False)
            pd.DataFrame(context_data).to_csv(save_dir / 'context.csv', index=False)
            print(f"Saved {len(items_data)} feedback samples.")
        else:
            print("No data to save. Check filtering criteria or input data.")
    
    def _preprocess_t6_feedback(self):
        print("[T6] Simulating feedback task with low-rated negatives...")
        save_dir = Path(f'data/t6_feedback/amazon_reviews/{self.subset}')
        save_dir.mkdir(parents=True, exist_ok=True)

        # Positive and negative pools
        df_positive = self.df[self.df['rating'] >= 4]
        df_negative = self.df[self.df['rating'] < 3]

       

        # All positive titles per user
        user_reviews_pos = df_positive.groupby('user_id').apply(
            lambda x: x[['clean_title', 'text']].to_dict(orient='records'),
            include_groups=False
        ).to_dict()

        # Filter users with ≥3 positive reviews
        filtered_users = {
            uid: revs for uid, revs in tqdm(user_reviews_pos.items(),desc="Filtering users with ≥3 positive reviews")
            if len(revs) >= 3
        }
        print(f"Users with >=3 positive reviews: {len(filtered_users)}")

        items_data = []
        neg_titles = df_negative['clean_title'].unique().tolist()

        for uid, reviews in tqdm(filtered_users.items(), desc="Generating feedback samples"):
            # List of all positive titles(year) for this user
            positive_titles = [r['clean_title'] for r in reviews]

            # Pick one random negative title(year)
            if not neg_titles:
                continue
            random_title = random.choice(neg_titles)

            items_data.append({
                "user_id": uid,
                "positive_titles": positive_titles,   # list of all positive titles(year)
                "neg_title": random_title      # low-rated movie title(year)
            })

        if items_data:
            pd.DataFrame(items_data).to_csv(save_dir / 'items.csv', index=False)
            print(f"Saved {len(items_data)} feedback samples.")
        else:
            print("No data to save. Check filtering criteria or input data.")



if __name__ == "__main__":
    processor = AmazonReviewProcessor(subset='raw_review_Movies_and_TV')
    processor.preprocess_all()
