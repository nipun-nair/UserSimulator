import pandas as pd
from pathlib import Path
import argparse
import ast
from tqdm import tqdm
import random
import json
import sys
from profile_creation import OllamaHelper, RandomPerson
sys.path.append('.')
from textgrad_ollama import PromptTuning

class DataPipeline:
    def __init__(self, args):
        self.args = args
        self.ollama = OllamaHelper(model="llama3.3:70b")  #deepseek-r1:32b  llama3.3:70b llama3.3-70b-2k:latest deepseek-r1:70b
        self.person = []
        for _ in range(self.args.start, self.args.end):
            person = RandomPerson(self.ollama)
            self.person.append(person)

        self.subset = 'raw_review_Movies_and_TV'
        self.prompttuning = PromptTuning()

    def get_indices(self, df):
        start_idx = self.args.start if self.args.start is not None else 0
        end_idx = self.args.end if self.args.end is not None else len(df)
        print(len(df))
        assert (start_idx >= 0 and end_idx <= len(df))
        return start_idx, end_idx

    def save_jsonl(self, path, data):
        with open(path, 'w') as f:
            for response in data:
                f.write(json.dumps(response) + '\n')
        print(f"Saved: {path}")

    def task1_items(self, prompt_tuning=False):
        DIR = "data/t1_items"

        # Update save_dir based on prompt_tuning
        if prompt_tuning:
            save_dir = Path(f"{DIR}/generated/{self.ollama.model}/{self.args.target}/{self.subset}/prompt_tuning")
        else:
            save_dir = Path(f"{DIR}/generated/{self.ollama.model}/{self.args.target}/{self.subset}")

        save_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(f"{DIR}/{self.args.target}/{self.subset}/processed.csv")
        start_idx, end_idx = self.get_indices(df)
        
        generated_responses = []
        df = df.iloc[start_idx:end_idx]

        for i, row in tqdm(df.iterrows(), total=len(df)):
            test_answer = ast.literal_eval(row['test_answer'])
            history = ast.literal_eval(row['history_titles'])
            if len(history) > 5:
                selected = random.sample(history, 5)
            else:
                selected = history
            
            profile_summarization_prompt = f"Based on these 5 items {selected}, summarize the user's taste and behavior in simple English. Keep the description short."
            profile = self.ollama.call_model(profile_summarization_prompt)
            n_test = min(len(test_answer), 5)
            prompt = f"Pretend to be {self.person[i]} Profile: {profile}. You decide to talk about {n_test} movies."
            prompt += f" What would these {n_test} movies be? Reply as a list of <Title (yyyy)>. Say nothing else."

            # Pass prompt through process_example_textgrad if prompt_tuning is True
            if prompt_tuning:
                prompt = self.prompttuning.process_example_textgrad(prompt, 1)

            response = self.ollama.call_model(prompt)
            print("Inside for loop response : " + "task1\n")
            if response:
                generated_responses.append({'row_i': i, 'response': response, 'prompt': prompt})

            # Periodic saving and final save
            if response is None or (i % 1000 == 0 and i > start_idx) or i == end_idx - 1:
                save_path = f"{save_dir}/responses_{start_idx}-{i+1}_test.jsonl"
                self.save_jsonl(save_path, generated_responses)

            if response is None:
                break


    def task2_bin_preference(self, prompt_tuning=False):
        DIR = "data/t2_bin_preference"
        if prompt_tuning:
            save_dir = Path(f"{DIR}/generated/{self.ollama.model}/{self.subset}/prompt_tuning")
        else:
            save_dir = Path(f"{DIR}/generated/{self.ollama.model}/{self.subset}")
        save_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(f"{DIR}/{self.args.target}/{self.subset}/{self.args.data}.csv")
        start_idx, end_idx = self.get_indices(df)
        generated_responses = []
        df = df.iloc[start_idx:end_idx]

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Movies"):
            title = row['clean_title']
            rating = row['avg_rating'] #and rated it {rating} out of 5
            user_responses = []
            prompt = f"Pretend to be {self.person[i]}. You watched the movie {title} and rated it {rating} out of 5. Did you like the movie? Answer Yes or No. Don't say anything else."
            if prompt_tuning:
                prompt = self.prompttuning.process_example_textgrad(prompt, 2)
            for _ in range(10):
                response = self.ollama.call_model(prompt)
                print("Inside for loop response : " + "task2\n")  
                user_responses.append(response)
            generated_responses.append({
                'user_responses': user_responses,
                'title': title,
                'prompt': prompt
            })
        save_path = f"{save_dir}/responses_{start_idx}-{end_idx}_test.jsonl"
        self.save_jsonl(save_path, generated_responses)

    def task3_open_preference(self, prompt_tuning=False):
        DIR = "data/t3_open_preference"
        if prompt_tuning:
            save_dir = Path(f"{DIR}/generated/{self.ollama.model}/{self.subset}/prompt_tuning")
        else:
            save_dir = Path(f"{DIR}/generated/{self.ollama.model}/{self.subset}")
        save_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(f"{DIR}/{self.args.target}/{self.subset}/reviews_small.csv")
        start_idx, end_idx = self.get_indices(df)
        
        generated_responses = []
        df = df.iloc[start_idx:end_idx]
        for i, row in tqdm(df.iterrows(), total=len(df)):
            title = row['title']
            review_len = len(row['review'])
            prompt = f"Pretend to be {self.person[i]}. "
            prompt += f"You watched the movie {title}. What are your thoughts on this movie? Answer must not exceed {review_len} characters."
            if prompt_tuning:
                prompt = self.prompttuning.process_example_textgrad(prompt, 3)
            response = self.ollama.call_model(prompt)
            print("Inside for loop response : " + "task3\n")
            if response:
                generated_responses.append({
                    "response": response,
                    "target_len": review_len,
                    "actual_len": len(response),
                    "prompt": prompt
                })
            if response is None or (i % 500 == 0 and i > start_idx) or i == end_idx - 1:
                save_path = f"{save_dir}/responses_{start_idx}-{i+1}_test.jsonl"
                self.save_jsonl(save_path, generated_responses)
            if response is None:
                break

    def task4_requests(self, prompt_tuning=False):
        DIR = 'data/t4_requests'
        if prompt_tuning:
            save_dir = Path(f"{DIR}/generated/{self.ollama.model}/{self.subset}/prompt_tuning")
        else:
            save_dir = Path(f"{DIR}/generated/{self.ollama.model}/{self.subset}")
        save_dir.mkdir(exist_ok=True, parents=True)
        df = pd.read_csv(f"{DIR}/{self.args.target}/{self.subset}/requests.csv")
        start_idx, end_idx = self.get_indices(df)
        
        generated_responses = []
        df = df.iloc[start_idx:end_idx]
        for i, row in tqdm(df.iterrows(), total=len(df)):
            movies = row['extracted_names']
            if isinstance(movies, str) and movies.startswith('['):
                movies = ast.literal_eval(movies)  # string → list
            movies_str = movies[:2] 
            target_len = len(row['request'])
            prompt = f"Generate a movie recommendation request. Include (but do not request) the following movies in your text: {movies_str}. Make sure the length of the request is approximately {target_len} characters."
            if prompt_tuning:
                prompt = self.prompttuning.process_example_textgrad(prompt, 4)
            response = self.ollama.call_model(prompt)
            print("Inside for loop response : " + "task4\n")
            if response:
                generated_responses.append({
                    'response': response,
                    'movies_str': movies_str,
                    'actual_char_len': len(response),
                    'target_char_len': target_len,
                    'prompt':prompt
                })
            if response is None or (i % 1000 == 0 and i > start_idx) or i == end_idx - 1:
                save_path = f"{save_dir}/responses_{start_idx}-{i+1}_test.jsonl"
                self.save_jsonl(save_path, generated_responses)
            if response is None:
                break

    def task6_negative_feedback(self, prompt_tuning=False):
        DIR = "data/t6_feedback"

        # Update save_dir based on prompt_tuning
        if prompt_tuning:
            save_dir = Path(f"{DIR}/generated/{self.ollama.model}/{self.args.target}/{self.subset}/prompt_tuning")
        else:
            save_dir = Path(f"{DIR}/generated/{self.ollama.model}/{self.args.target}/{self.subset}")

        save_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(f"{DIR}/{self.args.target}/{self.subset}/items.csv")
        start_idx, end_idx = self.get_indices(df)
        
        generated_responses = []
        df = df.iloc[start_idx:end_idx]
        for i, row in tqdm(df.iterrows(), total=len(df)):
            history = ast.literal_eval(row['positive_titles'])
            neg_movie = row['neg_title']
            if len(history) > 5:
                selected = random.sample(history, 5)
            else:
                selected = history
            
            profile_summarization_prompt = f"Based on these 5 items {selected}, summarize the user's taste, behavior and preferences in simple English. Keep the description short."
            profile = self.ollama.call_model(profile_summarization_prompt)
            
            prompt = f"""You are now role-playing as {self.person[i]}. Your preferences are described below:{profile}. You received a movie recommendation: {neg_movie}. You do not like this recommendation.Your task: 1. Clearly reject the movie. 2. Explain why it doesn't match your preferences (e.g., tone, genre, theme). 3. Formulate your request to better express what you're looking for—without naming any specific movie. 4. Match your tone and reasoning style as described in your profile. 5. If you are unable to perform the task, respond with: "I don't know." """


            # Pass prompt through process_example_textgrad if prompt_tuning is True
            if prompt_tuning:
                prompt = self.prompttuning.process_example_textgrad(prompt, 6)

            response = self.ollama.call_model(prompt)
            print("Inside for loop response : " + "task6\n")
            if response:
                generated_responses.append({'row_i': i, 'response': response, 'prompt': prompt})

            # Periodic saving and final save
            if response is None or (i % 1000 == 0 and i > start_idx) or i == end_idx - 1:
                save_path = f"{save_dir}/responses_{start_idx}-{i+1}_limit.jsonl"
                self.save_jsonl(save_path, generated_responses)

            if response is None:
                break


    def run_all(self):
        
        #Task 1
        print("Running without prompt tuning")
        print("Running Task 1: Items")
        self.task1_items()
        
        print("Running with prompt tuning")
        print("Running Task 1: Items")
        self.task1_items(True)
        
        #Task 2
        print("Running without prompt tuning")
        print("Running Task 2: Bin Preference")
        self.task2_bin_preference()
        print("Running with prompt tuning")
        print("Running Task 2: Bin Preference")
        self.task2_bin_preference(True)
        
        #Task 3
        print("Running without prompt tuning")
        print("Running Task 3: Open Preference")
        self.task3_open_preference()
        print("Running with prompt tuning")
        print("Running Task 3: Open Preference")
        self.task3_open_preference(True)
        

        #Task 4 
        print("Running without prompt tuning")
        print("Running Task 4: Requests")
        self.task4_requests()
        print("Running with prompt tuning")
        print("Running Task 4: Requests")
        self.task4_requests(True)

        #Task 6
        print("Running without prompt tuning")
        print("Running Task 6: Negative Feedback")
        self.task6_negative_feedback()
        print("Running with prompt tuning")
        print("Running Task 6: Negative Feedback")
        self.task6_negative_feedback(True)
        
        
    
if __name__ == "__main__":
    print("Main")
    parser = argparse.ArgumentParser()
    # Add all possible arguments needed by any task
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--lm", type=str, default='llama3.3:70b')
    parser.add_argument("--target", type=str, default="amazon_reviews")
    parser.add_argument("--data", type=str, default="all")
    parser.add_argument("--rec", type=str, default="items", choices=["items", "context"])
    parser.add_argument("--action", type=str, default="reject", choices=["reject", "compare"])
    parser.add_argument("--ask_why", action='store_true', help='ask why such feedback was given')
    # Add a task selector if you want to run only specific tasks
    parser.add_argument("--tasks", nargs='+', type=int, default=[1,2,3,4,5], help="Tasks to run (1-5)")
    args = parser.parse_args()
    #os.environ['OPENAI_API_KEY'] = CONFIG["api_key"]
    pipeline = DataPipeline(args)
    # Run selected tasks
    '''
    for t in args.tasks:
        getattr(pipeline, f"task{t}_items" if t == 1 else f"task{t}_bin_preference" if t == 2 else f"task{t}_open_preference" if t == 3 else f"task{t}_requests" if t == 4 else f"task{t}_feedback")()
'''
    # Or, to run all in sequence:
    pipeline.run_all()
