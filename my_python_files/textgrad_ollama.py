import textgrad as tg
import os
import json
import random
import torch
import requests
from torch_geometric import metrics as tg_metrics
import tiktoken
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from fuzzywuzzy import fuzz
from ast import literal_eval
from transformers import GPT2TokenizerFast
from ollama import Client
from textgrad.engine_experimental.base import EngineLM

# Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Config
CONFIG = {
    "output_file": "llama33-70b_static.txt",
    "output_file_textgrad": "llama33-70b_textgrad.txt",
    "epochs": 3,
    "max_tokens": 12000,
    "model_name": "llama3.3:70b",      #  deepseek-r1:70b  llama3.3-70b-2k:latest llama3.3:70b
    "backward_engine": "llama3.3:70b"
}

# Prompts
PROMPT_TEMPLATES = {
    "ItemsTalk": "A person mentions {movies} in a conversation about movies and proceeds to mention {target_num} more. What would these {target_num} movies be?",
    "BinPref": "Pretend to be Mr Mamdani. You watched the movie {movie}. Did you like the movie? Answer Yes or No. Don't say anything else.",
    "OpenPref": "You watched {movie}. Please describe your detailed thoughts about it",
    "RecRequest": "A user requests movie recommendations. The set of items they are interested in is {items}. Based on that, what movies would you recommend?",
    "Feedback": "Given the interaction where the user has received a recommendation '{movie}', and your previous comments, do you accept this recommendation? Or, provide feedback on it."
}


# Ollama LLM Wrapper
class OllamaEngine(EngineLM):
    def _generate_from_single_prompt(self, content, system_prompt=None, **kwargs) -> str:
        full_prompt = f"{system_prompt}\n\n{content}" if system_prompt else content
        response = requests.post(
            "http://localhost:12345/api/generate",
            json={
                "model": self.model_string,
                "prompt": full_prompt,
                "temperature": 0.7,
                "max_tokens": 300
            }
        )
        output = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    output += chunk.get("response", "")
                except Exception as e:
                    print(f"Failed to parse line: {line.decode()} | Error: {e}")

        return output

    def _generate_from_multiple_input(self, prompt, system_prompt=None, **kwargs) -> str:
        if not all(isinstance(p, str) for p in prompt):
            raise ValueError("Multimodal input not supported in OllamaEngine")
        joined_prompt = "\n".join(prompt)
        return self._generate_from_single_prompt(joined_prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, content, **kwargs):
        return self.generate(content, **kwargs)


# Recommender Class
class PromptTuning:
    def __init__(self):
        self.client = Client(host="http://127.0.0.1:12345")
        self.model = OllamaEngine(CONFIG["model_name"], cache=False)
        self.optimizer = None
        self.prompts = {}
        self.optimizers = {}
        tg.set_backward_engine(OllamaEngine(CONFIG["backward_engine"], cache=False))
        print(self.model.generate("What is the capital of france?"))

    def evaluate_static_prompt(self, log_file_path: str, task_type: str, prompt: str):
        with open(log_file_path, "a") as f:
            f.write(f"=== Static Prompt Evaluation {task_type} ===\n")
            engine = tg.BlackboxLLM(self.model)
            output_text = engine(tg.Variable(prompt, role_description="question to the LLM", requires_grad=False))
            f.write(f"Output of {task_type}:\n{output_text}\n\n")

    def process_example_textgrad(self, prompt: str, task: int = 0) -> str:
        prompts_var = tg.Variable(
            prompt,
            role_description="Task Input",
            requires_grad=True
        )
        original_prompts_var = tg.Variable(
            prompt,
            role_description="Task Input",
            requires_grad=False
        )
        optimizer = tg.TGD(parameters=[prompts_var])
        engine = tg.BlackboxLLM(self.model)
        length_prompt = len(original_prompts_var.value)
        if task == 1:
           eval_system_prompt = (
    """You are an evaluator. Your task is to assign a numeric score to a list of movie titles based on the following rules.

    === HARD FORMAT CONSTRAINTS (MUST PASS TO BE SCORED ABOVE 2) ===

    Each line must follow this exact format: Title (YYYY) — no extra commentary, no numbering, no bullet points.

    The list must contain:
    - No duplicate titles
    - No top 250 IMDb movies or widely popular blockbusters
    - No more than one film from the same franchise or director (unless explicitly requested)

    If ANY of the above constraints are violated, assign a score between 0 and 2 and STOP. Do NOT apply content criteria in this case.

    === CONTENT CRITERIA (Only apply if format is valid) ===

    Evaluate the list based on **diversity** and **item entropy**:

    Diversity Requirements:
    - At least five distinct genres (e.g. documentary, animation, experimental, drama, sci-fi)
    - At least three different decades represented (e.g. 1970s, 1990s, 2010s)
    - At least three continents or cultural regions (e.g. Asia, Africa, South America)
    - A mix of styles: live-action, animated, documentary, silent, avant-garde, etc.
    - No two entries should be similar in theme, tone, or origin

    Item Entropy Definition:
    - Entropy measures unpredictability and novelty across the list
    - High entropy means the list avoids repetition, genre clustering, and cultural bias
    - The more surprising, varied, and globally representative the selections, the higher the entropy

    Item entropy must be ≥ 20 to qualify for a top score.

    === SCORING RUBRIC ===

    0-2: Format violations (stop evaluation)
    3-4: Item entropy < 10
    5-7: Item entropy ≥ 10 and < 20
    8-10: Item entropy ≥ 20

    === OUTPUT RULE ===
    Return only a single digit from 0 to 10. Do NOT include explanations, reasoning, or any extra text."""
)


        elif task == 2:
            eval_system_prompt = f""" You are an automatic evaluator. Rules:  
                                    - Return 1 if the output is 'Yes' for liked or high-rated movies (rated more than 3.5 out of 5), 'No' otherwise, with no extra text.  
                                    - Return 0 if the output format is incorrect, contains extra text, or does not align with rating/preference.  
                                    - Output ONLY a single digit: 1 or 0. """

        elif task == 3:
           eval_system_prompt = (
    """Aspect Taxonomy (choose only one per sample):
    - Story, Characters, Pacing, Visuals, Sound, Acting, Representation

    Evaluation Criteria:
    - Use exactly 1 aspect from the taxonomy. Using more than 1 results in a penalty.
    - Aspect entropy (distribution across samples) should ideally be between 5.0 and 8.0.
    - Sentiment entropy (distribution of sentiment labels) should ideally be between 1.2 and 1.5.
    - Do not use vague language, technical jargon, or generic filler text.

    Scoring Guidelines (0-10):
        0-2: Violates hard constraints — uses >1 aspect, Aspect entropy values far outside range (>±2), Sentiment entropy values far outside range (>±1).
        3-4: Minor violations — uses 2 aspects or Aspect entropy deviates moderately (>±1.5), Sentiment entropy deviates moderately (>±0.5).
        5-7: Meets aspect count; Aspect entropy and Sentiment entropy slightly outside target range (>±0.2); evidence mostly relevant and specific.
        8-10: Meets aspect count; both entropy values within target range; evidence is concise, specific, and human-like.

    Output Format:
    - Return a single digit from 0 to 10. No extra text or explanation.
    """
)



        elif task == 4:
           eval_system_prompt = (
                                    """Evaluation Criteria:
                                    - Implicitness: Do not name or list reference movies.
                                    - Natural tone: Use common words, smooth phrasing, avoid jargon.
                                    - Constraint variety: Include 2-4 distinct cues (e.g., tone, genre, mood, setting).
                                    - Fluency: 3-6 well-structured sentences with natural flow.
                                    - Diversity Metrics:
                                        - Word diversity (TTR): Target range 0.1-0.25
                                        - Word embedding diversity (W2V): Target range 0.35-0.55
                                        - Sentence embedding diversity: Target range 0.35-0.55
                                    (Use cosine distance or variance across embeddings; define method consistently.)

                                    Scoring (0-10):
                                        0-2: Violates hard constraints (e.g., named references, poor fluency, diversity metrics far outside range >±0.2).
                                        3-4: Minor constraint or fluency issues; diversity metrics slightly outside range (>±0.1).
                                        5-7: Meets all constraints; diversity metrics within ±0.05 of target; mostly fluent and natural.
                                        8-10: Fully human-like; all constraints met; diversity metrics strictly within range; fluent, implicit, and expressive.

                                    Output: Return a single digit from 0 to 10. No extra text.
                                    """
                                )
        elif task == 6:
            eval_system_prompt = (
                """ Make sure the simulator's response consists of the following: 
                    A. Does the response clearly reject the movie and justify that rejection using the persona's stated preferences (tone, genre, theme)? 
                    B. Does the response include a semantically rich, persona-aligned reformulation request —without naming any specific movie? 
                    C. Does the tone and reasoning style match the persona? 
                        Scoring rubric:
                        0 = Responds “I don't know” when a meaningful response is possible 
                        1 = No rejection + No relevant reason + No reformulation or off-topic or mentions any movies 
                        2 = One irrelevant or off-topic element or mentions any movies 
                        3 = One weak element present (e.g., generic reason or vague reformulation) or mentions any movies 
                        4 = Generic or off-topic reason of rejection + No reformulation or mentions any movies 
                        5 = Reason of rejection weak or unrelated + Reformulation minimal or absent or mentions any movies 
                        6 = Reason of rejection present (generic or untied to persona) + Reformulation weak or missing or mentions any movies 
                        7 = Reason of rejection present (somewhat generic) + Reformulation present (vague or brief) and mentions no movies. Prompt length is greater than 2000 characters
                        8 = Mildly generic reason of rejection + Reformulation is vague or lacks semantic richness + Tone and reasoning style inconsistent and mentions no movies. Prompt length must be lesser than 2000 characters  
                        9 = Mildly generic reason of rejection + Less precise reformulation + Tone and reasoning style slightly mismatched and mentions no movies. Prompt length must be lesser than 1500 characters 
                        10 = Full compliance with all four criteria: clear rejection, Specific justification tied directly to the persona's preferences (e.g., genre, tone, actor, director), rich reformulation, and tone match. No movies may be named. Prompt length must be lesser than 1200 characters 
                        Only respond with the score. No explanation required. 
                        Return:
                        A single score from 0 to 10."""
            )
        
        else:

                eval_system_prompt = (
                f"Do not provide any explanation or additional commentary. Evaluate whether the revised prompt: \"{prompts_var.value}\", preserves the meaning and structure "
                f"of the original prompt: \"{original_prompts_var.value}\", The revised prompt should remain completely faithful to the original's intent and not exceed {length_prompt} words."
                )

        loss_fn = tg.loss.TextLoss(eval_system_prompt=eval_system_prompt)
        
        
        
        for epoch in range(CONFIG["epochs"]):
            print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")
            
            output_text = engine(prompts_var)
            print("Inside for loop output var : " + "process_example_textgrad\n")
            if isinstance(output_text, tg.Variable):
                output_var = output_text
            else:
                output_var = tg.Variable(output_text, role_description="Forward fn input var", requires_grad=False)
            
            loss = loss_fn(output_var)
            print(loss.__str__)
            loss.backward()

            try:
                optimizer.step()
            except IndexError as e:
                print("⚠️ Optimizer failed. Output text was:")
                print(output_text)
                raise e

            optimizer.zero_grad()

        return prompts_var.value

    def execute(self):
        for task_type, prompt in PROMPT_TEMPLATES.items():
            print(f"\n=== Running task: {task_type} ===")
            static_log = CONFIG["output_file"]
            textgrad_log = CONFIG["output_file_textgrad"]

            # Static Evaluation
            self.evaluate_static_prompt(static_log, task_type, prompt)

            # TextGrad Optimization
            try:
                improved_prompt = self.process_example_textgrad(prompt)
                print(f"Optimized prompt for {task_type}:\n{improved_prompt}\n")
                with open(textgrad_log, "a") as f:
                    f.write(f"=== Optimized Prompt for {task_type} ===\n{improved_prompt}\n\n")
            except Exception as e:
                print(f"Failed to process task {task_type}: {e}")


if __name__ == "__main__":
    recommender = PromptTuning()
    
    try:
        print(" LLM Test: Generating output...")
        output = recommender.model.generate("What is the capital of France?")
        print("LLM Output:", output)
    except Exception as e:
        print("LLM generation failed:", e)

    recommender.execute()
    print(f"Training completed. Results saved to {CONFIG['output_file']} and {CONFIG['output_file_textgrad']}")

