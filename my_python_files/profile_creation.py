import random
import subprocess
import json
import requests
import re
import ast
class OllamaHelper:
    """Helper class to interact with Ollama LLaMA models."""
    def __init__(self, model="llama3"):
        self.model = model
        self.name_cache = {}  # Cache generated names

    def call_model(self, prompt):
        url = "http://localhost:12345/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload)
        remove_think_sections = lambda text: '\n'.join(
                                    line for line in re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
                                    .splitlines()
                                    if not line.lower().strip().startswith('<think>')
                                ).strip()
        return remove_think_sections(response.json()["response"])

    def generate_ethnic_names(self, ethnicity, gender, count=10):
        """Generate a list of names based on ethnicity and gender."""
        cache_key = f"{ethnicity}_{gender}"
        if cache_key in self.name_cache:
            return self.name_cache[cache_key]

        gender_male_or_female = "male" if gender == "Mr" else "female"
        prompt = (
            f"Generate {count} {gender_male_or_female} surnames commonly used by people of {ethnicity} descent. "
            "Return only a JSON list of names like [\"Name1\", \"Name2\", ...]. Do not include any explanation or prefix."
        )

        response = self.call_model(prompt)

        # Try parsing as JSON directly
        try:
            names = json.loads(response)
            if isinstance(names, list) and all(isinstance(name, str) for name in names):
                self.name_cache[cache_key] = names
                return names
        except json.JSONDecodeError:
            pass

        # Fallback: extract list-looking text using regex
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            try:
                names = ast.literal_eval(match.group(0))
                if isinstance(names, list) and all(isinstance(name, str) for name in names):
                    self.name_cache[cache_key] = names
                    return names
            except:
                pass

        # If all fails, report and return empty list
        print(f"Failed to parse names for {ethnicity}, {gender}. Raw output:\n{response}")
        return []

    


class RandomPerson:
    genders = ["Mr", "Ms"]
    pickiness_levels = ["extremely picky", "moderately picky", "not picky"]
    ethnicities = ["White", "Black or African descent", "Latino or Hispanic", "East Asian (e.g., Chinese, Japanese, Korean)", "South Asian (e.g., Indian, Pakistani, Bangladeshi)", "Southeast Asian (e.g., Filipino, Vietnamese, Thai)", "Middle Eastern or North African (MENA)", "Indigenous or Native (e.g., Native American, Aboriginal)", "Pacific Islander", "Mixed or Multiracial"]

    def __init__(self, ollama_helper):
        self.ollama = ollama_helper
        self.ethnicity = random.choice(self.ethnicities)
        self.gender_title = random.choice(self.genders)
        self.gender = "Male" if self.gender_title == "Mr" else "Female"
        self.name = self.get_name()
        self.age = random.randint(5, 80)
        self.pickiness = random.choice(self.pickiness_levels)

    def get_name(self):
        names = self.ollama.generate_ethnic_names(self.ethnicity, self.gender, count=10)
        return random.choice(names) if names else "Nair"

    def __str__(self):
        return f"{self.gender_title} {self.name} ({self.ethnicity}), Age: {self.age}, Pickiness: {self.pickiness}"


# --- Usage Example ---
if __name__ == "__main__":
    ollama = OllamaHelper(model="qwen2:7b")
    person = RandomPerson(ollama)
    print(person)

    
