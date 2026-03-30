from datasets import load_dataset
import re

# On teste avec un dataset de titres de presse plus large
# Note: 'ag_news' est un classique pour les news mondiales (World, Sports, Business, Sci/Tech)
dataset = load_dataset("ag_news", split="test")

geopolitics_keywords = ["war", "tariffs", "sanctions", "conflict", "election", "missile", "border"]
regex_pattern = re.compile(r'\b(' + '|'.join(geopolitics_keywords) + r')\b', re.IGNORECASE)

# On scanne les news de la catégorie 0 (World / Monde)
count = 0
for example in dataset:
    text = example['text']
    if regex_pattern.search(text):
        print(f"--- ALERTE DÉTECTÉE ---")
        print(f"News: {text}")
        count += 1
        if count >= 5: break

print(f"\nAlertes trouvées : {count}")