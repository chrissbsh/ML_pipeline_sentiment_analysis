from datasets import load_dataset

# 1. On charge un dataset de référence en finance (Financial PhraseBank)
# Il contient des phrases annotées par des experts financiers.
dataset = load_dataset("lmassaron/FinancialPhraseBank", "default")

# 2. Regardons à quoi ça ressemble
print(f"Nombre de phrases : {len(dataset['train'])}")
print(f"Exemple : {dataset['train'][0]}")

import re

# Ta nouvelle liste de mots-clés
geopolitics_keywords = ["war", "tariffs", "sanctions", "conflict", "trade war"]

# On compile une "expression régulière" qui cherche les mots exacts
# r'\b' signifie "bord du mot"
regex_pattern = re.compile(r'\b(' + '|'.join(geopolitics_keywords) + r')\b', re.IGNORECASE)

for i in range(len(dataset['train'])):
    text = dataset['train'][i]['sentence']
    
    # Recherche exacte du mot
    if regex_pattern.search(text):
        sentiment = dataset['train'][i]['label']
        print(f"--- ALERTE GÉOPOLITIQUE ---")
        print(f"Texte: {text}")
        print(f"Sentiment: {sentiment}\n")