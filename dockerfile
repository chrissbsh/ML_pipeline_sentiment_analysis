# 1. Utiliser une image Python légère
FROM python:3.12-slim

# 2. Définir le dossier de travail
WORKDIR /app

# 3. Copier les fichiers de dépendances
COPY requirements.txt .

# 4. Installer les libs (PyTorch CPU pour économiser de la RAM au début)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier le reste du code
COPY . .

# 6. Lancer FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]