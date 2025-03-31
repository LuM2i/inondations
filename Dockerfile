# Utilisation d'une image Python légère
FROM python:3.10-slim
 
# Définition du répertoire de travail dans le conteneur
WORKDIR /app
 
# Copier les fichiers de l'application dans le conteneur
COPY . /app
 
# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt
 
# Exposition du port de l'API
EXPOSE 8000

# Définir la variable d'environnement PORT (Render l'utilisera)
ENV PORT=8000
 
# Commande pour démarrer l'API FastAPI avec Uvicorn
#CMD ["sh", "-c", "uvicorn industrialisation:app --host 0.0.0.0 --port ${PORT}"]

# Commande pour démarrer l'API FastAPI avec uvicorn via le script Python
CMD ["python", "industrialisation.py"]