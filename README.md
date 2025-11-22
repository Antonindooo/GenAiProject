# L'Agent Coach de Triathlon Autonome

Projet IA Générative : Agents Intelligents, Raisonnement Avancé & Streamlit


## 1. Objectif du Projet : Résoudre un Problème Complexe

L'objectif du projet IronMind est de concevoir un système intelligent capable de générer des plans d'entraînement Ironman (distance ultra-longue : 3.8km natation, 180km vélo, 42.2km course) hautement personnalisés, fiables et sécurisés.

La préparation Ironman est un problème complexe car elle nécessite l'équilibre de trois disciplines et impose un risque élevé de surentraînement ou de blessure. L'application garantit la sécurité du plan grâce à une boucle de Raisonnement Avancé et une connaissance experte ancrée.

Interface : L'application est développée avec Streamlit pour une interaction utilisateur fluide.

## 2. Architecture des Agents et Raisonnement Avancé

Notre application implémente une architecture Multi-Agents avec une technique de Self-Correction (Réflexion), garantissant que chaque plan généré est immédiatement soumis à une vérification de sécurité avant d'être présenté à l'utilisateur.

A. Le Raisonnement en Cascade (Self-Correction)

Le processus se déroule en trois étapes distinctes et visibles dans l'interface :

Phase 1 : Planification (Agent Coach - CoT)

Rôle : Générateur de contenu.

Raisonnement : Utilise le Chain of Thought (CoT). L'agent est contraint de penser selon le processus RAG → ANALYSE → PLANIFICATION pour structurer le plan (Périodisation, Volumes).

Phase 2 : Critique (Agent Physiologiste - Réflexion/Ancrage RAG)

Rôle : Vérificateur de sécurité.

Raisonnement : L'Agent Physiologiste lit le plan initial et effectue une Réflexion critique. Il doit obligatoirement utiliser le RAG pour récupérer des règles d'expertise (ex: la "règle des 10%" de progression, les protocoles de récupération) pour justifier ses critiques. Il fournit ensuite une correction structurée (format JSON).

Phase 3 : Révision Finale

Rôle : Garantie de la qualité.

Raisonnement : L'Agent Coach reçoit la critique et génère une nouvelle version du plan en appliquant la directive de sécurité, brisant ainsi le cycle risque-erreur.

B. Le RAG (Retrieval-Augmented Generation) pour la Fiabilité

Le RAG est la fondation de notre agent Physiologiste. Il empêche l'agent d'halluciner sur des sujets techniques et le force à baser ses critiques sur des sources vérifiées.

Corpus de Connaissances : L'application est enrichie par l'indexation de [Nombre total de PDF] documents PDF spécialisés (guides d'entraînement Ironman, principes de périodisation, zones d'intensité, règles de sécurité de progression).

Implémentation : Nous utilisons LangChain et ChromaDB pour stocker les vecteurs des documents, permettant une récupération sémantique rapide des règles physiologiques.

## 3. Installation et Lancement

Ce projet nécessite Python 3.12 (recommandé pour la stabilité des dépendances) et une clé API OpenAI.

A. Prérequis

Clonez ce dépôt

Créez et activez un environnement virtuel :

python3.12 -m venv venv
source venv/bin/activate


B. Configuration des Dépendances et de l'API

Installez les packages Python :

pip install -r requirements.txt


Créez un fichier .env à la racine du projet pour stocker votre clé API (ce fichier est ignoré par Git) :

OPENAI_API_KEY="votre_clé_api_openai_ici"


C. Construction de la Base de Données RAG

Vous devez d'abord construire l'index vectoriel. Assurez-vous que les fichiers PDF de la knowledge_base/ sont présents.
Vous devez également créer un dossier chroma_data afin d'enregistrer les documents vectorisés.
python rag_builder.py


D. Lancement de l'Application Streamlit

streamlit run app.py


## 4. Livrables et Fichiers Clés

app.py: L'application Streamlit principale, contenant la logique des Agents et la boucle de Self-Correction.

rag_builder.py: Script utilisé pour charger les PDF, les chunker et construire l'index ChromaDB.

knowledge_base/: Dossier contenant le corpus de documents experts

requirements.txt: Liste des dépendances.

.gitignore: Assure que .env et chroma_data/ ne sont jamais exposés.
