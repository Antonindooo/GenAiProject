import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. Paramètres et Environnement ---

# Charge les variables d'environnement (comme OPENAI_API_KEY)
load_dotenv()

# Vérifie si la clé OpenAI est disponible
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("La variable d'environnement OPENAI_API_KEY n'est pas définie.")
else:
    print("todo va bene")

# Nom du dossier où seront stockés les vecteurs
CHROMA_DB_PATH = "chroma_data"
# Dossier contenant tes fichiers PDF
KNOWLEDGE_BASE_DIR = "knowledge_base"

# Liste des noms de tes fichiers (assure-toi que ces fichiers sont bien dans le dossier knowledge_base/)
PDF_FILES = [
    "Ironman.pdf",
    "50-conseills-pour-reussir-vos-debuts-en-triathlon.pdf",
    "Les Fondamentaux d.pdf",
    "WTS_Blog_Nutrition_French.pdf",
    "3x3EssentialSwimBikeAndRunSessions.pdf",
    "Triathlon Sprint Avancé 24 semaines.pdf",
]


# --- 2. Fonctions de Construction de la Base RAG ---

def create_vector_store():
    """
    Charge, découpe, vectorise les documents et construit la base de données ChromaDB.
    """
    print("--- Démarrage de la construction de la base de connaissances RAG ---")

    documents = []

    # a. Chargement des documents
    print(f"Chargement des {len(PDF_FILES)} fichiers PDF...")
    for file_name in PDF_FILES:
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, file_name)
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"  -> {file_name} chargé avec succès.")
        except Exception as e:
            print(f"  -> ERREUR lors du chargement de {file_name}: {e}")

    if not documents:
        print(
            "Aucun document n'a pu être chargé. Assurez-vous que les fichiers PDF sont dans le dossier 'knowledge_base/'.")
        return None

    # b. Découpage du texte (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Taille maximale des morceaux de texte
        chunk_overlap=200,  # Chevauchement entre les morceaux pour préserver le contexte
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    print(f"Documents découpés en {len(texts)} morceaux (chunks).")

    # c. Vectorisation (Embeddings)
    print("Création des Embeddings (cela peut prendre quelques instants)...")
    # On utilise ici le modèle d'embeddings d'OpenAI (nécessite la clé API)
    embeddings = OpenAIEmbeddings()

    # d. Stockage dans ChromaDB
    print(f"Stockage des embeddings dans la base vectorielle locale ({CHROMA_DB_PATH})...")
    # Créer ou charger la base de données
    vector_store = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    # Sauvegarde sur le disque
    vector_store.persist()
    print("--- Base RAG créée et sauvegardée avec succès ! ---")
    return vector_store


# --- 3. Initialisation de l'outil Retriever ---

def get_retriever():
    """
    Charge la base de données vectorielle existante et crée l'objet Retriever.
    C'est la fonction que l'Agent appellera pour "rechercher" l'information.
    """
    # 1. Définir le modèle d'embeddings (DOIT être le même que celui utilisé pour la création)
    embeddings = OpenAIEmbeddings()

    # 2. Charger l'index depuis le disque
    try:
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        # 3. Créer le retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Récupère les 5 morceaux les plus pertinents
        print(f"Base RAG chargée depuis '{CHROMA_DB_PATH}'.")
        return retriever

    except Exception as e:
        print(f"Erreur lors du chargement de la base ChromaDB: {e}")
        print("Veuillez d'abord exécuter 'create_vector_store()' pour construire l'index.")
        return None


# --- Bloc d'exécution principal ---

if __name__ == "__main__":
    # Assure-toi que tes fichiers PDF sont dans le sous-dossier 'knowledge_base/'
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        print(f"Dossier '{KNOWLEDGE_BASE_DIR}' créé. Veuillez y placer vos fichiers PDF.")

    # Optionnel: supprimer la base existante pour la reconstruire
    # import shutil
    # if os.path.exists(CHROMA_DB_PATH):
    #     shutil.rmtree(CHROMA_DB_PATH)
    #     print(f"Ancienne base '{CHROMA_DB_PATH}' supprimée.")

    create_vector_store()

    # Test rapide de la récupération après la création
    retriever_test = get_retriever()
    if retriever_test:
        test_query = "Quelle est la règle d'augmentation de volume?"
        retrieved_docs = retriever_test.invoke(test_query)
        print(f"\nTest de récupération pour : '{test_query}'")
        print(f"Résultat pertinent trouvé (Source) : {retrieved_docs[0].metadata['source']}")
        print(f"Contenu (Début) : {retrieved_docs[0].page_content[:150]}...")