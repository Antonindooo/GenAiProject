import streamlit as st
import json
import os
from dotenv import load_dotenv
import time
# --- IMPORTS LANGCHAIN / AGENTS ---
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
# Imports des PROMPTS (Templates et Placeholders)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Imports des MESSAGES (System, Human, etc.)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# --- Connexion à ton RAG (Assure-toi que rag_builder.py est accessible) ---
# NOTE: Si get_retriever() est dans un autre fichier, importe-le :
# from rag_builder import get_retriever

# --- Fonctions de base (à placer idéalement dans un module à part ou au début du fichier) ---

# RAPPEL: Place ici les fonctions get_retriever() / create_coach_agent_executor / critique_plan
# (Pour le reste du code, je suppose qu'elles sont définies ou importées.)

# --- Initialisation ---
load_dotenv()

# --- CONSTANTES ET CONFIGURATION ---
CHROMA_DB_PATH = "chroma_data"


# --- 1. FONCTIONS DE BASE DU RAG ---

def get_retriever():
    """
    Charge la base de données vectorielle existante et crée l'objet Retriever.
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
        return retriever

    except Exception as e:
        # Ceci est géré par le bloc try/except principal de Streamlit
        raise Exception(f"Erreur lors du chargement de la base ChromaDB: {e}")


# --- 2. LOGIQUE DES AGENTS ---

# Modèles LLM
llm_coach = ChatOpenAI(model="gpt-4o", temperature=0.7)
llm_critique = ChatOpenAI(model="gpt-4o", temperature=0.5)


def create_coach_agent_executor(retriever_tool, user_params):
    """ Agent 1 : Planificateur, utilise CoT et le RAG, avec des règles impératives."""

    # Règle IMPÉRATIVE N°1: Ne JAMAIS demander d'information.
    # Règle N°2: Utiliser les inputs Streamlit directement.
    coach_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            f"""
            TU ES L'AGENT COACH IRONMAN. Ton rôle EXCLUSIF est de générer des plans d'entraînement.
            RÈGLE IMPÉRATIVE N°1 : TU NE DOIS JAMAIS POSER DE QUESTION À L'UTILISATEUR, NI LUI DEMANDER DE PRÉCISER SES ENTRÉES.
            RÈGLE N°2 : Utilise les paramètres que je te donne pour immédiatement générer le plan.
            Utilise l'outil PlanificationTriathlonExpert pour toutes les décisions de volumes, progression (règle des 10%) et séances Brick.
            Ton processus de raisonnement doit être : ANALYSE (Utilise l'outil) -> PLANIFICATION (Génère le plan structuré en Markdown).

            Paramètres actuels : {user_params}
            """
        ),
        # On retire MessagesPlaceholder(variable_name="chat_history") pour simplifier
        HumanMessage(content="{user_input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm_coach, [retriever_tool], coach_prompt)
    # Suppression de "chat_history" du prompt et du agent executor
    return AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True, handle_parsing_errors=True)


def critique_plan(plan_brouillon, retriever_tool):
    """ Agent 2 : Physiologiste, implémente la Self-Correction avec RAG et un mécanisme de Retry."""

    max_retries = 3
    rules = []

    # --- MÉCANISME DE RETRY POUR L'APPEL RAG ---
    for attempt in range(max_retries):
        try:
            # RAG pour ancrer la critique
            rules = retriever_tool.invoke(
                "Règles de progression, enchaînement vélo-course, et charge maximale par semaine.")
            if rules:
                # Succès : sortir de la boucle de retry
                break
        except Exception as e:
            # Log de l'échec
            print(f"Échec de l'appel RAG (tentative {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # Temporisation exponentielle: 2s, 4s, ...
                sleep_time = 2 ** (attempt + 1)
                time.sleep(sleep_time)
            else:
                # Dernier échec
                print("Le RAG a échoué après toutes les tentatives.")

    # --- Gestion des résultats RAG (FallBack) ---
    if rules:
        # Si des documents sont trouvés, utiliser l'extrait du premier document
        rules_snippet = rules[0].page_content[:600] + '...'
    else:
        # Fallback si le RAG ne trouve rien après tous les retries
        rules_snippet = "ERREUR RAG CRITIQUE: Échec de la récupération après 3 tentatives. Baser la critique sur les règles universelles (règle des 10%, nécessité de repos)."

    critique_prompt = f"""
    En tant qu'Agent Physiologiste expert en prévention des blessures, critique le plan d'entraînement Ironman suivant.

    Règles de sécurité RAG :
    ---
    {rules_snippet}
    ---

    Plan d'entraînement proposé :
    ---
    {plan_brouillon}
    ---

    Analyse le plan et réponds uniquement en format JSON. Ton raisonnement doit être :
    1. CRITIQUE_PRINCIPALE : Identifie la violation de règle la plus grave (ex: Surcharge > 10%, manque de repos).
    2. JUSTIFICATION_RAG : Cite la règle spécifique des documents RAG qui est violée (ex: "Règle des 10%").
    3. CORRECTION_PROPOSEE : Propose une modification concrète pour rendre le plan plus sûr.

    Exemple de sortie JSON :
    {{
        "CRITIQUE_PRINCIPALE": "Le volume total est trop élevé par rapport à la progression recommandée.",
        "JUSTIFICATION_RAG": "La règle des 10% stipule qu'il ne faut pas augmenter de plus de 10% le volume hebdomadaire.",
        "CORRECTION_PROPOSEE": "Réduire le kilométrage de vélo de 20 km et ajouter 1 jour de repos actif."
    }}
    """

    response = llm_critique.invoke(critique_prompt, response_format={"type": "json_object"})
    return response.content

# --- 3. LOGIQUE STREAMLIT ---

# Charge les variables d'environnement
load_dotenv()

st.set_page_config(page_title="IronMind AI", layout="wide")
st.title("🧠 IronMind : L'Agent Coach Triathlon Autonome (RAG & Self-Correction)")
st.markdown("---")

# --- Connexion RAG (Gérée par le try/except pour le feedback utilisateur) ---
try:
    retriever = get_retriever()
    tool_rag = Tool(
        name="PlanificationTriathlonExpert",
        func=lambda query: retriever.invoke(query),
        description="Utilise cet outil pour chercher des informations sur la structure d'entraînement Ironman, les règles de progression (règle des 10%), les zones d'intensité, les Brick Sessions et les protocoles de récupération.",
    )
    st.sidebar.success("Base de connaissances RAG chargée.")

except Exception as e:
    st.sidebar.error("Erreur critique : La base RAG n'a pas pu être chargée. Lancez 'rag_builder.py'.")
    st.stop()
# ------------------------------------------------------------------------


# --- Configuration du Plan (Sidebar) ---
st.sidebar.header("Configuration du Plan")
user_level = st.sidebar.selectbox("Niveau actuel :",
                                  ["Débutant (moins de 2 ans)", "Intermédiaire (2-4 ans)", "Avancé (4+ ans)"],
                                  key="level_select")
weekly_hours = st.sidebar.slider("Heures d'entraînement disponibles (semaine 1) :", min_value=5, max_value=20, value=10,
                                 key="hours_slider")
goal_race = st.sidebar.text_input("Objectif de course :", value="Ironman Nice (dans 8 mois)", key="goal_input")

user_params_dict = {
    "Niveau": user_level,
    "Heures/semaine": weekly_hours,
    "Objectif": goal_race
}

# La commande impérative que l'Agent va recevoir
user_command = "Génère la PREMIÈRE SEMAINE détaillée du plan d'entraînement Ironman."

if st.sidebar.button("⚙️ Lancer la Planification Agentielle"):

    # Initialisation de l'Agent Coach avec les paramètres de la sidebar (Règle Impérative)
    coach_executor = create_coach_agent_executor(tool_rag, user_params_dict)

    # --- ÉTAPE A : Exécution du Planificateur (CoT) ---
    st.header("1. Planification Initiale (Agent Coach)")
    st.markdown("L'Agent Coach utilise l'outil RAG pour structurer le plan (Chain of Thought).")

    try:
        with st.spinner("L'Agent Coach élabore le plan (Analyse RAG)..."):
            # L'input est seulement la commande, les paramètres sont dans le SystemMessage
            plan_draft_result = coach_executor.invoke({"user_input": user_command})
            plan_draft = plan_draft_result["output"]

        st.info("Brouillon du plan généré :")
        st.markdown(plan_draft)
        st.markdown("---")

        # --- ÉTAPE B : Exécution du Critique (Self-Correction/Réflexion) ---
        st.header("2. Critique & Raisonnement (Agent Physiologiste)")
        st.markdown("L'Agent Physiologiste vérifie la sécurité du plan en consultant la base RAG (Self-Correction).")

        with st.spinner("Analyse du risque et génération de la critique ancrée..."):
            critique_json_str = critique_plan(plan_draft, tool_rag)

            # Nettoyage du JSON (si le LLM l'a enveloppé dans ```json...```)
            if critique_json_str.strip().startswith('```'):
                critique_json_str = critique_json_str.split('```json')[1].split('```')[0].strip()

            critique = json.loads(critique_json_str)

            st.error("🚨 Le plan présente un risque potentiel :")
            st.info(f"**Critique Principale :** {critique['CRITIQUE_PRINCIPALE']}")
            st.warning(f"**Justification (Ancrage RAG) :** {critique['JUSTIFICATION_RAG']}")
            st.success(f"**Correction Proposée :** {critique['CORRECTION_PROPOSEE']}")

            st.markdown("---")

            # --- ÉTAPE C : Révision Finale du Plan ---
            st.header("3. Plan Final Corrigé (Assurance Qualité)")

            # Input pour la révision (on inclut la correction et les paramètres pour la nouvelle génération)
            correction_text = critique.get('CORRECTION_PROPOSEE', 'Aucune correction spécifique.')

            final_input = f"""
            ACTION REQUISE : RÉVISION IMMÉDIATE.
            Génère un NOUVEAU plan d'entraînement complet pour la première semaine en appliquant STRICTEMENT cette directive de sécurité : "{correction_text}".
            Les paramètres utilisateur sont : {json.dumps(user_params_dict)}.
            Affiche le plan final corrigé, structuré en Markdown, sans poser de questions.
            """

            with st.spinner("L'Agent Coach intègre la correction et finalise la version 2.0..."):
                final_plan_result = coach_executor.invoke({"user_input": final_input})

            st.success("✅ Plan Final Sûr, Personnalisé et Validé !")
            st.markdown(final_plan_result["output"])

    except json.JSONDecodeError as e:
        st.error("Une erreur s'est produite lors de la critique (Erreur JSON). Le LLM n'a pas pu respecter le format.")
        st.code(critique_json_str, language='json')
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors de l'exécution des agents : {e}")