import streamlit as st
import json
import os
import time
import shutil
from dotenv import load_dotenv
from fpdf import FPDF  # Pour la génération de PDF

# --- CHARGEMENT DES VARIABLES D'ENVIRONNEMENT (CLÉ API) ---
# C'est ici que le changement est nécessaire : charger l'environnement AVANT tout le reste
load_dotenv()

# --- IMPORTS LANGCHAIN / AGENTS ---
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONSTANTES ET CONFIGURATION ---
CHROMA_DB_PATH = "chroma_data"
KNOWLEDGE_BASE_DIR = "knowledge_base"

# Liste des fichiers PDF pour la reconstruction
PDF_FILES = [
    "Ironman.pdf",
    "50-conseills-pour-reussir-vos-debuts-en-triathlon.pdf",
    "Les Fondamentaux d.pdf",
    "WTS_Blog_Nutrition_French.pdf",
    "3x3EssentialSwimBikeAndRunSessions.pdf",
    "Triathlon Sprint Avancé 24 semaines.pdf",
]


# --- 1. UTILITAIRES (PDF & RAG) ---

def create_pdf(text, filename="Plan_Entrainement_IronMind.pdf"):
    """Génère un PDF basique à partir du texte du plan."""

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'IronMind - Plan d\'Entrainement', 0, 1, 'C')
            self.ln(10)

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Nettoyage basique pour FPDF (qui ne gère pas bien tous les caractères unicode par défaut)
    # On remplace les caractères Markdown gras/titres pour la lisibilité
    clean_text = text.replace('**', '').replace('#', '').replace('__', '')

    # Gestion de l'encodage pour éviter les erreurs latin-1
    # On écrit ligne par ligne
    for line in clean_text.split('\n'):
        # Encodage/décodage pour gérer les accents au mieux avec la police de base
        encoded_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, encoded_line)

    return pdf.output(dest='S').encode('latin-1')


def create_vector_store():
    """Charge et vectorise les documents."""
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    documents = []
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)

    for file_name in PDF_FILES:
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, file_name)
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Erreur chargement PDF {file_name}: {e}")

    if not documents:
        # Création d'un document vide factice pour ne pas planter si pas de PDF
        # (Juste pour éviter le crash, mais le RAG sera vide)
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_PATH)
    return vector_store


def get_retriever():
    """Récupère l'outil de recherche RAG."""
    embeddings = OpenAIEmbeddings()
    if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
        with st.spinner("Reconstruction de la base de connaissances..."):
            create_vector_store()
            st.success("Base RAG reconstruite.")

    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 5})


# --- 2. AGENTS ---

llm_coach = ChatOpenAI(model="gpt-4o", temperature=0.7)
llm_critique = ChatOpenAI(model="gpt-4o", temperature=0.5)


def create_coach_agent_executor(retriever_tool, user_params):
    """Agent Coach avec instructions impératives."""
    coach_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            f"""
            TU ES L'AGENT COACH IRONMAN. Ton rôle est de générer et d'adapter des plans d'entraînement.
            RÈGLE 1 : NE JAMAIS POSER DE QUESTIONS. AGIS.
            RÈGLE 2 : Utilise les paramètres fournis ou les retours utilisateurs pour construire le plan.
            RÈGLE 3 : Structure ta réponse en Markdown clair.

            Paramètres initiaux : {json.dumps(user_params)}
            """
        ),
        HumanMessage(content="{user_input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm_coach, [retriever_tool], coach_prompt)
    return AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True, handle_parsing_errors=True)


def critique_plan(plan_brouillon, retriever_tool):
    """Agent Physiologiste (Self-Correction)."""
    max_retries = 3
    rules = []
    for attempt in range(max_retries):
        try:
            rules = retriever_tool.invoke("Règles progression, sécurité, volume max.")
            if rules: break
        except:
            time.sleep(1)

    rules_snippet = rules[0].page_content[:600] if rules else "Règles standards (10%, repos)."

    critique_prompt = f"""
    Critique ce plan Ironman en tant que physiologiste.
    Règles RAG : {rules_snippet}...
    Plan : {plan_brouillon}

    Réponds UNIQUEMENT en JSON :
    {{
        "CRITIQUE_PRINCIPALE": "Violation majeure...",
        "JUSTIFICATION_RAG": "Selon la règle...",
        "CORRECTION_PROPOSEE": "Action concrète..."
    }}
    """
    response = llm_critique.invoke(critique_prompt, response_format={"type": "json_object"})
    return response.content


# --- 3. INTERFACE STREAMLIT ---

# load_dotenv() <- Supprimé d'ici car déplacé tout en haut
st.set_page_config(page_title="IronMind AI", layout="wide")

# Initialisation du Session State pour garder la mémoire
if "current_plan" not in st.session_state:
    st.session_state.current_plan = None
if "plan_history" not in st.session_state:
    st.session_state.plan_history = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

st.title("🧠 IronMind : Coach Triathlon Interactif")
st.markdown("---")

# Chargement RAG unique
if not st.session_state.rag_ready:
    try:
        retriever = get_retriever()
        st.session_state.retriever = retriever
        st.session_state.rag_ready = True
    except Exception as e:
        st.error(f"Erreur RAG: {e}")
        st.stop()

# Outil RAG
tool_rag = Tool(
    name="ExpertTriathlon",
    func=lambda q: st.session_state.retriever.invoke(q),
    description="Expertise Ironman."
)

# Sidebar
st.sidebar.header("Paramètres Athlète")
user_level = st.sidebar.selectbox("Niveau", ["Débutant", "Intermédiaire", "Avancé"])
weekly_hours = st.sidebar.slider("Heures/semaine", 5, 25, 12)
goal_race = st.sidebar.text_input("Objectif", "Ironman Nice")

user_params = {"Niveau": user_level, "Heures": weekly_hours, "Objectif": goal_race}

# --- LOGIQUE PRINCIPALE ---

# Bouton de génération initiale
if st.sidebar.button("🚀 Générer le Plan Initial"):
    st.session_state.plan_history = []  # Reset
    st.session_state.current_plan = None

    coach = create_coach_agent_executor(tool_rag, user_params)

    # Phase 1 : Brouillon
    with st.spinner("Phase 1 : Génération initiale..."):
        draft = coach.invoke({"user_input": "Génère la SEMAINE 1 détaillée."})["output"]
        st.session_state.plan_history.append({"role": "Coach (V1)", "content": draft})

    # Phase 2 : Critique
    with st.spinner("Phase 2 : Analyse Physiologique..."):
        critique_str = critique_plan(draft, tool_rag)
        # Nettoyage JSON
        if critique_str.strip().startswith('```'):
            critique_str = critique_str.split('```json')[1].split('```')[0].strip()
        critique = json.loads(critique_str)
        st.session_state.plan_history.append({"role": "Physio", "content": critique})

    # Phase 3 : Correction
    with st.spinner("Phase 3 : Application des corrections..."):
        correction = critique.get('CORRECTION_PROPOSEE', 'Aucune')
        final_input = f"RÉVISION IMMÉDIATE. Applique STRICTEMENT cette correction : '{correction}'. Génère le plan final."
        final_plan = coach.invoke({"user_input": final_input})["output"]

        st.session_state.current_plan = final_plan
        st.session_state.plan_history.append({"role": "Coach (Final)", "content": final_plan})

# --- AFFICHAGE DU RÉSULTAT & BOUCLE DE FEEDBACK ---

if st.session_state.current_plan:

    # Affichage de l'historique (Optionnel, pour montrer le raisonnement)
    with st.expander("Voir le processus de raisonnement (V1 & Critique)"):
        for step in st.session_state.plan_history[:-1]:
            if step["role"] == "Physio":
                st.error(f"🚨 Critique : {step['content']['CRITIQUE_PRINCIPALE']}")
                st.success(f"💡 Correction : {step['content']['CORRECTION_PROPOSEE']}")
            else:
                st.text(f"--- {step['role']} ---")
                # st.markdown(step["content"]) # Peut être long

    st.subheader("📅 Votre Plan d'Entraînement Actuel")
    st.markdown(st.session_state.current_plan)

    st.markdown("---")
    st.header("💬 Ajustements & Validation")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Demander des modifications")
        user_feedback = st.text_area(
            "Ce plan vous convient-il ? Sinon, demandez des changements (ex: 'Je ne peux pas nager le mardi', 'Moins de vélo').")

        if st.button("🔄 Mettre à jour le plan"):
            if user_feedback:
                coach = create_coach_agent_executor(tool_rag, user_params)
                with st.spinner("L'Agent Coach réadapte le plan selon vos retours..."):
                    update_prompt = f"""
                    ACTION : MISE À JOUR DU PLAN - MODIFICATION IMPÉRATIVE.

                    PLAN À MODIFIER :
                    {st.session_state.current_plan}

                    CONSIGNE DE L'UTILISATEUR (A RESPECTER ABSOLUMENT) : 
                    "{user_feedback}"

                    TÂCHE :
                    Réécris entièrement le plan pour qu'il intègre cette nouvelle contrainte.
                    IMPORTANT : Si l'utilisateur demande de changer un jour (ex: "pas de natation mardi"), tu DOIS déplacer cette séance ou réorganiser la semaine. Le plan final NE DOIT PAS contenir l'élément que l'utilisateur a rejeté.

                    Format : Markdown complet.
                    """
                    new_plan = coach.invoke({"user_input": update_prompt})["output"]
                    st.session_state.current_plan = new_plan
                    st.session_state.plan_history.append({"role": "Mise à jour Utilisateur", "content": new_plan})
                    st.rerun()  # Recharge la page pour afficher le nouveau plan
            else:
                st.warning("Veuillez entrer un feedback pour modifier le plan.")

    with col2:
        st.subheader("Valider")
        st.write("Si le plan est parfait, téléchargez-le en PDF.")

        # Génération du PDF binaire
        pdf_bytes = create_pdf(st.session_state.current_plan)

        st.download_button(
            label="✅ Valider et Télécharger (PDF)",
            data=bytes(pdf_bytes),
            file_name="Mon_Plan_IronMind.pdf",
            mime="application/pdf"
        )