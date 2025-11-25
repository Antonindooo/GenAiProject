import streamlit as st
import json
import os
import time
import shutil
import re
from dotenv import load_dotenv
from fpdf import FPDF  # Pour la génération de PDF

# --- CHARGEMENT DES VARIABLES D'ENVIRONNEMENT (CLÉ API) ---
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

# Liste des fichiers PDF
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
    """
    Génère un PDF stylisé en interprétant le Markdown basique (Titres, Listes).
    """

    class PDF(FPDF):
        def header(self):
            # Titre Principal Rouge
            self.set_font('Arial', 'B', 20)
            self.set_text_color(201, 43, 43)  # Rouge #C92B2B
            self.cell(0, 10, "IRONMIND - Plan d'Entrainement", 0, 1, 'C')
            self.ln(5)

            # Ligne de séparation
            self.set_draw_color(201, 43, 43)
            self.set_line_width(0.5)
            self.line(10, 25, 200, 25)
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Nettoyage des caractères non supportés par FPDF standard (latin-1)
    # On remplace les emojis courants par rien ou des symboles simples
    replacements = {
        "🏊": "Natation: ", "🚴": "Velo: ", "🏃": "Course: ", "🏋️": "Renfo: ",
        "✅": "[OK]", "❌": "[NO]", "->": ">", "—": "-"
    }

    lines = text.split('\n')

    for line in lines:
        line = line.strip()

        # Application des remplacements d'emojis
        for key, val in replacements.items():
            line = line.replace(key, val)

        # Encodage sécurisé pour éviter les crashs (latin-1)
        try:
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        except:
            continue  # Saute la ligne si illisible

        if not safe_line:
            pdf.ln(3)  # Petit espace pour les lignes vides
            continue

        # --- DÉTECTION DU STYLE MARKDOWN ---

        # TITRE 1 (ex: # Lundi) -> Rouge + Gros
        if safe_line.startswith('# '):
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 16)
            pdf.set_text_color(201, 43, 43)  # Rouge
            content = safe_line.replace('#', '').strip()
            pdf.cell(0, 10, content, 0, 1)

            # Petite ligne sous le jour
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.set_draw_color(220, 220, 220)
            pdf.line(x, y, x + 190, y)
            pdf.ln(2)

        # TITRE 2 (ex: ## Matin) -> Bleu Foncé
        elif safe_line.startswith('## '):
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 13)
            pdf.set_text_color(44, 62, 80)  # Bleu nuit
            content = safe_line.replace('#', '').strip()
            pdf.cell(0, 8, content, 0, 1)

        # LISTE A PUCES (ex: - 10min échauffement)
        elif safe_line.startswith('- '):
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(50, 50, 50)  # Gris foncé
            content = safe_line[2:]  # Enlève le tiret

            # Astuce pour simuler une puce propre
            current_y = pdf.get_y()
            pdf.set_x(15)  # Indentation
            pdf.cell(5, 5, chr(149), 0, 0)  # Puce ronde (bullet)
            pdf.set_x(20)
            pdf.multi_cell(0, 5, content.replace('**', ''))  # On enlève les ** du gras Markdown

        # TEXTE NORMAL
        else:
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 5, safe_line.replace('**', ''))

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
            print(f"Erreur chargement PDF {file_name}: {e}")

    if not documents:
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
    """Agent Coach (Générateur V1)."""
    coach_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            f"""
            TU ES L'AGENT COACH (GÉNÉRATEUR). Ton rôle est de créer une première ébauche de plan.
            Ne te soucie pas trop des détails physiologiques fins, concentre-toi sur la structure.
            Paramètres : {json.dumps(user_params)}
            """
        ),
        HumanMessage(content="{user_input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm_coach, [retriever_tool], coach_prompt)
    return AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True, handle_parsing_errors=True)


def finalize_plan_with_agent2(draft_plan, retriever_tool, user_params, feedbacks):
    """Agent Physiologiste (Raffineur & Finaliseur)."""

    # 1. Récupération contexte RAG
    try:
        rules = retriever_tool.invoke("Règles progression, sécurité, volume max.")
        rules_snippet = rules[0].page_content[:800] if rules else "Règles standards."
    except:
        rules_snippet = "Règles standards triathlon."

    # 2. Construction des contraintes unifiées (Params + Feedbacks)
    feedback_str = "\n".join([f"- {fb}" for fb in feedbacks]) if feedbacks else "Aucun feedback supplémentaire."

    # Prompt de finalisation
    refine_prompt = f"""
    TU ES L'AGENT 2 : EXPERT PHYSIOLOGISTE ET COACH SENIOR.

    Ton rôle est de générer le PLAN FINAL en perfectionnant l'ébauche fournie.

    SOURCES D'INFORMATION (PAR ORDRE DE PRIORITÉ) :
    1. PARAMÈTRES & FEEDBACKS UTILISATEUR (PRIORITÉ ABSOLUE) :
       - Profil : {json.dumps(user_params)}
       - Feedbacks Récents : 
         {feedback_str}

    2. RÈGLES PHYSIOLOGIQUES (RAG) :
       {rules_snippet}

    3. PLAN ACTUEL / ÉBAUCHE (BASE DE TRAVAIL) :
       {draft_plan}

    INSTRUCTIONS :
    - Si le plan actuel respecte les feedbacks, garde sa structure.
    - Si le plan actuel viole un feedback (ex: "Pas de natation le lundi" mais qu'il y en a), MODIFIE LE PLAN pour respecter le feedback.
    - Améliore la précision physiologique.
    - **FORMATTING IMPORTANT** : Utilise une structure Markdown très claire.
      - Utilise des emojis pour chaque sport (🏊, 🚴, 🏃, 🏋️).
      - Utilise des **Titres** pour les jours (ex: `### Lundi`).
      - Mets les points clés en **Gras**.
      - Fais une liste claire et aérée (tirets).

    Génère directement le plan d'entraînement final (Format Markdown propre et esthétique).
    IMPORTANT : Ne mets PAS de balises ```markdown au début ou à la fin. Donne juste le texte brut.
    """

    # Appel direct au modèle (pas besoin d'agent complexe ici, c'est du raffinement textuel)
    messages = [
        SystemMessage(content="Tu es un expert en planification de triathlon. Tu finalises les plans."),
        HumanMessage(content=refine_prompt)
    ]
    response = llm_critique.invoke(messages)

    # --- CORRECTION 1 : NETTOYAGE DU TEXTE ---
    # On retire les balises de code Markdown si l'IA en a mis
    clean_content = response.content.replace("```markdown", "").replace("```", "").strip()

    return clean_content


def simple_markdown_to_html(text):
    """Convertit un markdown simple en HTML pour l'affichage stylisé."""
    if not text: return ""

    # Conversion des Headers
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)

    # Conversion du Gras
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    # Conversion des Listes (simplifiée pour le style carte)
    lines = text.split('\n')
    new_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- '):
            if not in_list:
                new_lines.append('<ul>')
                in_list = True
            content = stripped[2:]
            new_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                new_lines.append('</ul>')
                in_list = False
            # Conserver les lignes de texte normales
            if not stripped.startswith('<h') and stripped:
                new_lines.append(f'<p>{line}</p>')
            else:
                new_lines.append(line)

    if in_list:
        new_lines.append('</ul>')

    return "\n".join(new_lines)


# --- 3. INTERFACE STREAMLIT ---

st.set_page_config(page_title="IronMind AI", layout="wide", page_icon="🏊‍♂️")

# --- CSS PERSONNALISÉ "GRANDIOSE" ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;800&display=swap');

    /* TYPOGRAPHIE GLOBALE */
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    /* EN-TÊTE PRINCIPAL */
    .main-header {
        background: linear-gradient(90deg, #C92B2B 0%, #8E0E00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .sub-header {
        font-size: 1.8rem;
        color: #2C3E50;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #C92B2B;
        padding-left: 15px;
    }

    /* CARTE DU PLAN D'ENTRAÎNEMENT */
    .plan-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1); /* Ombre profonde */
        border: 1px solid #f0f0f0;
        position: relative;
        overflow: hidden;
        color: #333; /* Couleur par défaut du texte */
    }

    /* Bandeau décoratif sur la carte */
    .plan-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 8px;
        background: linear-gradient(90deg, #C92B2B, #FF4B4B);
    }

    /* STYLE HTML INTERNE DE LA CARTE */
    .plan-card h1 {
        color: #C92B2B;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 20px;
        border-bottom: 2px solid #eee;
        padding-bottom: 10px;
    }

    .plan-card h2 {
        color: #2C3E50;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .plan-card h3 {
        color: #C92B2B;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 20px;
        text-transform: uppercase;
    }

    .plan-card strong {
        color: #2C3E50;
        background-color: #f8f9fa;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 700;
        border: 1px solid #e9ecef;
    }

    .plan-card ul {
        list-style: none;
        padding-left: 0;
    }

    .plan-card li {
        margin-bottom: 12px;
        padding-left: 30px;
        position: relative;
        font-size: 1.05rem;
        color: #444;
        line-height: 1.6;
    }

    .plan-card li::before {
        content: "➤";
        color: #C92B2B;
        position: absolute;
        left: 0;
        font-weight: bold;
    }

    .plan-card p {
        font-size: 1rem;
        color: #666;
        line-height: 1.6;
    }

    /* BOUTONS STYLISÉS */
    .stButton>button {
        background: linear-gradient(45deg, #C92B2B, #FF416C);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        height: 55px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(201, 43, 43, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(201, 43, 43, 0.5);
    }

    /* SIDEBAR */
    .css-1d391kg {
        background-color: #f7f9fc;
    }

    /* ZONE DE TEXTE */
    .stTextArea>div>div>textarea {
        border-radius: 12px;
        border: 2px solid #eee;
        padding: 15px;
        font-family: 'Montserrat', sans-serif;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #C92B2B;
        box-shadow: 0 0 0 2px rgba(201, 43, 43, 0.2);
    }

</style>
""", unsafe_allow_html=True)

# Initialisation du Session State
if "current_plan" not in st.session_state:
    st.session_state.current_plan = None
if "plan_history" not in st.session_state:
    st.session_state.plan_history = []
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

# En-tête Principal
st.markdown('<div class="main-header">IRONMIND AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Le Coach Intelligent pour votre performance Triathlon</div>', unsafe_allow_html=True)

# Chargement RAG
if not st.session_state.rag_ready:
    try:
        retriever = get_retriever()
        st.session_state.retriever = retriever
        st.session_state.rag_ready = True
    except Exception as e:
        st.error(f"Erreur RAG: {e}")
        st.stop()

tool_rag = Tool(
    name="ExpertTriathlon",
    func=lambda q: st.session_state.retriever.invoke(q),
    description="Expertise Ironman."
)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2413/2413074.png", width=80)
    st.header("Profil Athlète")
    st.markdown("---")
    user_level = st.selectbox("🏅 Niveau", ["Débutant", "Intermédiaire", "Avancé"])
    weekly_hours = st.slider("⏱️ Volume (Heures/semaine)", 5, 25, 12)
    goal_race = st.text_input("🎯 Objectif Principal", "Ironman Nice")

    st.markdown("---")
    st.info("💡 **Conseil :** Ajustez ces paramètres avant de générer votre premier plan.")

user_params = {"Niveau": user_level, "Heures": weekly_hours, "Objectif": goal_race}

# --- LOGIQUE PRINCIPALE ---

if st.sidebar.button("🚀 GÉNÉRER MON PLAN", type="primary"):
    # Reset
    st.session_state.plan_history = []
    st.session_state.feedbacks = []
    st.session_state.current_plan = None

    coach = create_coach_agent_executor(tool_rag, user_params)

    # --- AGENT 1 : BROUILLON ---
    with st.spinner("🏃‍♂️ Agent 1 : Structuration du macrocycle..."):
        prompt_initial = "Génère une ébauche de la SEMAINE 1."
        draft = coach.invoke({"user_input": prompt_initial})["output"]
        st.session_state.plan_history.append({"role": "Agent 1 (Brouillon)", "content": draft})

    # --- AGENT 2 : FINALISATION ---
    with st.spinner("🧬 Agent 2 : Optimisation physiologique & Design..."):
        # Reçoit Params initiaux + Brouillon
        final_plan = finalize_plan_with_agent2(draft, tool_rag, user_params, st.session_state.feedbacks)

        st.session_state.current_plan = final_plan
        st.session_state.plan_history.append({"role": "Agent 2 (Final)", "content": final_plan})

# --- AFFICHAGE ET BOUCLE DE FEEDBACK ---

if st.session_state.current_plan:

    # Affichage de l'historique dans un expander discret
    with st.expander("🔬 Voir les logs du raisonnement IA"):
        for step in st.session_state.plan_history:
            st.code(f"Role: {step['role']}", language="text")

    st.markdown('<div class="sub-header">📅 VOTRE SEMAINE TYPE</div>', unsafe_allow_html=True)

    # CONVERSION MARKDOWN -> HTML pour affichage dans la carte
    # Cela permet d'avoir le contenu DANS le div blanc avec le bon style
    html_plan = simple_markdown_to_html(st.session_state.current_plan)

    st.markdown(f"""
    <div class="plan-card">
        {html_plan}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sub-header">💬 COACHING & AJUSTEMENTS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.write("**Une contrainte ? Une blessure ? Dites-le au coach :**")

        # --- CORRECTION 2 : UTILISATION D'UN FORMULAIRE (st.form) ---
        # Cela permet de valider le feedback en UN SEUL CLIC
        with st.form(key="feedback_form"):
            user_feedback = st.text_area(
                "feedback_input",
                height=100,
                label_visibility="collapsed",
                placeholder="Ex: 'Pas de vélo le mercredi', 'Je veux plus de volume en natation'..."
            )
            submit_btn = st.form_submit_button("🔄 METTRE À JOUR LE PLAN")

        if submit_btn:
            if user_feedback:
                # 1. Mise à jour des feedbacks
                st.session_state.feedbacks.append(user_feedback)

                # 2. Récupération de l'ancien plan (celui affiché juste avant)
                previous_plan = st.session_state.current_plan

                # --- AGENT 2 SEUL : MISE À JOUR ---
                with st.spinner("🤖 Agent 2 : Révision intelligente du plan..."):

                    updated_plan = finalize_plan_with_agent2(
                        previous_plan,  # On passe l'ancien plan comme "brouillon" à affiner
                        tool_rag,
                        user_params,
                        st.session_state.feedbacks  # Les feedbacks sont concaténés ici
                    )

                    st.session_state.current_plan = updated_plan
                    st.session_state.plan_history.append(
                        {"role": "Agent 2 (Mise à jour Feedback)", "content": updated_plan})

                st.rerun()
            else:
                st.warning("⚠️ Veuillez entrer un feedback avant de valider.")

    with col2:
        st.write("**Satisfait ?**")
        if st.session_state.current_plan:
            pdf_bytes = create_pdf(st.session_state.current_plan)
            st.download_button(
                label="📥 TÉLÉCHARGER PDF",
                data=bytes(pdf_bytes),
                file_name="Plan_IronMind_Pro.pdf",
                mime="application/pdf",
                type="primary"
            )