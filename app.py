import streamlit as st
import json
import os
import time
import shutil
import re
from dotenv import load_dotenv
from fpdf import FPDF

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


# --- 1. UTILITAIRES (PDF STYLISÉ & RAG) ---

def create_pdf(text, filename="Plan_Entrainement_IronMind.pdf"):
    """
    Génère un PDF stylisé (Couleurs, Mise en forme) à partir du plan Markdown.
    """

    class PDF(FPDF):
        def header(self):
            # Titre Principal Rouge
            self.set_font('Arial', 'B', 20)
            self.set_text_color(201, 43, 43)  # Rouge IronMind
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

    # Dictionnaire pour remplacer les emojis (FPDF ne gère pas les emojis)
    replacements = {
        "🏊": "Natation: ", "🚴": "Velo: ", "🏃": "Course: ", "🏋️": "Renfo: ",
        "✅": "[OK]", "❌": "[NO]", "->": ">", "—": "-"
    }

    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        # Remplacement des emojis
        for key, val in replacements.items():
            line = line.replace(key, val)

        # Encodage sécurisé latin-1
        try:
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        except:
            continue

        if not safe_line:
            pdf.ln(3)
            continue

        # --- DÉTECTION DU STYLE MARKDOWN ---
        # TITRE 1 (Jours)
        if safe_line.startswith('# '):
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 16)
            pdf.set_text_color(201, 43, 43)
            content = safe_line.replace('#', '').strip()
            pdf.cell(0, 10, content, 0, 1)
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.set_draw_color(220, 220, 220)
            pdf.line(x, y, x + 190, y)
            pdf.ln(2)

        # TITRE 2 (Sections matin/soir)
        elif safe_line.startswith('## '):
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 13)
            pdf.set_text_color(44, 62, 80)
            content = safe_line.replace('#', '').strip()
            pdf.cell(0, 8, content, 0, 1)

        # TITRE 4 (Sous-sections comme les jours si format ####)
        elif safe_line.startswith('#### '):
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(201, 43, 43)
            content = safe_line.replace('#', '').strip()
            pdf.cell(0, 8, content, 0, 1)

        # LISTE A PUCES
        elif safe_line.startswith('- '):
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(50, 50, 50)
            content = safe_line[2:]
            pdf.set_x(15)
            pdf.cell(5, 5, chr(149), 0, 0)
            pdf.multi_cell(0, 5, content.replace('**', ''))

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
            Le plan doit faire le même nombre de semaines que dans les parametresx.
            Paramètres : {json.dumps(user_params)}
            """
        ),
        HumanMessage(content="{user_input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm_coach, [retriever_tool], coach_prompt)
    return AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True, handle_parsing_errors=True)


def finalize_plan_with_agent2(draft_plan, retriever_tool, user_params, feedbacks):
    """
    Agent Physiologiste (Raffineur & Finaliseur).
    Retourne (Explication, Plan) nettoyés.
    """

    # 1. Récupération contexte RAG
    try:
        rules = retriever_tool.invoke("Règles progression, sécurité, volume max.")
        rules_snippet = rules[0].page_content[:800] if rules else "Règles standards."
    except:
        rules_snippet = "Règles standards triathlon."

    # 2. Construction des contraintes unifiées
    feedback_str = "\n".join(
        [f"- {fb}" for fb in feedbacks]) if feedbacks else "Aucun feedback pour l'instant (Plan initial)."

    # Prompt de finalisation
    refine_prompt = f"""
    TU ES L'AGENT 2 : EXPERT PHYSIOLOGISTE ET COACH SENIOR.

    Ton rôle est de générer le PLAN FINAL et d'EXPLIQUER tes modifications si il y a des feedbacks.

    SOURCES :
    1. PROFIL : {json.dumps(user_params)}
    2. FEEDBACKS UTILISATEUR : 
       {feedback_str}
    3. RÈGLES RAG : {rules_snippet}
    4. ÉBAUCHE : {draft_plan}

    INSTRUCTIONS DE STRUCTURE (TRES IMPORTANT) :
    Tu dois répondre en DEUX PARTIES séparées par la balise exacte "---PLAN_START---".

    PARTIE 1 : MESSAGE AU COACHÉ
    - Si c'est la première génération : Dis un mot d'encouragement court.
    - Si il y a des feedbacks : Explique brièvement (2-3 phrases) ce que tu as modifié pour respecter la demande.

    ---PLAN_START---

    PARTIE 2 : LE PLAN (MARKDOWN)
    - Utilise une structure Markdown très claire.
    - Emojis : 🏊, 🚴, 🏃, 🏋️.
    - Titres pour les jours : Utilise `#### Lundi`, `#### Mardi` (Niveau 4).
    - Liste à puces claire.
    - PAS de balises ```markdown ou ```. Juste le texte brut.
    """

    messages = [
        SystemMessage(content="Tu es un expert en planification. Tu expliques tes choix puis tu donnes le plan."),
        HumanMessage(content=refine_prompt)
    ]

    response = llm_critique.invoke(messages)
    content = response.content

    # Parsing de la réponse avec nettoyage robuste
    if "---PLAN_START---" in content:
        parts = content.split("---PLAN_START---")
        explanation = parts[0].strip()
        plan_clean = parts[1].strip()
    else:
        # Fallback
        explanation = "Voici votre plan mis à jour."
        plan_clean = content.strip()

    # --- NETTOYAGE SUPPLÉMENTAIRE POUR EVITER LES ERREURS D'AFFICHAGE ---
    # On enlève les titres parasites que l'IA a tendance à répéter
    explanation = explanation.replace("PARTIE 1 : MESSAGE AU COACHÉ", "").replace("PARTIE 1", "").strip()

    plan_clean = plan_clean.replace("PARTIE 2 : LE PLAN (MARKDOWN)", "").replace("PARTIE 2", "")
    plan_clean = plan_clean.replace("```markdown", "").replace("```", "").strip()

    return explanation, plan_clean


def simple_markdown_to_html(text):
    """Convertit un markdown simple en HTML pour l'affichage stylisé."""
    if not text: return ""

    # Conversion des Headers
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    # NOUVEAU : Support des headers #### (souvent utilisés pour les jours)
    text = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)

    # Conversion du Gras
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    # Conversion des Listes
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
            if not stripped.startswith('<h') and stripped:
                new_lines.append(f'<p>{line}</p>')
            else:
                new_lines.append(line)

    if in_list:
        new_lines.append('</ul>')

    return "\n".join(new_lines)


# --- 3. INTERFACE STREAMLIT ---

st.set_page_config(page_title="IronMind AI", layout="wide", page_icon="🏊‍♂️")

# --- CSS PERSONNALISÉ ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }

    .main-header {
        background: linear-gradient(90deg, #C92B2B 0%, #8E0E00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem; text-align: center; font-weight: 800;
        margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 2px;
    }
    .subtitle { text-align: center; color: #555; font-size: 1.2rem; margin-bottom: 2rem; }
    .sub-header {
        font-size: 1.8rem; color: #2C3E50; font-weight: 700;
        margin-top: 2rem; margin-bottom: 1rem;
        border-left: 5px solid #C92B2B; padding-left: 15px;
    }
    .plan-card {
        background: #ffffff; border-radius: 20px; padding: 40px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1); border: 1px solid #f0f0f0;
        position: relative; overflow: hidden; color: #333;
    }
    .plan-card::before {
        content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 8px;
        background: linear-gradient(90deg, #C92B2B, #FF4B4B);
    }
    .plan-card h1 { color: #C92B2B; font-size: 2rem; font-weight: 800; margin-top: 20px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
    .plan-card h2 { color: #2C3E50; font-size: 1.5rem; font-weight: 700; margin-top: 25px; }
    .plan-card h3 { color: #C92B2B; font-size: 1.2rem; font-weight: 600; margin-top: 20px; text-transform: uppercase; }

    /* NOUVEAU STYLE H4 pour les Jours (Lundi, Mardi...) */
    .plan-card h4 { 
        color: #2C3E50; font-size: 1.1rem; font-weight: 800; 
        margin-top: 15px; margin-bottom: 10px; 
        text-transform: uppercase; letter-spacing: 1px;
    }

    .plan-card strong { color: #2C3E50; background-color: #f8f9fa; padding: 2px 6px; border-radius: 4px; font-weight: 700; border: 1px solid #e9ecef; }
    .plan-card ul { list-style: none; padding-left: 0; }
    .plan-card li { margin-bottom: 12px; padding-left: 30px; position: relative; font-size: 1.05rem; color: #444; }
    .plan-card li::before { content: "➤"; color: #C92B2B; position: absolute; left: 0; font-weight: bold; }
    .stButton>button {
        background: linear-gradient(45deg, #C92B2B, #FF416C); color: white;
        border: none; border-radius: 12px; font-weight: 700; font-size: 1.1rem;
        height: 55px; width: 100%; transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(201, 43, 43, 0.3);
    }
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(201, 43, 43, 0.5); }
    .stTextArea>div>div>textarea { border-radius: 12px; border: 2px solid #eee; padding: 15px; }
</style>
""", unsafe_allow_html=True)

# Initialisation du Session State
if "current_plan" not in st.session_state:
    st.session_state.current_plan = None
if "plan_history" not in st.session_state:
    st.session_state.plan_history = []
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []
if "coach_message" not in st.session_state:
    st.session_state.coach_message = ""
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
    st.header("Profil Athlète")
    st.markdown("---")
    user_level = st.selectbox("🏅 Niveau", ["Débutant", "Intermédiaire", "Avancé"])
    weekly_hours = st.slider("⏱️ Volume (Heures/semaine)", 5, 25, 12)
    goal_race = st.text_input("🎯 Objectif Principal", "Ironman Nice")
    weeks_until_race = st.number_input("⏳ Semaines avant la course", min_value=1, max_value=52, value=12)
    st.markdown("---")
    st.info("💡 **Conseil :** Ajustez ces paramètres avant de générer votre premier plan.")

user_params = {
    "Niveau": user_level,
    "Heures": weekly_hours,
    "Objectif": goal_race,
    "Semaines_Restantes": weeks_until_race
}

# --- LOGIQUE PRINCIPALE ---

if st.sidebar.button("🚀 GÉNÉRER MON PLAN", type="primary"):
    # Reset
    st.session_state.plan_history = []
    st.session_state.feedbacks = []
    st.session_state.current_plan = None
    st.session_state.coach_message = ""

    coach = create_coach_agent_executor(tool_rag, user_params)

    # --- AGENT 1 : BROUILLON ---
    with st.spinner("🏃‍♂️ Agent 1 : Structuration du macrocycle..."):
        prompt_initial = "Génère une ébauche de la SEMAINE 1."
        draft = coach.invoke({"user_input": prompt_initial})["output"]
        st.session_state.plan_history.append({"role": "Agent 1 (Brouillon)", "content": draft})

    # --- AGENT 2 : FINALISATION ---
    with st.spinner("🧬 Agent 2 : Optimisation physiologique & Design..."):
        # Double unpacking
        explanation, final_plan = finalize_plan_with_agent2(draft, tool_rag, user_params, st.session_state.feedbacks)

        st.session_state.coach_message = explanation
        st.session_state.current_plan = final_plan
        st.session_state.plan_history.append({"role": "Agent 2 (Final)", "content": final_plan})

# --- AFFICHAGE ET BOUCLE DE FEEDBACK ---

if st.session_state.current_plan:

    with st.expander("🔬 Voir les logs du raisonnement IA"):
        for step in st.session_state.plan_history:
            st.code(f"Role: {step['role']}", language="text")

    st.markdown('<div class="sub-header">📅 VOTRE SEMAINE TYPE</div>', unsafe_allow_html=True)

    # --- AFFICHAGE DU MESSAGE DU COACH ---
    if st.session_state.coach_message:
        st.info(f"🗣️ **Coach IronMind :** {st.session_state.coach_message}")
    # -------------------------------------

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

        with st.form(key="feedback_form", clear_on_submit=True):
            user_feedback = st.text_area(
                "input_feedback",
                height=100,
                label_visibility="collapsed",
                placeholder="Ex: 'Pas de vélo le mercredi', 'Je veux plus de volume en natation'..."
            )
            submit_btn = st.form_submit_button("🔄 METTRE À JOUR LE PLAN")

        if submit_btn:
            if user_feedback:
                st.session_state.feedbacks.append(user_feedback)
                previous_plan = st.session_state.current_plan

                with st.spinner("🤖 Agent 2 : Révision intelligente du plan..."):

                    explanation, updated_plan = finalize_plan_with_agent2(
                        previous_plan,
                        tool_rag,
                        user_params,
                        st.session_state.feedbacks
                    )

                    st.session_state.coach_message = explanation
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