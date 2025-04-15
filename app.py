import streamlit as st
import speech_recognition as sr
import fitz  # PyMuPDF
import io
import json
import time
from groq import Groq
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
from matplotlib import font_manager
from functools import lru_cache
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Lexlingua",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Groq Client
api_key = "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"  # Replace with your actual Groq API key
client = Groq(api_key=api_key)

# Language mapping
LANGUAGE_CODES = {
    "English": "en-IN",
    "Tamil": "ta-IN",
    "Hindi": "hi-IN",
    "Telugu": "te-IN",
    "Malayalam": "ml-IN",
    "Kannada": "kn-IN"
}

# Initialize components
recognizer = sr.Recognizer()

# Initialize session state variables
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'document_language' not in st.session_state:
    st.session_state.document_language = "English"
if 'legal_risks' not in st.session_state:
    st.session_state.legal_risks = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'voice_input' not in st.session_state:
    st.session_state.voice_input = ""
if 'selected_speech_lang' not in st.session_state:
    st.session_state.selected_speech_lang = "en-IN"
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'is_legal_document' not in st.session_state:
    st.session_state.is_legal_document = True

# Function to set font for matplotlib
def set_matplotlib_font(language_name):
    font_mapping = {
        "Tamil": ["Latha", "Nirmala UI", "DejaVu Sans"],
        "Hindi": ["Mangal", "Nirmala UI", "DejaVu Sans"],
        "Telugu": ["Gautami", "Nirmala UI", "DejaVu Sans"],
        "Malayalam": ["Kartika", "Nirmala UI", "DejaVu Sans"],
        "Kannada": ["Tunga", "Nirmala UI", "DejaVu Sans"],
        "English": ["DejaVu Sans", "Arial", "FreeSans"]
    }
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    selected_fonts = font_mapping.get(language_name, ["DejaVu Sans", "Arial", "FreeSans"])
    
    for font in selected_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font] + ["Arial", "FreeSans"]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ["sans-serif"]
    return "sans-serif"

# Functions
def record_audio(language_code):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            st.info(f"🎙️ Listening for {language_code}... Speak now!")
            audio = recognizer.listen(source, timeout=12, phrase_time_limit=20)
            st.success("✅ Recording complete!")
            return audio
        except sr.WaitTimeoutError:
            st.error("⏰ Listening timed out. Please try again.")
            return None
        except Exception as e:
            st.error(f"🎤 Microphone error: {str(e)}")
            return None

def transcribe_audio(audio, language_code):
    try:
        text = recognizer.recognize_google(audio, language=language_code)
        return text
    except sr.UnknownValueError:
        st.error("❓ Could not understand audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"🌐 Could not request results; {e}")
        return None

def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
            text += page_text + "\n"
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else None
    except Exception as e:
        st.error(f"📜 Error extracting text: {e}")
        return None

@lru_cache(maxsize=32)
def is_legal_document(text, language_name, language_code):
    prompt = f"""
You are an expert in document classification with deep knowledge of Indian legal systems.
The document is in {language_name} (language code: {language_code}). Your task is to determine if the provided text is a legal document (e.g., contract, agreement, legal notice, will, or court filing) based on the presence of legal terminology and structure typical to Indian legal documents in {language_name}.

INSTRUCTIONS:
1. Look for legal terms such as 'contract', 'agreement', 'clause', 'liability', 'compliance', 'penalty', 'court', 'jurisdiction', or their equivalents in {language_name}. For example, in Tamil, look for terms like 'ஒப்பந்தம்' (contract) or 'நீதிமன்றம்' (court).
2. Consider document structure (e.g., numbered clauses, formal tone, signatures) common in Indian legal documents.
3. If the text is ambiguous or lacks legal context, classify it as non-legal.
4. Respond with a JSON object containing:
   - "is_legal": true/false
   - "reason": A concise explanation in {language_name}, max 100 words, explaining why the document is or is not legal.
5. Ensure the response is valid JSON. If no legal terms are found, provide a clear reason in {language_name}.

Text (first 2000 characters):
{text[:2000]}

Output Format:
{{
  "is_legal": true/false,
  "reason": "Explanation in {language_name}"
}}
"""
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        response_text = completion.choices[0].message.content
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            if start_idx >= 0 and end_idx >= 0:
                response_text = response_text[start_idx:end_idx+1]
            result = json.loads(response_text)
        except json.JSONDecodeError:
            st.error("⚠️ Invalid JSON response from model")
            return False, "Model returned invalid response format"
        return result.get("is_legal", False), result.get("reason", "")
    except Exception as e:
        st.error(f"🤖 Error checking document type: {e}")
        return False, str(e)

@lru_cache(maxsize=32)
def analyze_legal_risks(legal_text, language_code, language_name):
    prompt = f"""
You are a senior legal analyst specializing in Indian law, fluent in {language_name} ({language_code}). Your task is to analyze the provided legal document and identify key legal risks, referencing relevant Indian laws, including the Indian Penal Code (IPC), Indian Contract Act, IT Act, or others as applicable.

INSTRUCTIONS:
1. **Language**: All responses, including field names and descriptions, must be in {language_name}.
2. **Risk Identification**:
   - Identify legal risks (e.g., contractual breaches, non-compliance, liabilities, fraud).
   - Categorize risks (e.g., 'Contractual', 'Compliance', 'Liability', translated into {language_name}, such as 'ஒப்பந்தம்' for Contractual in Tamil).
3. **Analysis**:
   - For each risk:
     - Provide a short description (2-3 sentences) in {language_name}.
     - Explain its significance (why it poses a risk) in {language_name}.
     - Suggest practical mitigation strategies tailored to Indian law in {language_name}.
     - Quote the relevant document text ("occurrence") in {language_name}.
     - Assign severity (1-10, 1=low, 10=critical) and impact (1-5, 1=minimal, 5=severe).
     - List relevant IPC sections or other laws (e.g., IPC 420 for fraud) with descriptions in {language_name}.
4. **Edge Cases**:
   - If the text is too short or unclear, return an empty risks list with a note in {language_name}.
   - If no IPC sections apply, specify "No IPC sections directly applicable" in {language_name}.
5. **Output**:
   - Return a valid JSON object.
   - Translate all field names (e.g., "name", "category") into {language_name} (e.g., "பெயர்", "வகை" in Tamil).
   - Ensure descriptions are concise yet informative.

OUTPUT FORMAT:
{{
  "risks": [
    {{
      "name": "[Risk Name in {language_name}]",
      "category": "[Category in {language_name}]",
      "description": "[Description in {language_name}]",
      "why_it_matters": "[Significance in {language_name}]",
      "mitigation": "[Mitigation in {language_name}]",
      "occurrence": "[Relevant Text in {language_name}]",
      "severity": [1-10],
      "impact": [1-5],
      "ipc_sections": [
        {{
          "section": "[Law/Section, e.g., IPC 420]",
          "description": "[Explanation in {language_name}]"
        }}
      ]
    }}
  ]
}}

Legal Document (first 4000 characters):
{legal_text[:4000]}
"""
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096
        )
        response_text = completion.choices[0].message.content
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            if start_idx >= 0 and end_idx >= 0:
                response_text = response_text[start_idx:end_idx+1]
            risks_data = json.loads(response_text)
        except json.JSONDecodeError:
            st.error("⚠️ Invalid JSON response from model")
            return []
        return risks_data.get("risks", [])
    except Exception as e:
        st.error(f"🤖 Error calling Groq API: {e}")
        return []

def chat_about_legal_document(user_query, legal_risks, extracted_text, language_code, language_name, chat_history):
    risk_context = "\n".join(
        f"- {risk.get('name', 'Unnamed')} (Severity: {risk.get('severity', 'N/A')}/10): {risk.get('description', '')}"
        for risk in legal_risks
    ) if legal_risks else "No identified risks"
    
    # Summarize chat history (last 10 messages to avoid token overflow)
    history_summary = ""
    if chat_history:
        history_summary = "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in chat_history[-10:]
        )
    
    prompt = f"""
You are a legal assistant fluent in {language_name} ({language_code}), specializing in Indian law. Your task is to answer the user's query based on the provided legal document, identified risks, and previous conversation history, ensuring responses are accurate and relevant to Indian legal contexts.

CONTEXT:
- Document excerpt: {extracted_text[:2000]}...
- Identified risks: {risk_context}
- Chat history (recent interactions):
{history_summary}

USER QUERY: {user_query}

INSTRUCTIONS:
1. **Language**: Respond entirely in {language_name}, using grammatically correct sentences.
2. **Response**:
   - Answer in 1-3 concise sentences, considering the chat history for context.
   - Reference specific risks, document content, or past interactions if relevant.
   - If the query relates to a risk, mention the risk's name and mitigation if applicable.
3. **Relevance**:
   - If the query is unrelated to the legal domain (e.g., asking about weather), respond with: "This query is not relevant to the legal domain. Please ask about legal risks or document content." translated into {language_name}.
   - If the query is ambiguous, ask for clarification in {language_name}, referencing past context if applicable.
4. **Tone**: Maintain a professional and helpful tone, avoiding legal jargon unless necessary.

OUTPUT:
- A plain text response in {language_name}, max 100 words.
"""
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        font-family: 'Arial', sans-serif;
    }
    .header-gradient {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
    }
    .custom-card {
        background: #252525;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        color: #ffffff;
    }
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }
    .risk-card {
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .risk-card:hover {
        transform: scale(1.03);
        box-shadow: 0 12px 25px rgba(0,0,0,0.5);
    }
    .critical { background: linear-gradient(135deg, #ff4b4b 0%, #2e1a1a 100%); }
    .high { background: linear-gradient(135deg, #ffa500 0%, #2e2a1a 100%); }
    .medium { background: linear-gradient(135deg, #ffcc00 0%, #2e2e1a 100%); }
    .low { background: linear-gradient(135deg, #4CAF50 0%, #1a2e1a 100%); }
    .severity-badge {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2em;
        font-weight: bold;
        color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        position: absolute;
        top: 15px;
        right: 15px;
    }
    .severity-critical { background: #ff4b4b; }
    .severity-high { background: #ffa500; }
    .severity-medium { background: #ffcc00; color: black; }
    .severity-low { background: #4CAF50; }
    .risk-title {
        font-size: 1.5em;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-category {
        font-style: italic;
        color: #dddddd !important;
        font-size: 1em;
        background: rgba(255,255,255,0.1);
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
    .risk-card p {
        color: #ffffff !important;
        line-height: 1.8;
        margin: 10px 0;
    }
    .risk-card strong {
        color: #ffffff !important;
        font-size: 1.1em;
        display: flex;
        align-items: center;
    }
    .risk-card strong::before {
        content: "➤ ";
        color: #ffffff;
        margin-right: 8px;
    }
    .ipc-section {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4d94ff;
    }
    .ipc-section-title {
        font-weight: bold;
        color: #4d94ff !important;
        font-size: 1.1em;
    }
    .metric-card {
        background: #252525;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-value {
        color: #4d94ff !important;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        color: #bbbbbb !important;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .chat-container {
        background: #252525;
        border-radius: 15px;
        padding: 20px;
        max-height: 500px;
        overflow-y: auto;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .user-message {
        background: #2a5298;
        border-radius: 20px 20px 5px 20px;
        padding: 15px;
        margin: 10px 20px 10px auto;
        max-width: 70%;
        color: white;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        position: relative;
        animation: slideIn 0.3s ease;
    }
    .assistant-message {
        background: #3a3a3a;
        border-radius: 20px 20px 20px 5px;
        padding: 15px;
        margin: 10px auto 10px 20px;
        max-width: 70%;
        color: white;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        position: relative;
        animation: slideIn 0.3s ease;
    }
    .message-header {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .message-timestamp {
        font-size: 0.8em;
        color: #bbbbbb;
        margin-left: 10px;
    }
    .copy-button {
        position: absolute;
        top: 10px;
        right: 10px;
        background: transparent;
        border: none;
        color: #ffffff;
        cursor: pointer;
        font-size: 0.9em;
    }
    .chat-input-container {
        display: flex;
        align-items: center;
        background: #252525;
        border-radius: 25px;
        padding: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        margin-top: 20px;
    }
    .chat-input-container input {
        flex: 1;
        background: transparent;
        border: none;
        color: white;
        font-size: 1em;
        padding: 10px;
        outline: none;
    }
    .chat-input-container button {
        background: #2a5298;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 15px;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-left: 5px;
    }
    .chat-input-container button:hover {
        background: #1e3c72;
    }
    .clear-chat-button {
        background: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 10px;
        width: 100%;
        margin-top: 10px;
        transition: all 0.3s ease;
    }
    .clear-chat-button:hover {
        background: #cc0000;
        transform: translateY(-2px);
    }
    .stButton>button {
        background: #2a5298;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #1e3c72;
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab"] {
        background: #2e2e2e;
        color: white;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #2a5298 !important;
        color: white !important;
    }
    .sidebar .sidebar-content {
        background: #252525;
        padding: 20px;
        border-radius: 15px;
    }
    .graph-container {
        background: #252525;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown("""
<div class="header-gradient">
    <h1 style="color: white; margin: 0; padding: 0; font-size: 2.5em;">⚖️ Lexi-Lingua</h1>
    <p style="color: #e0e0e0; margin: 0; padding: 0; font-size: 1.2em;">AI-Powered Legal Analysis with Multilingual Support</p>
</div>
""", unsafe_allow_html=True)

# Reset Button
if st.button("🔄 Reset Application", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Create tabs
tab1, tab2 = st.tabs(["📑 Document Analysis", "💬 Legal Chatbot"])

# Document Analysis Tab
with tab1:
    colored_header(
        label="📑 Document Analysis",
        description="Upload and analyze your legal documents with ease",
        color_name="blue-70"
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.session_state.document_language = st.selectbox(
            "🌐 Select Document Language",
            options=list(LANGUAGE_CODES.keys()),
            index=0,
            help="Choose the language of your document"
        )
    with col2:
        uploaded_file = st.file_uploader(
            "📤 Upload Legal PDF",
            type="pdf",
            help="Upload a PDF document for analysis"
        )
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
    
    if st.session_state.uploaded_file is not None:
        with st.spinner("📝 Extracting text from PDF..."):
            text = extract_text_from_pdf(st.session_state.uploaded_file)
            st.session_state.extracted_text = text
            if text:
                is_legal, reason = is_legal_document(
                    text,
                    st.session_state.document_language,
                    LANGUAGE_CODES[st.session_state.document_language]
                )
                st.session_state.is_legal_document = is_legal
                if not is_legal:
                    st.error(f"🚫 {reason}")
                    st.stop()
            else:
                st.warning("⚠️ No text found in the PDF. Please upload a valid document.")
                st.stop()
        
        with st.expander("👁️‍🗨️ View Extracted Text", expanded=False):
            st.text_area("Document Content", st.session_state.extracted_text, height=250, label_visibility="collapsed")
        
        if st.button("🔍 Analyze Legal Risks", type="primary", use_container_width=True):
            with st.spinner("⚖️ Analyzing document for legal risks..."):
                lang_code = LANGUAGE_CODES[st.session_state.document_language]
                risks = analyze_legal_risks(
                    st.session_state.extracted_text,
                    lang_code,
                    st.session_state.document_language
                )
                st.session_state.legal_risks = risks
                st.session_state.analysis_complete = True
                st.success("✅ Analysis complete!")
                st.balloons()
        
        if st.session_state.analysis_complete and st.session_state.legal_risks:
            colored_header(
                label="📊 Risk Assessment Dashboard",
                description="Explore identified legal risks with interactive visualizations",
                color_name="violet-70"
            )
            
            viz_tab, risks_tab = st.tabs(["📈 Visual Insights", "📋 Detailed Risks"])
            
            with viz_tab:
                set_matplotlib_font(st.session_state.document_language)
                
                col1, col2, col3, col4 = st.columns(4)
                severity_scores = [risk.get('severity', 0) for risk in st.session_state.legal_risks]
                impact_scores = [risk.get('impact', 0) for risk in st.session_state.legal_risks]
                risk_count = len(st.session_state.legal_risks)
                
                with col1:
                    st.markdown(f'<div class="metric-card"><span class="metric-value">📊 {risk_count}</span><br><span class="metric-label">Total Risks</span></div>', unsafe_allow_html=True)
                with col2:
                    avg_severity = f"{sum(severity_scores)/risk_count:.1f}/10" if risk_count else "0"
                    st.markdown(f'<div class="metric-card"><span class="metric-value">⚠️ {avg_severity}</span><br><span class="metric-label">Avg Severity</span></div>', unsafe_allow_html=True)
                with col3:
                    max_severity = f"{max(severity_scores) if severity_scores else 0}/10"
                    st.markdown(f'<div class="metric-card"><span class="metric-value">🔥 {max_severity}</span><br><span class="metric-label">Max Severity</span></div>', unsafe_allow_html=True)
                with col4:
                    avg_impact = f"{sum(impact_scores)/risk_count:.1f}/5" if risk_count else "0"
                    st.markdown(f'<div class="metric-card"><span class="metric-value">💥 {avg_impact}</span><br><span class="metric-label">Avg Impact</span></div>', unsafe_allow_html=True)
                
                st.markdown("### 📉 Impact Analysis")
                st.markdown("Understand the potential impact (1-5) of each risk on your legal document.")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                plt.style.use('dark_background')
                fig2.patch.set_facecolor('#000000')
                ax2.set_facecolor('#000000')
                risks_sorted = sorted(st.session_state.legal_risks, key=lambda x: x.get('severity', 0), reverse=True)
                risk_names = [risk.get('name', f'Risk {i+1}')[:20] for i, risk in enumerate(risks_sorted)]
                impacts = [risk.get('impact', 0) for risk in risks_sorted]
                colors = ['#FF6B6B', '#FFA500', '#FFCC00', '#4CAF50', '#9C27B0']
                bars2 = ax2.barh(risk_names, impacts, color=colors[:len(risk_names)], height=0.7)
                ax2.set_xlabel('Impact Score (1-5)', fontsize=12, color='white', labelpad=10)
                ax2.set_ylabel('Risk Name', fontsize=12, color='white', labelpad=10)
                ax2.set_title('Impact Analysis', color='white', fontsize=14, pad=15)
                ax2.set_xlim(0, 5)
                ax2.grid(axis='x', linestyle='--', alpha=0.3, color='white')
                ax2.tick_params(colors='white', labelsize=10)
                for bar in bars2:
                    width = bar.get_width()
                    ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                             ha='left', va='center', color='white', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig2)
                
                st.markdown("### 🥧 Severity Distribution")
                st.markdown("See the severity distribution for each identified risk.")
                fig3, ax3 = plt.subplots(figsize=(8, 8))
                plt.style.use('dark_background')
                risk_names = [risk.get('name', f'Risk {i+1}')[:20] for i, risk in enumerate(st.session_state.legal_risks)]
                severities = [risk.get('severity', 0) for risk in st.session_state.legal_risks]
                colors = ['#FF6B6B', '#FFA500', '#FFCC00', '#4CAF50', '#9C27B0', '#F06292', '#BA68C8']
                labels = [f"{name} ({sev}/10)" for name, sev in zip(risk_names, severities)]
                explode = [0.05 for _ in severities]
                wedges, texts, autotexts = ax3.pie(
                    severities, 
                    labels=labels, 
                    autopct='%1.1f%%', 
                    startangle=90,
                    colors=colors[:len(severities)],
                    explode=explode,
                    textprops={'color': 'white', 'fontsize': 10}
                )
                ax3.axis('equal')
                ax3.set_title('Severity Distribution by Risk', color='white', fontsize=14)
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontsize(8)
                plt.tight_layout()
                st.pyplot(fig3)
                
                st.markdown("### 🔥 Risk Heatmap")
                st.markdown("Compare average severity and impact across risk categories.")
                categories = list(set(risk.get('category', 'Uncategorized') for risk in st.session_state.legal_risks))
                heatmap_data = np.zeros((len(categories), 2))
                for i, category in enumerate(categories):
                    cat_risks = [r for r in st.session_state.legal_risks if r.get('category') == category]
                    if cat_risks:
                        heatmap_data[i] = [
                            sum(r.get('severity', 0) for r in cat_risks) / len(cat_risks),
                            sum(r.get('impact', 0) for r in cat_risks) / len(cat_risks)
                        ]
                fig4, ax4 = plt.subplots(figsize=(8, 4))
                sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", 
                           xticklabels=['Avg Severity', 'Avg Impact'], yticklabels=categories, ax=ax4,
                           cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='black')
                ax4.set_title('Risk Heatmap by Category', color='white', fontsize=12)
                ax4.tick_params(colors='white', labelsize=8)
                st.pyplot(fig4)
            
            with risks_tab:
                for i, risk in enumerate(st.session_state.legal_risks, 1):
                    severity = risk.get('severity', 0)
                    severity_class = "critical" if severity >= 9 else "high" if severity >= 7 else "medium" if severity >= 4 else "low"
                    badge_class = f"severity-{severity_class}"
                    
                    with st.expander(f"🔍 Risk {i}: {risk.get('name', 'Unnamed Risk')}", expanded=False):
                        st.markdown(f"""
                        <div class="risk-card {severity_class}">
                            <span class="severity-badge {badge_class}">{severity}/10</span>
                            <div class="risk-title">{risk.get('name', 'Unnamed Risk')}</div>
                            <div class="risk-category">{risk.get('category', 'N/A')}</div>
                            <p><strong>📝 Description:</strong> {risk.get('description', 'N/A')}</p>
                            <p><strong>❓ Why It Matters:</strong> {risk.get('why_it_matters', 'N/A')}</p>
                            <p><strong>🛡️ Mitigation:</strong> {risk.get('mitigation', 'N/A')}</p>
                            <p><strong>📍 Occurrence:</strong> {risk.get('occurrence', 'N/A')}</p>
                            <p><strong>💥 Impact:</strong> {risk.get('impact', 0)}/5</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if risk.get('ipc_sections'):
                            st.markdown("### 📜 Relevant IPC Sections")
                            for section in risk['ipc_sections']:
                                st.markdown(f"""
                                <div class="ipc-section">
                                    <div class="ipc-section-title">{section.get('section', 'N/A')}</div>
                                    <p>{section.get('description', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)

# AI Assistant Tab
with tab2:
    colored_header(
        label="💬 AI Legal Assistant",
        description="Chat with your document in your preferred language",
        color_name="violet-70"
    )
    
    if not st.session_state.extracted_text or not st.session_state.is_legal_document:
        st.info("ℹ️ Please upload and analyze a valid legal document first to use the chatbot.")
    else:
        # Language Selection
        selected_lang_name = st.selectbox(
            "🌍 Select Speaking Language",
            options=list(LANGUAGE_CODES.keys()),
            help="Choose your preferred language for interaction"
        )
        st.session_state.selected_speech_lang = LANGUAGE_CODES[selected_lang_name]
        
        # Chat History
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for idx, message in enumerate(st.session_state.chat_history):
                timestamp = message.get("timestamp", datetime.now().strftime("%H:%M:%S"))
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <div class="message-header">
                            <span>🧑 User</span>
                            <span class="message-timestamp">{timestamp}</span>
                        </div>
                        <p>{message['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <div class="message-header">
                            <span>🤖 Assistant</span>
                            <span class="message-timestamp">{timestamp}</span>
                            <button class="copy-button" onclick="navigator.clipboard.writeText('{message['content'].replace("'", "\\'")}')">Copy</button>
                        </div>
                        <p>{message['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clear Chat History with Confirmation
            if st.button("🗑️ Clear Chat History", key="clear_chat", help="Clear all chat history"):
                if st.checkbox("Confirm clearing chat history"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Initialize session state for input tracking
        if 'last_input' not in st.session_state:
            st.session_state.last_input = ""
        if 'input_processed' not in st.session_state:
            st.session_state.input_processed = False

        # Chat Input with Voice Record Button
        with stylable_container(
            key="chat_input_container",
            css_styles="""
            .chat-input-container {
                margin-top: 20px;
            }
            """
        ):
            with st.form(key="chat_form", clear_on_submit=True):
                input_col, button_col = st.columns([10, 1])
                with input_col:
                    user_input = st.text_input(
                        label="Chat Input",
                        placeholder="Type your legal question here... (Press Enter to send)",
                        key="chat_input",
                        label_visibility="collapsed"
                    )
                with button_col:
                    submitted = st.form_submit_button(
                        "🎙️",
                        help="Record voice input"
                    )
                
                # Handle Enter key submission
                if user_input.strip() and user_input != st.session_state.last_input:
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    with st.spinner("🤖 Generating response..."):
                        response = chat_about_legal_document(
                            user_input,
                            st.session_state.legal_risks,
                            st.session_state.extracted_text,
                            LANGUAGE_CODES[st.session_state.document_language],
                            st.session_state.document_language,
                            tuple(st.session_state.chat_history)
                        )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                    # Update last input to prevent reprocessing
                    st.session_state.last_input = user_input
                    # Limit history to 20 messages
                    if len(st.session_state.chat_history) > 20:
                        st.session_state.chat_history = st.session_state.chat_history[-20:]
                    st.session_state.voice_input = ""
                    st.rerun()
                elif not user_input.strip():
                    # Reset last_input when input is cleared
                    st.session_state.last_input = ""
        
        # Handle Voice Input
        if submitted:
            with st.spinner("🎤 Recording..."):
                audio = record_audio(st.session_state.selected_speech_lang)
                if audio:
                    transcribed_text = transcribe_audio(audio, st.session_state.selected_speech_lang)
                    if transcribed_text:
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": transcribed_text,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        with st.spinner("🤖 Generating response..."):
                            response = chat_about_legal_document(
                                transcribed_text,
                                st.session_state.legal_risks,
                                st.session_state.extracted_text,
                                LANGUAGE_CODES[st.session_state.document_language],
                                st.session_state.document_language,
                                tuple(st.session_state.chat_history)
                            )
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                        # Limit history to 20 messages
                        if len(st.session_state.chat_history) > 20:
                            st.session_state.chat_history = st.session_state.chat_history[-20:]
                        st.session_state.voice_input = ""
                        st.rerun()

# Sidebar
with st.sidebar:
    colored_header(
        label="📚 User Guide",
        description="How to use Lexi-Lingua",
        color_name="blue-70"
    )
    st.markdown("""
    ### 📑 Document Analysis
    1. **Upload**: Select a legal PDF document
    2. **Language**: Choose document language
    3. **Analyze**: Click "Analyze Legal Risks"
    4. **Review**: Explore risks and visualizations
    
    ### 💬 AI Assistant
    1. Analyze a legal document first
    2. Select your speaking language
    3. Ask legal questions via text (press Enter) or voice (click 🎙️)
    4. View and copy responses in the chat history
    
    **Sample Questions:**
    - What are the key risks?
    - Explain liability clauses
    - Suggest mitigation strategies
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #bbbbbb; font-size: 0.9em;">
        © 2025 Lexi-Lingua<br>
        Built with ❤️ by Pragateesh G
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #bbbbbb; margin-top: 2rem; padding: 1rem; border-top: 1px solid #3a3a3a;">
    Powered by AI ⚡ | Lexi-Lingua v1.0
</div>
""", unsafe_allow_html=True)