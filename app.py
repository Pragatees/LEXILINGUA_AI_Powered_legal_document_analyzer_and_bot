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
from datetime import datetime
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
import pandas as pd

# Streamlit app setup - Must be the first Streamlit command
st.set_page_config(page_title="Lexi-Lingua", layout="wide", initial_sidebar_state="expanded", page_icon="⚖️")

# Set your Groq API key
api_key = "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"  # Replace with your actual Groq API key
client = Groq(api_key=api_key)

# Available languages for translation
language_dict = {
    "english": {"name": "English", "code": "en"},
    "tamil": {"name": "தமிழ்", "code": "ta"},
    "hindi": {"name": "हिंदी", "code": "hi"},
    "malayalam": {"name": "മലയാളം", "code": "ml"},
    "telugu": {"name": "తెలుగు", "code": "te"},
    "kannada": {"name": "ಕನ್ನಡ", "code": "kn"}
}

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize session state variables
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'document_language' not in st.session_state:
    st.session_state.document_language = "english"
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
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""
if 'input_processed' not in st.session_state:
    st.session_state.input_processed = False
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = ""
if 'reset_clicked' not in st.session_state:
    st.session_state.reset_clicked = False
if 'recording' not in st.session_state:
    st.session_state.recording = False

# Translation function (placeholder)
def translate_to_language(text, language):
    return text  # Replace with actual translation API in production

# PDF text extraction
def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            text = page.get_text("text")
            if text:
                full_text += text
        return full_text
    except Exception as e:
        st.error(f"📜 Error extracting text: {e} 😕")
        return ""

# Speech recognition functions
def record_audio(language_code):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            st.info(f"🎙️ Listening for {language_code}... Speak now! 🗣️")
            audio = recognizer.listen(source, timeout=12, phrase_time_limit=20)
            st.success("✅ Recording complete! 🎉")
            return audio
        except sr.WaitTimeoutError:
            st.error("⏰ Listening timed out. Please try again. 😔")
            return None
        except Exception as e:
            st.error(f"🎤 Microphone error: {str(e)} 😕")
            return None

def transcribe_audio(audio, language_code):
    try:
        text = recognizer.recognize_google(audio, language=language_code)
        return text
    except sr.UnknownValueError:
        st.error("❓ Could not understand audio. Please try again. 😔")
        return None
    except sr.RequestError as e:
        st.error(f"🌐 Could not request results; {e} 😕")
        return None

# Legal risk analysis with corrected IPC mapping
def analyze_legal_risks(legal_text, language_code, language_name):
    prompt = f"""
You are a senior legal analyst specializing in Indian law, fluent in {language_name} ({language_code}). Your task is to analyze the provided legal document and identify key legal risks, referencing relevant Indian laws, including the Indian Penal Code (IPC), Indian Contract Act, IT Act, or others as applicable.

INSTRUCTIONS:
1. **Language**: All field values (e.g., risk name, description) must be in {language_name}, but field names (e.g., "risk name", "category") must remain in English.
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
     - List relevant IPC sections or other laws (e.g., IPC 420 for fraud, IPC 406 for criminal breach of trust) with accurate descriptions in {language_name}.
4. **Edge Cases**:
   - If the text is too short or unclear, return an empty risks list with a note in {language_name}.
   - If no IPC sections apply, specify "No IPC sections directly applicable" in {language_name}.
5. **Output**:
   - Return a valid JSON object.
   - Use English field names (e.g., "risk name", "category", "description", "why_it_matters", "mitigation", "occurrence", "severity", "impact", "ipc_sections").
   - Ensure all field values are in {language_name}.
   - Ensure descriptions are concise yet informative.

OUTPUT FORMAT:
{{
  "risks": [
    {{
      "risk name": "[Risk Name in {language_name}]",
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
                response_text = response_text[start_idx:end_idx + 1]
            risks_data = json.loads(response_text)
            # Validate and correct IPC sections if needed
            for risk in risks_data.get("risks", []):
                if not risk.get("ipc_sections"):
                    risk["ipc_sections"] = [{"section": "No IPC sections directly applicable", "description": f"No relevant IPC sections identified in {language_name}"}]
                elif any("No IPC" in sec["section"] for sec in risk["ipc_sections"]):
                    continue
                else:
                    # Example mapping (to be adjusted based on actual risks)
                    if "fraud" in risk["risk name"].lower():
                        risk["ipc_sections"].append({"section": "IPC 420", "description": f"Fraud under Indian law in {language_name}"})
                    elif "breach" in risk["risk name"].lower():
                        risk["ipc_sections"].append({"section": "IPC 406", "description": f"Criminal breach of trust in {language_name}"})
            return risks_data.get("risks", [])
        except json.JSONDecodeError:
            st.error("⚠️ Invalid JSON response from model 😕")
            return []
    except Exception as e:
        st.error(f"🤖 Error calling Groq API: {e} 😕")
        return []

# Document summary function with abstractive summary
def generate_document_summary(extracted_text, language_code, language_name):
    prompt = f"""
You are a legal expert fluent in {language_name} ({language_code}), specializing in Indian law. Your task is to generate an abstractive summary and key points of the provided legal document based on your understanding of its content.

INSTRUCTIONS:
1. **Language**: Provide the summary and key points entirely in {language_name}.
2. **Summary**:
   - Write a concise abstractive summary (5-7 sentences) describing the document's purpose, main clauses, legal context, key obligations, and potential implications in your own words.
   - Focus on legal intent, terms, and relevance to Indian law, avoiding direct text extraction.
3. **Key Points**:
   - List 5-7 bullet points highlighting critical legal elements (e.g., parties involved, obligations, penalties, compliance requirements) in your own words.
   - Ensure each point is clear, original, and relevant to Indian law.
4. **Edge Cases**:
   - If the text is too short or unclear, return a note in {language_name} stating "Insufficient text for summary."
5. **Output**:
   - Return a JSON object with "summary" (string) and "key_points" (list of strings) fields.
   - All content must be in {language_name}.

OUTPUT FORMAT:
{{
  "summary": "[Abstractive Summary in {language_name}]",
  "key_points": [
    "[Point 1 in {language_name}]",
    "[Point 2 in {language_name}]",
    ...
  ]
}}

Document (first 4000 characters):
{extracted_text[:4000]}
"""
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1500
        )
        response_text = completion.choices[0].message.content
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            if start_idx >= 0 and end_idx >= 0:
                response_text = response_text[start_idx:end_idx + 1]
            summary_data = json.loads(response_text)
            return summary_data
        except json.JSONDecodeError:
            st.error("⚠️ Invalid JSON response for summary 😕")
            return {"summary": f"Error generating summary in {language_name}", "key_points": []}
    except Exception as e:
        st.error(f"🤖 Error generating summary: {e} 😕")
        return {"summary": f"Error generating summary in {language_name}", "key_points": []}

# NER Extraction Function using LLaMA with updated LOCATION label
def extract_ner_with_llama(extracted_text, language_code, language_name):
    prompt = f"""
You are an AI expert in Named Entity Recognition (NER), fluent in {language_name} ({language_code}). Your task is to identify named entities in the provided legal document text and categorize them as PERSON, ORGANIZATION, LOCATION, DATE, or OTHER.

INSTRUCTIONS:
1. **Language**: Analyze the text in {language_name} and return entity text in the original language of the document.
2. **Entity Types**:
   - PERSON: Names of individuals (e.g., "John Doe").
   - ORGANIZATION: Names of companies, institutions, etc. (e.g., "Acme Corp").
   - LOCATION: Places like countries, cities, or other locations (e.g., "India", "Delhi").
   - DATE: Specific dates or time periods (e.g., "2023-01-01", "January 2023").
   - OTHER: Any other relevant entities not fitting the above categories.
3. **Output**:
   - Return a valid JSON array of objects, each with "text" (the entity) and "label" (the entity type).
   - Ensure entities are unique and relevant to the legal context.
   - If no entities are found, return an empty array.
4. **Edge Cases**:
   - If the text is too short or unclear, return an empty array.
   - Avoid generic terms (e.g., "the company") unless they are specific named entities.
5. **Limit**: Process up to the first 10000 characters of the text to avoid performance issues.

OUTPUT FORMAT:
[
  {{"text": "[Entity Text]", "label": "[PERSON|ORGANIZATION|LOCATION|DATE|OTHER]"}},
  ...
]

Document (first 10000 characters):
{extracted_text[:10000]}
"""
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1500
        )
        response_text = completion.choices[0].message.content
        try:
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]")
            if start_idx >= 0 and end_idx >= 0:
                response_text = response_text[start_idx:end_idx + 1]
            entities = json.loads(response_text)
            return entities
        except json.JSONDecodeError:
            st.error("⚠️ Invalid JSON response for NER 😕")
            return []
    except Exception as e:
        st.error(f"🤖 Error performing NER: {e} 😕")
        return []

# Chatbot function
def chat_about_legal_document(user_query, legal_risks, extracted_text, language_code, language_name, chat_history):
    risk_context = "\n".join(
        f"- {risk.get('risk name', 'Unnamed')} (Severity: {risk.get('severity', 'N/A')}/10): {risk.get('description', '')}"
        for risk in legal_risks
    ) if legal_risks else "No identified risks"
    
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
        return f"Error generating response: {str(e)} 😕"

# Custom CSS for enhanced UI with updated NER card-based display
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        font-family: 'Arial', sans-serif;
        color: #ffffff;
    }
    .header-gradient {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 15px rgba(0,0,0,0.2), 0 0 10px #2a5298;
        margin-bottom: 2rem;
        animation: shimmer 2s infinite alternate;
    }
    @keyframes shimmer {
        from { box-shadow: 0 8px 15px rgba(0,0,0,0.2), 0 0 5px #2a5298; }
        to { box-shadow: 0 8px 15px rgba(0,0,0,0.2), 0 0 15px #4d94ff; }
    }
    .custom-card {
        background: linear-gradient(135deg, #252525 0%, #333333 100%);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        border: 2px solid #4d94ff;
        position: relative;
        overflow: hidden;
    }
    .custom-card::before {
        content: "✨";
        position: absolute;
        top: 10px;
        left: 10px;
        color: #4d94ff;
        opacity: 0.5;
    }
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(77, 148, 255, 0.4);
    }
    .metric-card {
        background: #2e2e2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        border: 1px solid #2a5298;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.5em;
        color: #2a5298;
        font-weight: bold;
        text-shadow: 0 0 5px #2a5298;
    }
    .metric-card p {
        margin: 5px 0 0;
        font-size: 1em;
        color: #ffffff;
        font-weight: bold;
    }
    .risk-card {
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
        border: 2px dashed #ffffff;
    }
    .risk-card:hover {
        transform: scale(1.03);
        box-shadow: 0 12px 25px rgba(255, 255, 255, 0.2);
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
        text-shadow: 0 0 5px #ffffff;
    }
    .risk-category {
        font-style: italic;
        color: #dddddd !important;
        font-size: 1em;
        background: rgba(255,255,255,0.1);
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
        border: 1px solid #4d94ff;
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
        color: #4d94ff;
        margin-right: 8px;
    }
    .ipc-section {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4d94ff;
        position: relative;
    }
    .ipc-section::after {
        content: "📜";
        position: absolute;
        top: 5px;
        right: 10px;
        color: #4d94ff;
        opacity: 0.7;
    }
    .ipc-section-title {
        font-weight: bold;
        color: #4d94ff !important;
        font-size: 1.1em;
    }
    .chat-container {
        background: linear-gradient(135deg, #252525 0%, #333333 100%);
        border-radius: 15px;
        padding: 20px;
        max-height: 500px;
        overflow-y: auto;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        border: 2px solid #2a5298;
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
    .user-message::before {
        content: "🧑";
        position: absolute;
        top: 5px;
        left: 5px;
        color: #ffffff;
        opacity: 0.7;
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
    .assistant-message::before {
        content: "🤖";
        position: absolute;
        top: 5px;
        left: 5px;
        color: #ffffff;
        opacity: 0.7;
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
        transition: all 0.3s ease;
    }
    .copy-button:hover {
        color: #4d94ff;
        transform: scale(1.1);
    }
    .chat-input-container {
        display: flex;
        align-items: center;
        background: #252525;
        border-radius: 25px;
        padding: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2), 0 0 5px #2a5298;
        margin-top: 20px;
        border: 2px solid #4d94ff;
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
        background: linear-gradient(90deg, #2a5298 0%, #4d94ff 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 15px;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-left: 5px;
    }
    .chat-input-container button:hover {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        transform: translateY(-2px);
    }
    .clear-chat-button {
        background: linear-gradient(135deg, #ff4b4b 0%, #cc0000 100%);
        color: white;
        border-radius: 10px;
        padding: 10px;
        width: 100%;
        margin-top: 10px;
        transition: all 0.3s ease;
    }
    .clear-chat-button:hover {
        background: linear-gradient(135deg, #cc0000 0%, #990000 100%);
        transform: translateY(-2px);
    }
    .stButton>button {
        background: linear-gradient(90deg, #2a5298 0%, #4d94ff 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        transform: translateY(-2px);
    }
    .stButton>button.clicked {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        animation: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: #2e2e2e;
        color: white;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        border: 1px solid #2a5298;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2a5298 0%, #4d94ff 100%) !important;
        color: white !important;
        transform: scale(1.05);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #252525 0%, #333333 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #4d94ff;
    }
    .color-legend {
        display: flex;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    .color-box {
        width: 20px;
        height: 20px;
        margin-right: 10px;
        border-radius: 3px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin-right: 20px;
        margin-bottom: 10px;
        color: white;
        font-size: 0.9em;
    }
    .ner-card-container {
        background: linear-gradient(135deg, #2e2e2e 0%, #3a3a3a 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        border: 2px solid #4d94ff;
        margin-bottom: 20px;
    }
    .ner-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        border-left: 5px solid;
        position: relative;
    }
    .ner-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(77, 148, 255, 0.3);
    }
    .ner-card-person { border-left-color: #4CAF50; }
    .ner-card-organization { border-left-color: #FFA500; }
    .ner-card-location { border-left-color: #FF69B4; }
    .ner-card-date { border-left-color: #00CED1; }
    .ner-card-other { border-left-color: #2a5298; }
    .ner-label {
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: bold;
        text-transform: uppercase;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        display: inline-block;
        position: absolute;
        top: 10px;
        right: 10px;
    }
    .ner-label-person { background: #4CAF50; color: white; }
    .ner-label-organization { background: #FFA500; color: white; }
    .ner-label-location { background: #FF69B4; color: white; }
    .ner-label-date { background: #00CED1; color: white; }
    .ner-label-other { background: #2a5298; color: white; }
    .ner-text {
        font-size: 1.2em;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 5px;
    }
    .ner-filter {
        background: #252525;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
        border: 2px solid #4d94ff;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown("""
<div class="header-gradient">
    <h1 style="color: white; margin: 0; padding: 0; font-size: 2.5em;">⚖️ Lexi-Lingua ✨</h1>
    <p style="color: #e0e0e0; margin: 0; padding: 0; font-size: 1.2em;">AI-Powered Legal Analysis with Multilingual Support 🌟🎯</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📑 Document Analysis 🌍", "💬 AI Legal Assistant 🤖"])

# Document Analysis Tab
with tab1:
    colored_header(
        label="📑 Document Analysis 📊",
        description="Upload and analyze your legal documents with ease 🚀✨",
        color_name="blue-70"
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.session_state.document_language = st.selectbox(
            "🌐 Select Document Language 🌍",
            options=list(language_dict.keys()),
            index=0,
            help="Choose the language of your document 🌏✨"
        )
    with col2:
        uploaded_file = st.file_uploader(
            "📤 Upload Legal PDF 📜",
            type="pdf",
            help="Upload a PDF document for analysis 📑🚀"
        )
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
    
    if st.session_state.uploaded_file is not None:
        with st.spinner("📝 Extracting text from PDF... ⏳"):
            st.session_state.extracted_text = extract_text_from_pdf(st.session_state.uploaded_file)
            if st.session_state.extracted_text.strip():
                st.success("✅ Text extraction complete! 🎉✨")
                with st.expander("👁️‍🗨️ View Extracted Text 📖", expanded=False):
                    st.text_area("Document Content", st.session_state.extracted_text, height=250, label_visibility="collapsed")
                
                if st.button("🔍 Analyze Legal Risks ⚖️", type="primary", use_container_width=True, key="analyze_button"):
                    with st.spinner("⚖️ Analyzing document for legal risks... ⏳"):
                        lang_code = language_dict[st.session_state.document_language]["code"]
                        lang_name = language_dict[st.session_state.document_language]["name"]
                        st.session_state.legal_risks = analyze_legal_risks(
                            st.session_state.extracted_text,
                            lang_code,
                            lang_name
                        )
                        # Generate document summary
                        st.session_state.document_summary = generate_document_summary(
                            st.session_state.extracted_text,
                            lang_code,
                            lang_name
                        )
                        st.session_state.analysis_complete = True
                        st.success("✅ Analysis complete! 🎉🌟")
                        st.balloons()
                
                if st.session_state.analysis_complete and st.session_state.legal_risks:
                    colored_header(
                        label="📋 Legal Risk Analysis 🔍",
                        description="Explore insights and visualizations for identified risks 📈✨",
                        color_name="violet-70"
                    )
                    
                    # Calculate metrics
                    total_risks = len(st.session_state.legal_risks)
                    avg_severity = round(np.mean([risk.get('severity', 0) for risk in st.session_state.legal_risks]), 1) if total_risks > 0 else 0
                    max_severity = max([risk.get('severity', 0) for risk in st.session_state.legal_risks], default=0)
                    critical_risks = sum(1 for risk in st.session_state.legal_risks if risk.get('severity', 0) >= 9)
                    
                    # Display metrics in four columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{total_risks} 🚨</h3>
                            <p>Total Risks</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{avg_severity} 📊</h3>
                            <p>Avg Severity</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{max_severity} ⚠️</h3>
                            <p>Max Severity</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{critical_risks} 🔥</h3>
                            <p>Critical Risks</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add spacing to lower tabs
                    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
                    
                    # Create tabs for visualizations, data, and summary
                    vis_tab, data_tab, summary_tab = st.tabs(["📊 Risk Visualizations 🌈", "📋 Risk Data 📋", "📝 Document Summary 📖"])
                    
                    with vis_tab:
                        # Impact Analysis Bar Graph
                        impacts = [risk.get('impact', 0) for risk in st.session_state.legal_risks]
                        if st.session_state.document_language == "english":
                            risk_labels = [risk.get('risk name', f"Risk {i+1}") for i, risk in enumerate(st.session_state.legal_risks)]
                        else:
                            risk_labels = [f"Risk {i+1}" for i in range(len(st.session_state.legal_risks))]
                        colors = ['#FFFF00', '#FFA500', '#FF69B4', '#00CED1']  # Yellow, Orange, Pink, Cyan
                        plt.figure(figsize=(4, 3))
                        bars = plt.barh(risk_labels, impacts, color=colors[:len(risk_labels)])
                        plt.title("Impact Analysis 📉", color='white', fontsize=10)
                        plt.xlabel("Impact Score (1-5)", color='white', fontsize=8)
                        plt.ylabel("Risks", color='white', fontsize=8)
                        plt.xticks(np.arange(0, 6, 1), color='white', fontsize=8)
                        plt.yticks(color='white', fontsize=8)
                        plt.gca().set_facecolor('#1a1a1a')
                        plt.gcf().set_facecolor('#1a1a1a')
                        for i, v in enumerate(impacts):
                            plt.text(v + 0.1, i, str(v), color='white', va='center', fontsize=8)
                        st.pyplot(plt)
                        plt.close()

                        # Display risk mapping outside the graph for non-English languages
                        if st.session_state.document_language != "english":
                            st.markdown(f"### Risk Legend (Translated to {st.session_state.document_language.title()}) 🌐:")
                            for i, risk in enumerate(st.session_state.legal_risks, 1):
                                translated_name = risk.get('risk name', 'Unnamed')
                                if st.session_state.document_language == "tamil":
                                    if i == 1:
                                        translated_name = "தகவல் பாதுகாப்பு இல்லாமை"  # Unauthorized Disclosure
                                    elif i == 2:
                                        translated_name = "ஒப்பந்தம் இல்லாமை"  # Non-Compliance
                                    elif i == 3:
                                        translated_name = "சட்டவிரோத பரிமாற்றம்"  # Illegal Transaction
                                    elif i == 4:
                                        translated_name = "பொறுப்பு மீறல்"  # Liability Breach
                                st.markdown(f"- Risk {i} -> {translated_name} ✨")

                        # Severity Distribution Pie Chart
                        severities = [risk.get('severity', 0) for risk in st.session_state.legal_risks]
                        if st.session_state.document_language == "english":
                            labels = [risk.get('risk name', f"Risk {i+1} ({risk.get('severity', 0)}/10)") for i, risk in enumerate(st.session_state.legal_risks)]
                        else:
                            labels = [f"Risk {i+1} ({risk.get('severity', 0)}/10)" for i, risk in enumerate(st.session_state.legal_risks)]
                        colors = ['#FFFF00', '#FFA500', '#FF69B4', '#00CED1']  # Yellow, Orange, Pink, Cyan
                        plt.figure(figsize=(4, 4))
                        plt.pie(severities, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 8})
                        plt.title("Severity Distribution by Risk 📊", color='white', fontsize=10)
                        plt.gca().set_facecolor('#1a1a1a')
                        plt.gcf().set_facecolor('#1a1a1a')
                        st.pyplot(plt)
                        plt.close()

                        # Display risk mapping outside the graph for non-English languages
                        if st.session_state.document_language != "english":
                            st.markdown(f"### Risk Legend (Translated to {st.session_state.document_language.title()}) 🌐:")
                            for i, risk in enumerate(st.session_state.legal_risks, 1):
                                translated_name = risk.get('risk name', 'Unnamed')
                                if st.session_state.document_language == "tamil":
                                    if i == 1:
                                        translated_name = "தகவல் பாதுகாப்பு இல்லாமை"  # Unauthorized Disclosure
                                    elif i == 2:
                                        translated_name = "ஒப்பந்தம் இல்லாமை"  # Non-Compliance
                                    elif i == 3:
                                        translated_name = "சட்டவிரோத பரிமாற்றம்"  # Illegal Transaction
                                    elif i == 4:
                                        translated_name = "பொறுப்பு மீறல்"  # Liability Breach
                                st.markdown(f"- Risk {i} -> {translated_name} ✨")
                    
                    with data_tab:
                        # Sort risks by severity (descending)
                        sorted_risks = sorted(st.session_state.legal_risks, key=lambda x: x.get('severity', 0), reverse=True)
                        for i, risk in enumerate(sorted_risks, 1):
                            severity = risk.get('severity', 0)
                            severity_class = "critical" if severity >= 9 else "high" if severity >= 7 else "medium" if severity >= 4 else "low"
                            badge_class = f"severity-{severity_class}"
                            
                            with st.expander(f"🔍 Risk {i}: {risk.get('risk name', 'Unnamed Risk')} 🚨", expanded=False):
                                st.markdown(f"""
                                <div class="risk-card {severity_class}">
                                    <span class="severity-badge {badge_class}">{severity}/10</span>
                                    <div class="risk-title">{risk.get('risk name', 'Unnamed Risk')} 🎯</div>
                                    <div class="risk-category">{risk.get('category', 'N/A')}</div>
                                    <p><strong>📝 description:</strong> {risk.get('description', 'N/A')}</p>
                                    <p><strong>❓ why_it_matters:</strong> {risk.get('why_it_matters', 'N/A')}</p>
                                    <p><strong>🛡️ mitigation:</strong> {risk.get('mitigation', 'N/A')}</p>
                                    <p><strong>📍 occurrence:</strong> {risk.get('occurrence', 'N/A')}</p>
                                    <p><strong>💥 impact:</strong> {risk.get('impact', 0)}/5</p>
                                </div>
                                """, unsafe_allow_html=True)
                                if risk.get('ipc_sections'):
                                    st.markdown("### 📜 Relevant IPC Sections 📚")
                                    for section in risk['ipc_sections']:
                                        st.markdown(f"""
                                        <div class="ipc-section">
                                            <div class="ipc-section-title">{section.get('section', 'N/A')}</div>
                                            <p>{section.get('description', 'N/A')}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"No IPC sections directly applicable (in {lang_name}: {translate_to_language('No IPC sections directly applicable', st.session_state.document_language)}).")
                    
                    with summary_tab:
                        summary_data = st.session_state.document_summary
                        st.markdown("### 📝 Document Summary 📖")
                        st.write(summary_data.get('summary', 'No summary available.'))
                        st.markdown("### 🔑 Key Points 🔍")
                        for point in summary_data.get('key_points', []):
                            st.markdown(f"- {point} ✨")
                        
                        # NER Section with Card-Based UI
                        st.markdown("### 🧠 Named Entity Recognition (NER) 🔍")
                        lang_code = language_dict[st.session_state.document_language]["code"]
                        lang_name = language_dict[st.session_state.document_language]["name"]
                        entities = extract_ner_with_llama(st.session_state.extracted_text, lang_code, lang_name)
                        if entities:
                            # Prepare data for display
                            df = pd.DataFrame(entities, columns=["text", "label"])
                            df.columns = ["Entity", "Type"]
                            
                            # Filter by entity type
                            entity_types = sorted(set(df["Type"]))
                            selected_types = st.multiselect(
                                "🔍 Filter by Entity Type",
                                options=entity_types,
                                default=entity_types,
                                key="ner_filter",
                                help="Select entity types to display 🌟"
                            )
                            
                            # Apply filter
                            if selected_types:
                                filtered_df = df[df["Type"].isin(selected_types)]
                            else:
                                filtered_df = df
                            
                            # Display NER results as cards
                            st.markdown('<div class="ner-card-container">', unsafe_allow_html=True)
                            for _, row in filtered_df.iterrows():
                                entity = row["Entity"]
                                label = row["Type"].lower()
                                st.markdown(f"""
                                <div class="ner-card ner-card-{label}">
                                    <div class="ner-text">{entity}</div>
                                    <span class="ner-label ner-label-{label}">{label.upper()}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info("ℹ️ No named entities detected in the document. 😔")
            else:
                st.error("⚠ No text found. It might be a scanned PDF (image-based). OCR is needed in that case. 😔")
                st.info("💡 **Tip**: If the PDF is image-based, consider using OCR tools like Tesseract or EasyOCR. 🛠️")

# AI Assistant Tab
with tab2:
    colored_header(
        label="💬 AI Legal Assistant 🤖",
        description="Chat with your document in your preferred language 🌐✨",
        color_name="violet-70"
    )
    
    if not st.session_state.extracted_text or not st.session_state.is_legal_document:
        st.info("ℹ️ Please upload and analyze a valid legal document first to use the chatbot. 📜🚀")
    else:
        # Language Selection for Speech
        selected_speech_lang = st.selectbox(
            "🌍 Select Speaking Language 🌏",
            options=list(language_dict.keys()),
            help="Choose your preferred language for interaction 🗣️✨",
            key="speech_lang"
        )
        st.session_state.selected_speech_lang = f"{language_dict[selected_speech_lang]['code']}-IN"
        
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
                        <p>{message['content']} 🎉</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <div class="message-header">
                            <span>🤖 Assistant</span>
                            <span class="message-timestamp">{timestamp}</span>
                            <button class="copy-button" onclick="navigator.clipboard.writeText('{message['content'].replace("'", "\\'")}')">Copy 📋</button>
                        </div>
                        <p>{message['content']} 🌟</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clear Chat History with Confirmation
            if st.button("🗑️ Clear Chat History 🧹", key="clear_chat", help="Clear all chat history ✨"):
                if st.checkbox("Confirm clearing chat history ✅"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Chat Input with Voice Record and Stop Buttons
        with stylable_container(
            key="chat_input_container",
            css_styles="""
            .chat-input-container {
                margin-top: 20px;
            }
            """
        ):
            with st.form(key="chat_form", clear_on_submit=True):
                input_col, button_col1, button_col2 = st.columns([10, 1, 1])
                with input_col:
                    user_input = st.text_input(
                        label="Chat Input",
                        placeholder="Type your legal question here... (Press Enter to send) ✍️",
                        key="chat_input",
                        label_visibility="collapsed"
                    )
                with button_col1:
                    record_button = st.form_submit_button(
                        "🎙️ Record Voice 🗣️",
                        help="Start recording voice input ✨"
                    )
                with button_col2:
                    stop_button = st.form_submit_button(
                        "🛑 Stop Recording ⏹️",
                        help="Stop recording voice input ✨"
                    )
                
                # Handle Enter key submission
                if user_input.strip() and user_input != st.session_state.last_input:
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    with st.spinner("🤖 Generating response... ⏳"):
                        lang_code = language_dict[st.session_state.document_language]["code"]
                        lang_name = language_dict[st.session_state.document_language]["name"]
                        response = chat_about_legal_document(
                            user_input,
                            st.session_state.legal_risks,
                            st.session_state.extracted_text,
                            lang_code,
                            lang_name,
                            tuple(st.session_state.chat_history)
                        )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                    st.session_state.last_input = user_input
                    if len(st.session_state.chat_history) > 20:
                        st.session_state.chat_history = st.session_state.chat_history[-20:]
                    st.session_state.voice_input = ""
                    st.rerun()
                elif not user_input.strip():
                    st.session_state.last_input = ""
        
        # Handle Voice Recording with Stop Functionality
        if record_button and not st.session_state.recording:
            st.session_state.recording = True
            with st.spinner("🎤 Recording... ⏳"):
                audio = record_audio(st.session_state.selected_speech_lang)
                if audio:
                    transcribed_text = transcribe_audio(audio, st.session_state.selected_speech_lang)
                    if transcribed_text:
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": transcribed_text,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        with st.spinner("🤖 Generating response... ⏳"):
                            lang_code = language_dict[st.session_state.document_language]["code"]
                            lang_name = language_dict[st.session_state.document_language]["name"]
                            response = chat_about_legal_document(
                                transcribed_text,
                                st.session_state.legal_risks,
                                st.session_state.extracted_text,
                                lang_code,
                                lang_name,
                                tuple(st.session_state.chat_history)
                            )
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                        if len(st.session_state.chat_history) > 20:
                            st.session_state.chat_history = st.session_state.chat_history[-20:]
                        st.session_state.voice_input = ""
                        st.session_state.recording = False
                        st.rerun()
                st.session_state.recording = False
        
        # Handle Stop Recording
        if stop_button and st.session_state.recording:
            st.session_state.recording = False
            st.info("🛑 Recording stopped. 😊")
            st.rerun()

# Sidebar
with st.sidebar:
    colored_header(
        label="📚 User Guide 📖",
        description="How to use Lexi-Lingua 🚀✨",
        color_name="blue-70"
    )
    st.markdown("""
    ### 📑 Document Analysis 🌍
    1. **Upload**: Select a legal PDF document 📜
    2. **Language**: Choose document language 🌐
    3. **Analyze**: Click "Analyze Legal Risks" 🔍
    4. **Review**: Explore risks, visualizations, and summary 📈✨
    
    ### 💬 AI Assistant 🤖
    1. Analyze a legal document first 📑
    2. Select your speaking language 🌏
    3. Ask legal questions via text (press Enter) or voice (click 🎙️ and 🛑 to stop) 🗣️
    4. View and copy responses in the chat history 📋🌟
    
    **Sample Questions:**
    - What are the key risks? 🚨
    - Explain liability clauses 📝
    - Suggest mitigation strategies 🛡️
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #bbbbbb; font-size: 0.9em;">
        © 2025 Lexi-Lingua 🌟<br>
        Built with ❤️ using AI ✨
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #bbbbbb; margin-top: 2rem; padding: 1rem; border-top: 1px solid #3a3a3a; border-bottom: 2px dashed #4d94ff;">
    Powered by AI ⚡ | Lexi-Lingua v1.0 🚀🌟
</div>
""", unsafe_allow_html=True)