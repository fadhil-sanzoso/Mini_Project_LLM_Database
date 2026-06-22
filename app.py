import streamlit as st
import os, re
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import google.generativeai as genai
from groq import Groq # Import Groq client

# --- Configuration --- General LLM setup
# You need to set GEMINI_API_KEY, GROQ_API_KEY, and SUPABASE_DB_URL in Streamlit secrets
# Go to your Streamlit app -> left sidebar -> Settings -> App secrets
# Add GEMINI_API_KEY = "YOUR_API_KEY"
# Add GROQ_API_KEY = "YOUR_API_KEY"
# Add SUPABASE_DB_URL = "postgresql://postgres.xjrdfhzdiukojrvdkagc:[YOUR-PASSWORD]@aws-1-ap-southeast-2.pooler.supabase.com:6543/postgres"

# --- LLM Provider Selection ---
# Choose your desired LLM provider here. This can be made a Streamlit input later if needed.
LLM_PROVIDER = "GROQ" # Set to "GEMINI" or "GROQ"

GEMINI_API_KEY = None
GROQ_API_KEY = None

if LLM_PROVIDER == "GEMINI":
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
        st.stop()
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    MODEL_NAME_GEMINI = "gemini-2.5-flash-lite"  # Or your preferred Gemini model
    genai.configure(api_key=GEMINI_API_KEY)
elif LLM_PROVIDER == "GROQ":
    if "GROQ_API_KEY" not in st.secrets:
        st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it.")
        st.stop()
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GROQ_MODEL_NAME = "llama-3.3-70b-versatile" # Or your preferred Groq model
else:
    st.error("Invalid LLM_PROVIDER. Choose 'GEMINI' or 'GROQ'.")
    st.stop()

# --- Database Schema and Forbidden Keywords ---
SCHEMA_STR = """employees(nip, nama, divisi, jabatan, join_date)
trainings(training_id, nama_diklat, tanggal)
enrollments(nip, training_id, status, nilai)

Relasi:
- enrollments.nip      -> employees.nip
- enrollments.training_id -> trainings.training_id
Catatan: enrollments.nilai bisa kosong (NULL) jika status = 'berjalan'."""

FORBIDDEN = ["drop", "delete", "update", "insert", "alter", "truncate", "create", "grant"]

# --- Initialize LLM and Database Engine (cached) ---
@st.cache_resource
def get_llm_model():
    if LLM_PROVIDER == "GEMINI":
        return genai.GenerativeModel(MODEL_NAME_GEMINI)
    elif LLM_PROVIDER == "GROQ":
        return Groq(api_key=GROQ_API_KEY)
    return None # Should not happen due to previous checks

@st.cache_resource
def get_db_engine():
    if "SUPABASE_DB_URL" not in st.secrets:
        st.error("SUPABASE_DB_URL not found in Streamlit secrets. Please add it.")
        st.stop()
        
    db_url = st.secrets["SUPABASE_DB_URL"]
    
    # SQLAlchemy requires the psycopg2 driver specified in the dialect
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        
    return create_engine(db_url)

llm_model_or_client = get_llm_model()
db_engine = get_db_engine()

# --- Functions from Notebook (adapted for Streamlit output) ---

def build_prompt(question: str) -> str:
    prompt = f"""
Anda adalah pakar PostgreSQL. Berdasarkan skema database berikut, hasilkan HANYA SATU query PostgreSQL SELECT yang menjawab pertanyaan pengguna. JANGAN berikan penjelasan atau teks tambahan apa pun.

Skema Database:
{SCHEMA_STR}

Beberapa Contoh:
Pertanyaan: Berapa jumlah pegawai per divisi?
SQL: SELECT divisi, COUNT(nip) FROM employees GROUP BY divisi;
"""
    return prompt

def generate_sql(question: str) -> str:
    try:
        prompt = build_prompt(question)

        generation_config = {
            "temperature": 0, # Controls randomness. Lower values for more deterministic output.
            "top_k": 40,        # Considers the top_k most likely tokens.
            "top_p": 0.95       # Considers tokens whose probability sums to top_p.
        }

        sql = ""
        if LLM_PROVIDER == "GEMINI":
            if llm_model_or_client is None:
                raise RuntimeError("Gemini model not initialized.")
            resp = llm_model_or_client.generate_content(prompt, generation_config=generation_config)
            sql = resp.text.strip()
        elif LLM_PROVIDER == "GROQ":
            if llm_model_or_client is None:
                raise RuntimeError("Groq client not initialized.")
            response = llm_model_or_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=GROQ_MODEL_NAME,
                temperature=generation_config["temperature"],
                max_tokens=200,
                top_p=generation_config["top_p"],
                stop=None,
                stream=False,
            )
            sql = response.choices[0].message.content.strip()

         # Remove markdown code block fences if present safely
        md_ticks = "`" * 3
        if sql.startswith(md_ticks + "sql") and sql.endswith(md_ticks):
            sql = sql[6:-3].strip()
        elif sql.startswith(md_ticks) and sql.endswith(md_ticks):
            sql = sql[3:-3].strip()
        return sql

    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "type": "error", "content": f"Error generating SQL: {e}"})
        return ""

