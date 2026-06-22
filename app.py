%%writefile app.py
import streamlit as st
import os, re
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import google.generativeai as genai
from groq import Groq # Import Groq client

# --- Configuration --- General LLM setup
# You need to set GEMINI_API_KEY and GROQ_API_KEY in Streamlit secrets
# Go to your Streamlit app -> left sidebar -> Settings -> App secrets
# Add GEMINI_API_KEY = "YOUR_API_KEY"
# Add GROQ_API_KEY = "YOUR_API_KEY"

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

        # Remove markdown code block fences if present
        if sql.startswith('```sql') and sql.endswith('```'):
            sql = sql[6:-3].strip()
        elif sql.startswith('```') and sql.endswith('```'): # generic markdown code block
            sql = sql[3:-3].strip()
        return sql
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "type": "error", "content": f"Error generating SQL: {e}"})
        return ""

def validate_sql(sql: str) -> bool:
    if not sql:
        return False
    cleaned_sql = sql.strip().lower()
    if not cleaned_sql.startswith("select"):
        return False
    for keyword in FORBIDDEN:
        if keyword in cleaned_sql:
            return False
    if ';' in cleaned_sql[:-1]:
        return False
    return True

def run_sql(sql: str) -> pd.DataFrame:
    with db_engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

def visualize_results(df: pd.DataFrame):
    if df.empty:
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": "Tidak ada data untuk divisualisasikan."})
        return

    # Check for bar chart: 2 columns, second is numeric
    if len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
        st.session_state.messages.append({
            "role": "assistant",
            "type": "chart",
            "chart_type": "bar",
            "data": df.to_dict('records'), # Store data as dicts for serialization
            "x_col": df.columns[0],
            "y_col": df.columns[1]
        })
    # Check for line chart: first is datetime, second is numeric
    elif len(df.columns) >= 2 and pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]) and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
        st.session_state.messages.append({
            "role": "assistant",
            "type": "chart",
            "chart_type": "line",
            "data": df.to_dict('records'),
            "x_col": df.columns[0],
            "y_col": df.columns[1]
        })
    else:
        st.session_state.messages.append({"role": "assistant", "type": "dataframe", "content": df.to_dict('records')})

def ask_pipeline(question: str):
    st.session_state.messages.append({"role": "user", "type": "text", "content": question})

    sql_raw = generate_sql(question)
    if not sql_raw:
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": "Gagal menghasilkan SQL dari pertanyaan Anda."})
        return
    st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"SQL (raw): {sql_raw}"})

    sql_valid = sql_raw
    if not validate_sql(sql_valid):
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": "SQL tidak valid, mencoba generate ulang..."})
        sql_valid = generate_sql(question)
        if not sql_valid:
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": "Gagal menghasilkan SQL yang valid setelah coba ulang."})
            return
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"SQL (retry): {sql_valid}"})
        if not validate_sql(sql_valid):
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": "Gagal menghasilkan SQL yang valid setelah coba ulang. Mohon perbaiki pertanyaan atau prompt."})
            return

    st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"SQL (valid): {sql_valid}"})

    try:
        df_result = run_sql(sql_valid)
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": "Query berhasil dijalankan."})
        visualize_results(df_result)
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "type": "error", "content": f"Error saat menjalankan query: {e}. Gagal menjalankan query. Mohon perbaiki pertanyaan atau prompt."})
        return

# --- Streamlit UI ---
st.title("Mini Project — Conversational Analytics (Text-to-SQL)")
st.caption("Human Capital Analytics")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.write(message["content"])
        elif message["type"] == "dataframe":
            df = pd.DataFrame(message["content"])
            st.dataframe(df)
        elif message["type"] == "chart":
            df = pd.DataFrame(message["data"])
            if message["chart_type"] == "bar":
                st.bar_chart(df.set_index(message["x_col"]))
            elif message["chart_type"] == "line":
                st.line_chart(df.set_index(message["x_col"]))
        elif message["type"] == "error":
            st.error(message["content"])


if prompt := st.chat_input("Tanyakan sesuatu tentang data Anda..."):
    ask_pipeline(prompt)
