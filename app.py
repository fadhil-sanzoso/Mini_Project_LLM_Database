import streamlit as st
import os, re
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import google.generativeai as genai

# --- Configuration ---
# You need to set GEMINI_API_KEY in Streamlit secrets
# Go to your Streamlit app -> left sidebar -> Settings -> App secrets
# Add GEMINI_API_KEY = "YOUR_API_KEY"
if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
    st.stop()

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
MODEL_NAME = "gemini-flash-latest"

genai.configure(api_key=GEMINI_API_KEY)

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
def get_gemini_model():
    return genai.GenerativeModel(MODEL_NAME)

# NOTE: For a real Streamlit app, connecting to a PostgreSQL instance
# would require it to be accessible from where the app is hosted (e.g., public IP, cloud SQL proxy).
# For this Colab context and writing app.py, we assume a PostgreSQL instance
# is somehow accessible. If this app were to be deployed standalone,
# the database setup would need to be re-evaluated (e.g., SQLite for simplicity, or external PG).
@st.cache_resource
def get_db_engine():
    return create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/miniproject")

model_llm = get_gemini_model()
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
        resp = model_llm.generate_content(prompt)
        sql = resp.text.strip()
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
