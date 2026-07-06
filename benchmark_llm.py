import time
import os
import numpy as np
import psycopg2
import requests
import re
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

PDF_FOLDER = "pdfs"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
PG_CONN = (
    f"host={os.environ.get('PG_HOST', 'localhost')} "
    f"port={os.environ.get('PG_PORT', '5432')} "
    f"dbname={os.environ.get('PG_DBNAME', 'postgres')} "
    f"user={os.environ.get('PG_USER', 'postgres')} "
    f"password={os.environ.get('PG_PASSWORD', 'postgres')}"
)
CHUNK_SIZE = 1300
OVERLAP = 260
TOP_K = 15

MODELOS = {
    "GPT-4": "openai/gpt-4o",
    "Gemini 2.5 Flash Lite": "google/gemini-2.5-flash-lite-preview-09-2025"
}

QUERIES = [
    "configuración de trabajador en régimen RIA",
    "carpetas personalizadas por usuario en edoc",
    "presupuesto post control en NISIRA ERP",
    "programación de pagos en NISIRA",
    "distribución de costos por orden de producción",
    "restitución de drawback exportación",
    "provisión de documentos recepción y conformidad",
    "asignación de básico a personal",
    "programación de fertilización y riego",
    "stock mínimo por sucursal y almacén"
]

def extraer_textos(folder):
    textos = []
    archivos = [f for f in os.listdir(folder) if f.endswith('.pdf') and 'artilla' in f]
    for filename in sorted(archivos):
        try:
            reader = PdfReader(os.path.join(folder, filename))
            texto = " ".join(page.extract_text() or "" for page in reader.pages)
            if len(texto.strip()) > 100:
                textos.append((filename, texto))
        except:
            pass
    return textos

def llamar_llm(modelo, prompt):
    try:
        t0 = time.time()
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": modelo,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0
            },
            timeout=60
        )
        latencia = (time.time() - t0) * 1000
        data = response.json()
        if "choices" not in data:
            return 0.5, latencia
        text = data["choices"][0]["message"]["content"].strip()
        numeros = re.findall(r'\d+\.?\d*', text)
        if numeros:
            valor = float(numeros[0])
            if valor > 1:
                valor = valor / 10
            return min(max(valor, 0.0), 1.0), latencia
        return 0.5, latencia
    except Exception as e:
        print(f"    ERROR: {e}")
        return 0.5, 0

def evaluar_modelo(nombre_modelo, modelo_id, chunks_por_query):
    faithfulness_scores = []
    ar_scores = []
    latencias = []

    for i, query in enumerate(QUERIES):
        contexto = "\n\n".join(chunks_por_query[i][:5])

        prompt_f = f"""¿Qué tan fiel es la siguiente respuesta a los documentos de contexto?
Responde SOLO con un número entre 0 y 1.

Contexto: {contexto[:2000]}
Pregunta: {query}
Respuesta: {contexto[:300]}

Puntuación fidelidad (0-1):"""

        prompt_ar = f"""¿Qué tan relevante es esta respuesta para la pregunta?
Responde SOLO con un número entre 0 y 1.

Pregunta: {query}
Respuesta basada en contexto: {contexto[:300]}

Puntuación relevancia (0-1):"""

        f_score, lat_f = llamar_llm(modelo_id, prompt_f)
        ar_score, lat_ar = llamar_llm(modelo_id, prompt_ar)

        faithfulness_scores.append(f_score)
        ar_scores.append(ar_score)
        latencias.append(lat_f + lat_ar)

        print(f"  [{nombre_modelo}] Query {i+1}: F={f_score:.3f}, AR={ar_score:.3f}, Lat={lat_f+lat_ar:.0f}ms")

    return {
        "faithfulness": round(np.mean(faithfulness_scores), 3),
        "answer_relevancy": round(np.mean(ar_scores), 3),
        "latencia_ms": round(np.mean(latencias), 0)
    }

if __name__ == "__main__":
    print("Cargando modelo de embeddings y documentos...")
    modelo_emb = SentenceTransformer("all-mpnet-base-v2")

    textos = extraer_textos(PDF_FOLDER)
    print(f"Documentos cargados: {len(textos)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    embeddings_data = []
    for filename, texto in textos:
        chunks = splitter.split_text(texto)
        embs = modelo_emb.encode(chunks, batch_size=32, show_progress_bar=False)
        for chunk, emb in zip(chunks, embs):
            embeddings_data.append((filename, chunk, emb))

    print(f"Chunks generados: {len(embeddings_data)}")

    conn = psycopg2.connect(PG_CONN)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS llm_bench;")
    cur.execute(f"CREATE TABLE llm_bench (id serial PRIMARY KEY, filename text, chunk_text text, vec vector(768));")
    cur.execute("CREATE INDEX ON llm_bench USING hnsw (vec vector_cosine_ops) WITH (m=32, ef_construction=128);")
    conn.commit()
    for filename, chunk, emb in embeddings_data:
        cur.execute("INSERT INTO llm_bench (filename, chunk_text, vec) VALUES (%s, %s, %s::vector)", (filename, chunk, emb.tolist()))
    conn.commit()
    cur.execute("SET hnsw.ef_search = 200;")

    chunks_por_query = []
    for query in QUERIES:
        q_emb = modelo_emb.encode([query])[0]
        cur.execute(f"SELECT chunk_text FROM llm_bench ORDER BY vec <=> %s::vector LIMIT {TOP_K};", (q_emb.tolist(),))
        chunks_por_query.append([r[0] for r in cur.fetchall()])
    cur.close()
    conn.close()

    print("\nEjecutando benchmark de modelos LLM...")
    resultados = {}
    for nombre, modelo_id in MODELOS.items():
        print(f"\n--- {nombre} ({modelo_id}) ---")
        resultados[nombre] = evaluar_modelo(nombre, modelo_id, chunks_por_query)

    print("\n\n=== RESULTADOS FINALES ===")
    print(f"{'Modelo':<20} {'Faithfulness':<15} {'Answer Relevancy':<18} {'Latencia gen (ms)'}")
    print("-" * 70)
    for nombre, res in resultados.items():
        print(f"{nombre:<20} {res['faithfulness']:<15} {res['answer_relevancy']:<18} {res['latencia_ms']}")