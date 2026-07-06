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
MODEL_LLM = "google/gemini-2.5-flash-lite-preview-09-2025"
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
    archivos = [f for f in os.listdir(folder) if f.endswith('.pdf') and ('artilla' in f)]
    for filename in sorted(archivos):
        try:
            reader = PdfReader(os.path.join(folder, filename))
            texto = " ".join(page.extract_text() or "" for page in reader.pages)
            if len(texto.strip()) > 100:
                textos.append((filename, texto))
        except:
            pass
    return textos

def evaluar_llm(query, contexto, metrica):
    if metrica == "faithfulness":
        prompt = f"""¿Qué tan fiel es la siguiente respuesta a los documentos de contexto?
Responde SOLO con un número entre 0 y 1.

Contexto: {contexto[:2000]}
Pregunta: {query}
Responde: {contexto[:500]}

Puntuación (0-1):"""
    else:
        prompt = f"""¿Qué tan relevante es la siguiente respuesta para la pregunta?
Responde SOLO con un número entre 0 y 1.

Pregunta: {query}
Respuesta basada en contexto: {contexto[:500]}

Puntuación (0-1):"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_LLM,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0
            },
            timeout=30
        )
        data = response.json()
        if "choices" not in data:
            return 0.5
        text = data["choices"][0]["message"]["content"].strip()
        numeros = re.findall(r'\d+\.?\d*', text)
        if numeros:
            valor = float(numeros[0])
            if valor > 1:
                valor = valor / 10
            return min(max(valor, 0.0), 1.0)
        return 0.5
    except:
        return 0.5

def setup_db(conn, embeddings_data):
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS retrieval_bench;")
    cur.execute("""
        CREATE TABLE retrieval_bench (
            id serial PRIMARY KEY,
            filename text,
            chunk_text text,
            vec vector(768)
        );
    """)
    cur.execute("""
        CREATE INDEX ON retrieval_bench 
        USING hnsw (vec vector_cosine_ops) 
        WITH (m=32, ef_construction=128);
    """)
    conn.commit()

    for filename, chunk, emb in embeddings_data:
        cur.execute(
            "INSERT INTO retrieval_bench (filename, chunk_text, vec) VALUES (%s, %s, %s::vector)",
            (filename, chunk, emb.tolist())
        )
    conn.commit()
    cur.execute("SET hnsw.ef_search = 200;")
    cur.close()

def retrieval_semantico(conn, q_emb, top_k):
    cur = conn.cursor()
    cur.execute(
        f"SELECT chunk_text FROM retrieval_bench ORDER BY vec <=> %s::vector LIMIT {top_k};",
        (q_emb.tolist(),)
    )
    chunks = [r[0] for r in cur.fetchall()]
    cur.close()
    return chunks

def retrieval_hibrido(conn, q_emb, query_text, top_k, alpha=0.6):
    cur = conn.cursor()
    # Semántico
    cur.execute(
        f"SELECT chunk_text, 1 - (vec <=> %s::vector) as sim FROM retrieval_bench ORDER BY vec <=> %s::vector LIMIT {top_k*2};",
        (q_emb.tolist(), q_emb.tolist())
    )
    sem_results = {r[0]: r[1] for r in cur.fetchall()}

    # Lexical BM25-like con ts_rank
    cur.execute(
        f"""SELECT chunk_text, ts_rank(to_tsvector('spanish', chunk_text), plainto_tsquery('spanish', %s)) as rank
        FROM retrieval_bench
        WHERE to_tsvector('spanish', chunk_text) @@ plainto_tsquery('spanish', %s)
        ORDER BY rank DESC LIMIT {top_k*2};""",
        (query_text, query_text)
    )
    lex_results = {r[0]: r[1] for r in cur.fetchall()}
    cur.close()

    # Normalizar y combinar
    max_sem = max(sem_results.values()) if sem_results else 1
    max_lex = max(lex_results.values()) if lex_results else 1

    scores = {}
    for chunk, score in sem_results.items():
        scores[chunk] = alpha * (score / max_sem)
    for chunk, score in lex_results.items():
        if chunk in scores:
            scores[chunk] += (1 - alpha) * (score / max_lex)
        else:
            scores[chunk] = (1 - alpha) * (score / max_lex)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:top_k]]

if __name__ == "__main__":
    print("Cargando modelo y documentos...")
    modelo = SentenceTransformer("all-mpnet-base-v2")

    textos = extraer_textos(PDF_FOLDER)
    print(f"Documentos cargados: {len(textos)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    embeddings_data = []
    for filename, texto in textos:
        chunks = splitter.split_text(texto)
        embs = modelo.encode(chunks, batch_size=32, show_progress_bar=False)
        for chunk, emb in zip(chunks, embs):
            embeddings_data.append((filename, chunk, emb))

    print(f"Chunks generados: {len(embeddings_data)}")

    conn = psycopg2.connect(PG_CONN)
    print("Insertando en pgvector...")
    setup_db(conn, embeddings_data)
    print("Listo. Ejecutando benchmark...")

    resultados = {"semantico": {"faithfulness": [], "answer_relevancy": [], "latencias": []},
                  "hibrido": {"faithfulness": [], "answer_relevancy": [], "latencias": []}}

    for query in QUERIES:
        q_emb = modelo.encode([query])[0]

        t0 = time.time()
        chunks_sem = retrieval_semantico(conn, q_emb, TOP_K)
        lat_sem = (time.time() - t0) * 1000
        contexto_sem = "\n\n".join(chunks_sem)
        f_sem = evaluar_llm(query, contexto_sem, "faithfulness")
        ar_sem = evaluar_llm(query, contexto_sem, "answer_relevancy")
        resultados["semantico"]["faithfulness"].append(f_sem)
        resultados["semantico"]["answer_relevancy"].append(ar_sem)
        resultados["semantico"]["latencias"].append(lat_sem)

        t0 = time.time()
        chunks_hyb = retrieval_hibrido(conn, q_emb, query, TOP_K)
        lat_hyb = (time.time() - t0) * 1000
        contexto_hyb = "\n\n".join(chunks_hyb)
        f_hyb = evaluar_llm(query, contexto_hyb, "faithfulness")
        ar_hyb = evaluar_llm(query, contexto_hyb, "answer_relevancy")
        resultados["hibrido"]["faithfulness"].append(f_hyb)
        resultados["hibrido"]["answer_relevancy"].append(ar_hyb)
        resultados["hibrido"]["latencias"].append(lat_hyb)

        print(f"Query: {query[:40]}...")
        print(f"  Semántico  - F: {f_sem:.3f}, AR: {ar_sem:.3f}, Lat: {lat_sem:.2f}ms")
        print(f"  Híbrido    - F: {f_hyb:.3f}, AR: {ar_hyb:.3f}, Lat: {lat_hyb:.2f}ms")

    conn.close()

    print("\n=== RESULTADOS FINALES ===")
    print(f"{'Estrategia':<25} {'Faithfulness':<15} {'Answer Relevancy':<18} {'Latencia (ms)':<15} {'Chunks'}")
    print("-" * 80)
    print(f"{'Solo semántico':<25} {np.mean(resultados['semantico']['faithfulness']):<15.3f} {np.mean(resultados['semantico']['answer_relevancy']):<18.3f} {np.mean(resultados['semantico']['latencias']):<15.2f} {TOP_K}")
    print(f"{'Híbrido 60/40':<25} {np.mean(resultados['hibrido']['faithfulness']):<15.3f} {np.mean(resultados['hibrido']['answer_relevancy']):<18.3f} {np.mean(resultados['hibrido']['latencias']):<15.2f} {TOP_K}")