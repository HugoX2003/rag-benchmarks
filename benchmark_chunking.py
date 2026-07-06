import time
import os
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import requests

load_dotenv()

PDF_FOLDER = "pdfs"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"

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

CONFIGURACIONES = [
    (500, 100),
    (800, 160),
    (1000, 200),
    (1300, 260),
    (1500, 300),
]

def extraer_textos(folder):
    textos = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.pdf'):
            try:
                reader = PdfReader(os.path.join(folder, filename))
                texto = " ".join(page.extract_text() or "" for page in reader.pages)
                if len(texto.strip()) > 100:
                    textos.append(texto)
            except:
                pass
    return textos

def calcular_cr(query, chunks, top_k=5):
    contexto = "\n\n".join(chunks[:top_k])
    prompt = f"""Evalúa qué tan relevante es el siguiente contexto para responder la pregunta.
Responde SOLO con un número entre 0 y 1 (ejemplo: 0.75).

Pregunta: {query}

Contexto:
{contexto[:3000]}

Puntuación de relevancia (0-1):"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0
            },
            timeout=30
        )
        data = response.json()
        if "choices" not in data:
            print(f"    RESPUESTA API: {data}")
            return 0.5
        text = data["choices"][0]["message"]["content"].strip()
        import re
        numeros = re.findall(r'\d+\.?\d*', text)
        if numeros:
            valor = float(numeros[0])
            if valor > 1:
                valor = valor / 10
            return min(max(valor, 0.0), 1.0)
        return 0.5
    except Exception as e:
        print(f"    ERROR: {e}")
        return 0.5

def benchmark_chunking(chunk_size, overlap, textos, modelo):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    t0 = time.time()
    chunks = []
    for texto in textos:
        chunks.extend(splitter.split_text(texto))
    t_index = time.time() - t0

    n_chunks = len(chunks)
    embeddings = modelo.encode(chunks, batch_size=32, show_progress_bar=False)

    scores = []
    for query in QUERIES:
        q_emb = modelo.encode([query])[0]
        sims = np.dot(embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:5]
        top_chunks = [chunks[i] for i in top_idx]
        cr = calcular_cr(query, top_chunks)
        scores.append(cr)
        print(f"  Query: {query[:40]}... CR: {cr:.3f}")

    cr_promedio = np.mean(scores)
    return n_chunks, round(t_index, 2), round(cr_promedio, 3)

if __name__ == "__main__":
    print("Cargando modelo y textos...")
    modelo = SentenceTransformer("all-mpnet-base-v2")
    textos = extraer_textos(PDF_FOLDER)
    print(f"Documentos cargados: {len(textos)}")
    print(f"Ejemplo doc 1: {textos[0][:100]}")
    print(f"Ejemplo doc 50: {textos[49][:100]}")
    print(f"Ejemplo doc 100: {textos[99][:100]}")

    print(f"\n{'chunk_size':<12} {'overlap':<10} {'Chunks':<10} {'Indexación(s)':<15} {'CR promedio'}")
    print("-" * 60)

    for chunk_size, overlap in CONFIGURACIONES:
        print(f"\nBenchmarking chunk_size={chunk_size}, overlap={overlap}...")
        n_chunks, t_idx, cr = benchmark_chunking(chunk_size, overlap, textos, modelo)
        print(f"{chunk_size:<12} {overlap:<10} {n_chunks:<10} {t_idx:<15} {cr}")