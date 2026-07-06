import time
import os
import gc
import psutil
import numpy as np
import multiprocessing
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_FOLDER = "pdfs"
N_QUERIES = 10
N_RUNS = 10
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def extraer_textos(folder):
    textos = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            try:
                reader = PdfReader(os.path.join(folder, filename))
                texto = " ".join(page.extract_text() or "" for page in reader.pages)
                if len(texto.strip()) > 100:
                    textos.append(texto[:2000])
            except:
                pass
    return textos


def _cache_dir_name(nombre_modelo):
    if "/" in nombre_modelo:
        return "models--" + nombre_modelo.replace("/", "--")
    return "models--sentence-transformers--" + nombre_modelo


def verificar_cache(nombre_modelo):
    return (HF_CACHE_DIR / _cache_dir_name(nombre_modelo)).exists()


def _worker_benchmark(nombre_modelo, textos, queries, n_runs, cola):
    gc.collect()
    proceso = psutil.Process(os.getpid())
    ram_antes = proceso.memory_info().rss / 1024 / 1024

    t_carga_inicio = time.time()
    modelo = SentenceTransformer(nombre_modelo)
    t_carga = time.time() - t_carga_inicio

    ram_despues = proceso.memory_info().rss / 1024 / 1024
    ram_usada = ram_despues - ram_antes

    t_index_inicio = time.time()
    modelo.encode(textos, batch_size=32, show_progress_bar=False)
    t_index = time.time() - t_index_inicio

    latencias = []
    for _ in range(n_runs):
        for q in queries:
            t0 = time.time()
            modelo.encode([q])
            latencias.append((time.time() - t0) * 1000)

    cola.put({
        "t_carga": round(t_carga, 2),
        "t_index": round(t_index, 2),
        "latencia_ms": round(np.mean(latencias), 2),
        "ram_mb": round(ram_usada, 2),
    })


def benchmark_modelo(nombre_modelo, textos, queries):
    print(f"\n{'='*52}")
    print(f"Modelo: {nombre_modelo}")
    print(f"{'='*52}")

    en_cache = verificar_cache(nombre_modelo)
    print(f"Cache local: {'SI' if en_cache else 'NO'}")

    cola = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_worker_benchmark,
        args=(nombre_modelo, textos, queries, N_RUNS, cola)
    )
    p.start()
    p.join()

    res = cola.get()
    res["modelo"] = nombre_modelo
    res["cache"] = "SI" if en_cache else "NO"

    print(f"Tiempo de carga:            {res['t_carga']:.2f} s")
    print(f"Tiempo de indexacion:       {res['t_index']:.2f} s")
    print(f"Latencia promedio de query: {res['latencia_ms']:.2f} ms")
    print(f"RAM utilizada:              {res['ram_mb']:.2f} MB")

    return res


if __name__ == "__main__":
    print("Extrayendo textos de PDFs...")
    textos = extraer_textos(PDF_FOLDER)
    print(f"Documentos procesados: {len(textos)}\n")

    queries = [
        "procedimiento de control de documentos",
        "gestión de acceso y permisos de usuario",
        "auditoría interna ISO 9001",
        "política de seguridad de la información",
        "manual de procedimientos operativos",
        "control de versiones documentales",
        "registro de no conformidades",
        "capacitación del personal",
        "revisión por la dirección",
        "indicadores de desempeño organizacional"
    ]

    resultados = []
    resultados.append(benchmark_modelo("all-mpnet-base-v2", textos, queries))
    resultados.append(benchmark_modelo("BAAI/bge-m3", textos, queries))

    ancho = 90
    print("\n")
    print("=" * ancho)
    print(f"{'BENCHMARK - MODELOS DE EMBEDDING':^{ancho}}")
    print(f"{'Corpus: PDFs del sistema NISIRA ERP | Queries: ' + str(len(queries)):^{ancho}}")
    print("=" * ancho)
    print(f"{'Modelo':<24} {'Cache':>6} {'Carga (s)':>10} {'Indexacion (s)':>15} {'Latencia (ms)':>14} {'RAM (MB)':>10}")
    print("-" * ancho)
    for r in resultados:
        print(f"{r['modelo']:<24} {r['cache']:>6} {r['t_carga']:>10.2f} {r['t_index']:>15.2f} {r['latencia_ms']:>14.2f} {r['ram_mb']:>10.2f}")
    print("=" * ancho)
    print(f"  RAM: medida en subproceso aislado (proceso limpio por modelo)")
    print(f"  Cache: indica si los pesos del modelo estaban en ~/.cache/huggingface")
    print(f"  Latencia: promedio de {N_RUNS} ejecuciones x {len(queries)} queries")
    print(f"  Indexacion: tiempo de encode sobre {len(textos)} documentos")
    print("=" * ancho)
