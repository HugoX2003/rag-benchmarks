import time
import os
import numpy as np
import psycopg2
import faiss
import chromadb
from dotenv import load_dotenv

load_dotenv()

DIMS = 768
SEED = 42
N_QUERY_RUNS = 30
PG_CONN = (
    f"host={os.environ.get('PG_HOST', 'localhost')} "
    f"port={os.environ.get('PG_PORT', '5432')} "
    f"dbname={os.environ.get('PG_DBNAME', 'postgres')} "
    f"user={os.environ.get('PG_USER', 'postgres')} "
    f"password={os.environ.get('PG_PASSWORD', 'postgres')}"
)

np.random.seed(SEED)

def generar_vectores(n):
    vecs = np.random.randn(n, DIMS).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def recall_at_k(true_ids, retrieved_ids, k=10):
    hits = len(set(true_ids[:k]) & set(retrieved_ids[:k]))
    return hits / k

def benchmark_faiss(n, vectores, queries, ground_truth):
    index = faiss.IndexFlatL2(DIMS)
    t0 = time.time()
    index.add(vectores)
    t_index = time.time() - t0

    latencias = []
    resultados = []
    for q in queries:
        tiempos = []
        for _ in range(N_QUERY_RUNS):
            t0 = time.time()
            _, I = index.search(q.reshape(1, -1), 10)
            tiempos.append((time.time() - t0) * 1000)
        latencias.append(np.mean(tiempos))
        resultados.append(I[0].tolist())

    recalls = [recall_at_k(ground_truth[i], resultados[i]) for i in range(len(queries))]
    return round(t_index, 4), round(np.mean(latencias), 3), round(np.mean(recalls), 3)

def benchmark_chroma(n, vectores, queries, ground_truth):
    client = chromadb.Client()
    try:
        client.delete_collection("bench")
    except:
        pass
    col = client.create_collection("bench", metadata={"hnsw:space": "cosine"})

    t0 = time.time()
    batch = 500
    for i in range(0, n, batch):
        batch_vecs = vectores[i:i+batch]
        col.add(
            embeddings=batch_vecs.tolist(),
            ids=[str(j) for j in range(i, min(i+batch, n))]
        )
    t_index = time.time() - t0

    latencias = []
    resultados = []
    for q in queries:
        tiempos = []
        for _ in range(N_QUERY_RUNS):
            t0 = time.time()
            res = col.query(query_embeddings=[q.tolist()], n_results=10)
            tiempos.append((time.time() - t0) * 1000)
        latencias.append(np.mean(tiempos))
        resultados.append([int(x) for x in res['ids'][0]])

    recalls = [recall_at_k(ground_truth[i], resultados[i]) for i in range(len(queries))]
    return round(t_index, 4), round(np.mean(latencias), 3), round(np.mean(recalls), 3)

def benchmark_pgvector(n, vectores, queries, ground_truth):
    conn = psycopg2.connect(PG_CONN)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS bench_vecs;")
    cur.execute(f"CREATE TABLE bench_vecs (id serial, vec vector({DIMS}));")
    cur.execute("CREATE INDEX ON bench_vecs USING hnsw (vec vector_cosine_ops) WITH (m=32, ef_construction=128);")
    conn.commit()

    t0 = time.time()
    batch = 500
    for i in range(0, n, batch):
        batch_vecs = vectores[i:i+batch]
        args = [(vec.tolist(),) for vec in batch_vecs]
        cur.executemany("INSERT INTO bench_vecs (vec) VALUES (%s::vector)", args)
    conn.commit()
    t_index = time.time() - t0

    cur.execute(f"SET hnsw.ef_search = 200;")

    latencias = []
    resultados = []
    for q in queries:
        tiempos = []
        for _ in range(N_QUERY_RUNS):
            t0 = time.time()
            cur.execute(
                "SELECT id FROM bench_vecs ORDER BY vec <=> %s::vector LIMIT 10;",
                (q.tolist(),)
            )
            rows = cur.fetchall()
            tiempos.append((time.time() - t0) * 1000)
        latencias.append(np.mean(tiempos))
        resultados.append([r[0]-1 for r in rows])

    recalls = [recall_at_k(ground_truth[i], resultados[i]) for i in range(len(queries))]
    cur.execute("DROP TABLE IF EXISTS bench_vecs;")
    conn.commit()
    cur.close()
    conn.close()
    return round(t_index, 4), round(np.mean(latencias), 3), round(np.mean(recalls), 3)

if __name__ == "__main__":
    escalas = [100, 1000, 10000]
    n_queries = 10
    ancho = 78

    print("=" * ancho)
    print(f"{'BENCHMARK - BASES DE DATOS VECTORIALES':^{ancho}}")
    print(f"{'Dimensiones: ' + str(DIMS) + ' | Queries por escala: ' + str(n_queries) + ' | Runs por query: ' + str(N_QUERY_RUNS):^{ancho}}")
    print("=" * ancho)
    print(f"{'Backend':<20} {'N':>7} {'Indexacion (s)':>15} {'Latencia (ms)':>14} {'Recall@10':>12}")
    print("-" * ancho)

    for n in escalas:
        vectores = generar_vectores(n)
        queries = generar_vectores(n_queries)

        index_exact = faiss.IndexFlatL2(DIMS)
        index_exact.add(vectores)
        _, gt = index_exact.search(queries, 10)
        ground_truth = gt.tolist()

        print(f"  -- N = {n:,} vectores --")

        t_idx, lat, rec = benchmark_faiss(n, vectores, queries, ground_truth)
        print(f"{'FAISS Flat':<20} {n:>7,} {t_idx:>15.4f} {lat:>14.3f} {rec:>12.3f}")

        t_idx, lat, rec = benchmark_chroma(n, vectores, queries, ground_truth)
        print(f"{'ChromaDB HNSW':<20} {n:>7,} {t_idx:>15.4f} {lat:>14.3f} {rec:>12.3f}")

        t_idx, lat, rec = benchmark_pgvector(n, vectores, queries, ground_truth)
        print(f"{'pgvector HNSW':<20} {n:>7,} {t_idx:>15.4f} {lat:>14.3f} {rec:>12.3f}")

        if n != escalas[-1]:
            print()

    print("=" * ancho)
    print(f"  Ground truth: busqueda exacta con FAISS IndexFlatL2")
    print(f"  pgvector: indice HNSW (m=32, ef_construction=128, ef_search=200)")
    print(f"  ChromaDB: indice HNSW con distancia coseno")
    print("=" * ancho)