# RAG Benchmarks — NISIRA ERP

Scripts de evaluación y análisis estadístico desarrollados como parte de la tesis para medir el desempeño de un sistema RAG (*Retrieval-Augmented Generation*) aplicado a documentación del sistema NISIRA ERP.

---

## Requisitos

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copia `.env.example` a `.env` y completa las variables:

```bash
cp .env.example .env
```

Para los benchmarks que usan pgvector, levanta el contenedor:

```bash
docker run -d --name pgvector-bench \
  -e POSTGRES_PASSWORD=postgres \
  -p 5433:5432 pgvector/pgvector:pg16

docker exec -it pgvector-bench psql -U postgres \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

## Archivos

### `benchmark_embeddings.py`
Compara dos modelos de embeddings: `all-mpnet-base-v2` y `BAAI/bge-m3`.

Métricas medidas:
- Tiempo de carga del modelo
- Tiempo de indexación del corpus
- Latencia promedio de consulta (10 runs × 10 queries)
- RAM utilizada (medida en subproceso aislado por modelo)

Incluye detección automática de caché local de HuggingFace para identificar si el tiempo de carga corresponde a descarga o lectura desde disco.

```bash
python benchmark_embeddings.py
```

---

### `benchmark_vectordb.py`
Compara tres backends de búsqueda vectorial a diferentes escalas (100, 1 000 y 10 000 vectores de 768 dimensiones).

Backends evaluados:
- **FAISS Flat** — búsqueda exacta, referencia de ground truth
- **ChromaDB HNSW** — base de datos vectorial embebida
- **pgvector HNSW** — extensión vectorial sobre PostgreSQL (m=32, ef\_construction=128, ef\_search=200)

Métricas: tiempo de indexación, latencia de consulta (30 runs por query) y Recall@10.

Requiere Docker con pgvector corriendo.

```bash
python benchmark_vectordb.py
```

---

### `benchmark_chunking.py`
Evalúa cinco configuraciones de chunking con solapamiento del 20%:

| chunk\_size | overlap |
|------------|---------|
| 500        | 100     |
| 800        | 160     |
| 1 000      | 200     |
| 1 300      | 260     |
| 1 500      | 300     |

Para cada configuración mide el número de chunks generados, tiempo de indexación y Context Relevance (CR) evaluada con Gemini 2.5 Flash Lite como juez LLM vía OpenRouter.

```bash
python benchmark_chunking.py
```

---

### `benchmark_retrieval.py`
Compara dos estrategias de recuperación sobre las cartillas NISIRA (chunk\_size=1300, overlap=260, Top-K=15):

- **Solo semántico** — similitud coseno con embeddings `all-mpnet-base-v2`
- **Híbrido 60/40** — combinación de similitud coseno (60%) y BM25 lexical vía `ts_rank` de PostgreSQL (40%)

Métricas: Faithfulness, Answer Relevancy y latencia de recuperación, evaluadas con Gemini 2.5 Flash Lite como juez LLM vía OpenRouter.

Requiere Docker con pgvector corriendo.

```bash
python benchmark_retrieval.py
```

---

### `benchmark_llm.py`
Compara modelos LLM como evaluadores (*LLM-as-a-judge*) sobre los mismos chunks recuperados:

- **GPT-4o** (`openai/gpt-4o`)
- **Gemini 2.5 Flash Lite** (`google/gemini-2.5-flash-lite-preview-09-2025`)

Por cada query se realizan dos llamadas al LLM (faithfulness + answer relevancy) y se reporta la latencia promedio de generación.

Requiere Docker con pgvector corriendo.

```bash
python benchmark_llm.py
```

---

### `shapiro_wilk_oe4.py`
Prueba de normalidad Shapiro-Wilk sobre los valores de Faithfulness en dominio (N=14, Tabla 14 de la tesis), como verificación del supuesto previo a la prueba t de Student.

```bash
python shapiro_wilk_oe4.py
```

---

### `docimasia_oe4.py`
Docimasia de hipótesis del Objetivo Específico 4. Aplica una prueba t de una muestra, unilateral derecha, sobre los valores de Faithfulness en dominio para contrastar:

- **H0:** El sistema RAG no produce respuestas fieles al contexto (Faithfulness = 0)
- **H1:** El sistema RAG produce respuestas fieles al contexto (Faithfulness > 0)

Genera un gráfico de la distribución t con la zona de rechazo, guardado como `figura19_zona_rechazo.png`.

```bash
python docimasia_oe4.py
```

---

## Variables de entorno

| Variable | Descripción |
|---|---|
| `OPENROUTER_API_KEY` | Clave de API de OpenRouter (requerida por chunking, retrieval y llm) |
| `PG_HOST` | Host de PostgreSQL (por defecto: `localhost`) |
| `PG_PORT` | Puerto de PostgreSQL (por defecto: `5432`) |
| `PG_DBNAME` | Nombre de la base de datos (por defecto: `postgres`) |
| `PG_USER` | Usuario de PostgreSQL (por defecto: `postgres`) |
| `PG_PASSWORD` | Contraseña de PostgreSQL |
