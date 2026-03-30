# EmbeddingService - Composant 4

## Overview

L'`EmbeddingService` génère des vecteurs d'embedding (représentations vectorielles) pour chaque chunk de texte. Ces vecteurs capturent la sémantique du texte et permettent la recherche par similarité cosinus.

## Architecture

### Modèle d'embedding

- **Par défaut:** `BAAI/bge-small-en-v1.5` (fastembed)
  - 384 dimensions
  - ONNX-based (CPU-optimized, pas GPU requis)
  - Téléchargement automatique au premier usage
  - Modèle léger et très rapide

### Processus d'embedding

```
TextChunk
    ↓
Extraction du texte
    ↓
fastembed TextEmbedding.embed()
    ↓
Numpy array (384-dim)
    ↓
Conversion en liste (JSON-sérialisable)
    ↓
EmbeddedChunk (chunk + embedding + métadonnées)
```

## Classes principales

### `EmbeddedChunk`

Représente un chunk avec son vecteur d'embedding.

```python
class EmbeddedChunk(BaseModel):
    chunk: TextChunk  # Chunk original avec métadonnées
    embedding: List[float]  # Vecteur d'embedding (384 floats)
    embedding_dim: int  # Dimension du vecteur
    embedding_model: str  # Modèle utilisé (ex: "BAAI/bge-small-en-v1.5")
```

### `EmbeddingResult`

Résultat du processus d'embedding.

```python
class EmbeddingResult(BaseModel):
    document_path: str  # Chemin du document source
    embedded_chunks: List[EmbeddedChunk]  # Chunks avec embeddings
    total_chunks: int  # Nombre de chunks embedded
    embedding_model: str  # Modèle utilisé
    embedding_dim: int  # Dimension des embeddings
    total_time_seconds: float  # Temps d'exécution
    embedded_at: datetime  # Timestamp du processus
```

### `EmbeddingService`

Service principal pour générer les embeddings.

```python
class EmbeddingService:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        logger_instance: Optional[logging.Logger] = None,
    ):
        pass
```

#### Paramètres d'initialisation

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `model_name` | str | "BAAI/bge-small-en-v1.5" | Modèle fastembed à utiliser |
| `batch_size` | int | 32 | Taille des batches pour processing |
| `logger_instance` | Logger | None | Logger custom (optional) |

## Méthodes principales

### `embed_chunks()`

Génère des embeddings pour une liste de chunks.

```python
def embed_chunks(self, chunks: List[TextChunk]) -> EmbeddingResult:
    """
    Génère des embeddings pour une liste de chunks.

    Args:
        chunks: List[TextChunk] du ChunkingService

    Returns:
        EmbeddingResult avec chunks embedded

    Raises:
        ValueError: Si la liste est vide
        Exception: Si le modèle échoue
    """
```

**Exemple:**

```python
from ragindexer import ChunkingService, EmbeddingService

# Créer les services
chunking_service = ChunkingService(chunk_size=512, overlap_size=50)
embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")

# Chunk le document
chunking_result = chunking_service.chunk(parsed_document)

# Embed les chunks
embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

# Accéder aux embeddings
for embedded_chunk in embedding_result.embedded_chunks:
    print(f"Chunk: {embedded_chunk.chunk.content[:50]}...")
    print(f"Embedding dim: {embedded_chunk.embedding_dim}")
    print(f"Vector: {embedded_chunk.embedding[:5]}...")  # Premiers 5 values
```

### `embed_single_chunk()`

Génère un embedding pour un seul chunk.

```python
def embed_single_chunk(self, chunk: TextChunk) -> EmbeddedChunk:
    """
    Génère un embedding pour un seul chunk.

    Args:
        chunk: TextChunk à embedder

    Returns:
        EmbeddedChunk avec embedding
    """
```

### `embed_text()`

Génère un embedding pour du texte brut (utile pour les requêtes).

```python
def embed_text(self, text: str) -> np.ndarray:
    """
    Génère un embedding pour du texte brut.

    Args:
        text: Texte à embedder

    Returns:
        np.ndarray (384-dimensional numpy array)
    """
```

**Exemple (requête de recherche):**

```python
# Embedder une requête de recherche
query_embedding = embedding_service.embed_text("Comment fonctionnent les embeddings ?")
# query_embedding est un numpy array de shape (384,)
```

### `similarity()`

Calcule la similarité cosinus entre deux vecteurs.

```python
def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calcule la similarité cosinus entre deux embeddings.

    Args:
        embedding1: Premier vecteur
        embedding2: Deuxième vecteur

    Returns:
        float entre 0 et 1 (1 = identiques, 0 = orthogonaux)

    Raises:
        ValueError: Si dimensions différentes
    """
```

**Exemple:**

```python
# Calculer la similarité entre deux chunks
emb1 = embedding_result.embedded_chunks[0].embedding
emb2 = embedding_result.embedded_chunks[1].embedding

similarity_score = embedding_service.similarity(emb1, emb2)
print(f"Similarité: {similarity_score:.4f}")  # Ex: 0.7234
```

### `clear_cache()`

Vide le cache des modèles (libère la mémoire).

```python
def clear_cache():
    """Vide le cache des modèles chargés."""
```

## Performance & Optimisation

### Batch processing

Le batch processing accélère les embeddings. Par défaut `batch_size=32`.

```python
# Batch size plus grand = plus rapide mais plus de mémoire
embedding_service = EmbeddingService(batch_size=64)
result = embedding_service.embed_chunks(large_list_of_chunks)
```

### Model caching

Les modèles sont cachés au niveau de la classe pour éviter les rechargements.

```python
# Première initialisation: charge le modèle (~50-100ms)
service1 = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")

# Deuxième initialisation: utilise le cache (~0ms)
service2 = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")

# Les deux pointent vers le même modèle en mémoire
assert service1.model is service2.model
```

### ONNX (CPU-optimized)

fastembed utilise ONNX Runtime pour des inférences rapides sur CPU, sans PyTorch ni GPU requis.

## Intégration avec ChunkingService

EmbeddingService accepte directement la sortie de ChunkingService.

```python
from ragindexer import DocumentParser, ChunkingService, EmbeddingService, FileScanner

# Pipeline complète
scanner = FileScanner("/path/to/docs")
scan_result = scanner.scan()

parser = DocumentParser()
chunking_service = ChunkingService(chunk_size=512, overlap_size=50)
embedding_service = EmbeddingService()

for file_path, file_info in scan_result.files.items():
    # Step 1: Parse
    parsed_doc = parser.parse(file_info)

    # Step 2: Chunk
    chunking_result = chunking_service.chunk(parsed_doc)

    # Step 3: Embed
    embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

    # Les métadonnées fluent à travers:
    # file_info → ParsedDocument.metadata → ChunkMetadata → EmbeddedChunk.chunk.metadata
```

## Modèles disponibles (fastembed)

| Modèle | Dimensions | Taille | Qualité |
|--------|-----------|--------|---------|
| `BAAI/bge-small-en-v1.5` | 384 | ~33MB | Bon (défaut) |
| `BAAI/bge-base-en-v1.5` | 768 | ~109MB | Meilleur |
| `BAAI/bge-large-en-v1.5` | 1024 | ~335MB | Excellent |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~22MB | Bon |

**Note:** fastembed télécharge les modèles automatiquement dans `~/.cache/fastembed/` au premier usage.

## Gestion d'erreurs

```python
# Vérifier les dimensions
try:
    similarity = embedding_service.similarity(emb1, emb2)
except ValueError as e:
    print(f"Erreur: {e}")  # Dimensions mismatch

# Vérifier les chunks vides
try:
    result = embedding_service.embed_chunks([])
except ValueError as e:
    print(f"Erreur: {e}")  # Empty list
```

## Limitation des modèles

- **Max sequence length:** ~512 tokens (~2000 caractères)
  - Les chunks > 512 tokens seront tronqués
  - ChunkingService s'assure de respecter cette limite

- **Modèle stateless:** Pas de context entre chunks
  - Chaque chunk est embedded indépendamment
  - La similarité sémantique entre chunks est possible via cosine distance

## Tests

Tests complets dans `tests/test_embedding_service.py`:

```bash
# Tous les tests d'embedding
./.venv/Scripts/python.exe -m pytest tests/test_embedding_service.py -v

# Avec coverage
./.venv/Scripts/python.exe -m pytest tests/test_embedding_service.py --cov=ragindexer
```

**Coverage:** 88% (22 tests)

## Benchmarks

Sur une machine CPU standard (Intel i7, 8 cores):

| Chunks | Temps | Vitesse | Modèle |
|--------|-------|---------|--------|
| 10 | 0.3s | 33 chunks/s | BAAI/bge-small-en-v1.5 |
| 100 | 1.5s | 66 chunks/s | BAAI/bge-small-en-v1.5 |
| 1000 | 12s | 83 chunks/s | BAAI/bge-small-en-v1.5 |

(Premier run inclut le téléchargement du modèle si non en cache)

## Cas d'usage courants

### 1. Indexation complète

```python
embedding_service = EmbeddingService()
for file_info in scan_result.files.values():
    parsed_doc = parser.parse(file_info)
    chunking_result = chunking_service.chunk(parsed_doc)
    embedding_result = embedding_service.embed_chunks(chunking_result.chunks)
    # Stocker embedding_result dans la base vectorielle
```

### 2. Recherche sémantique

```python
query = "Quelle est la meilleure pratique pour les embeddings ?"
query_embedding = embedding_service.embed_text(query)

# Comparer avec les chunks indexés
for embedded_chunk in indexed_chunks:
    score = embedding_service.similarity(query_embedding.tolist(), embedded_chunk.embedding)
    if score > 0.7:  # Threshold de similarité
        print(f"Match: {embedded_chunk.chunk.content[:50]}... (score: {score:.3f})")
```

### 3. Déduplication

```python
# Trouver les chunks similaires (potentiellement dupliqués)
for i, chunk1 in enumerate(embedded_chunks):
    for chunk2 in embedded_chunks[i + 1 :]:
        similarity = embedding_service.similarity(chunk1.embedding, chunk2.embedding)
        if similarity > 0.95:
            print(f"Possible duplicate: {similarity:.3f}")
```

## Dépendances

- `fastembed>=0.4.0` - Embeddings ONNX-based (remplace sentence-transformers)
- `numpy>=1.24.0` - Opérations vectorielles

## Voir aussi

- [ChunkingService](ChunkingService.md) - Composant 3 (source des chunks)
- [Vector Database Architecture](../Architecture.md#5-vector-database) - Composant 5 (destination des embeddings)
- [Complete Pipeline Example](../examples/embedding_example.py)
