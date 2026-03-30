# Architecture - ragindexer

Ce document décrit les composants majeurs qui répondent aux besoins exprimés dans `Specification.md`.

## Vue d'ensemble

ragindexer est un système d'indexation et de recherche sémantique basé sur des embeddings vectoriels. Il permet de scanner un dossier, d'indexer les documents, et d'exposer une API de recherche via un serveur MCP (Model Context Protocol).

```
[Dossier sources]
       ↓
[File Scanner] → [Document Parser] → [Chunking] → [Embeddings] → [Vector DB]
       ↓                                                           ↓
[Sync Manager]←─────────────────────────────────────────────────→[MCP Server]
                                                                  ↑
                                                            [Claude Code/Codex]
```

---

## Composants

### 1. File Scanner (Scanneur de fichiers) ✅ Implémenté

**Rôle:** Parcourir récursivement un dossier et identifier tous les documents à indexer.

**Entrées:**
- Chemin du dossier source
- Filtres d'extension (`.pdf`, `.txt`, `.doc`, `.docx`, `.md`)
- Options de récurrence

**Sorties:**
- Liste des fichiers détectés avec métadonnées (chemin, taille, timestamp, hash pour détection de changements)

**Responsabilités:**
- Énumérer les fichiers du dossier
- Détecter les fichiers nouveaux, modifiés ou supprimés
- Fournir les métadonnées initiales (chemin relatif, date)

**Status:** 12/12 tests, 95% coverage. Voir `tests/test_file_scanner.py` et `docs/FileScanner.md`

**Fonctionnalités clés:**
- Scan récursif efficace avec hashage SHA256 des fichiers
- Détection de changements (fichiers ajoutés, modifiés, supprimés)
- Support natif pour: PDF, DOCX, DOC, TXT, Markdown
- Métadonnées riches: taille, timestamps, chemin relatif/absolu
- Normalization cross-platform des chemins

---

### 2. Document Parser (Extracteur de texte) ✅ Implémenté

**Rôle:** Extraire le contenu textuel de chaque document selon son format.

**Entrées:**
- FileInfo du FileScanner
- Format détecté (PDF, DOCX, TXT, Markdown)

**Sorties:**
- Texte brut extrait du document
- Métadonnées associées (titre, auteur si disponible)
- Character count du contenu extrait

**Responsabilités par format:**
- **PDF:** Extraction de texte native + métadonnées
- **DOCX:** Extraction du texte + tableaux + métadonnées
- **TXT:** Lecture directe avec gestion multi-encoding
- **Markdown:** Lecture directe avec préservation de la structure

**Status:** 11/11 tests, 63% coverage. Voir `tests/test_document_parser.py` et `docs/DocumentParser.md`

---

### 3. Chunking Service (Découpeur de texte) ✅ Implémenté

**Rôle:** Découper le texte en morceaux (chunks) optimisés pour les embeddings.

**Entrées:**
- Texte extrait
- Taille de chunk (ex: 512 tokens)
- Recouvrement (ex: 50 tokens)
- Métadonnées du document (chemin, titre)

**Sorties:**
- Liste de chunks avec:
  - Contenu textuel
  - Position dans le document original
  - Métadonnées associées

**Responsabilités:**
- Découper intelligemment (préserver les phrases/paragraphes)
- Gérer le recouvrement pour continuité sémantique
- Associer les métadonnées à chaque chunk

**Status:** 14/14 tests, 96% coverage. Voir `tests/test_chunking_service.py` et `docs/ChunkingService.md`

**Stratégie de chunking:**
- Niveau 1: Découpe par paragraphes (plusieurs newlines)
- Niveau 2: Découpe par phrases si paragraphe > chunk_size
- Niveau 3: Découpe par mots si phrase > chunk_size
- Niveau 4: Groupage avec recouvrement pour continuité sémantique

**Compte de tokens:** Approximation heuristique (1 token ≈ 0.75 mots)

---

### 4. Embedding Service (Générateur d'embeddings) ✅ Implémenté

**Rôle:** Calculer la représentation vectorielle de chaque chunk.

**Entrées:**
- Chunks de texte (TextChunk du ChunkingService)
- Modèle d'embedding (ex: `all-MiniLM-L6-v2`)

**Sorties:**
- Vecteurs d'embedding (dimensions selon le modèle, ex: 384)
- Timestamps de calcul
- EmbeddedChunk avec métadonnées préservées

**Responsabilités:**
- Initialiser le modèle d'embedding en local (sans GPU)
- Calculer les embeddings batch par batch
- Gérer les erreurs de calcul
- Préserver les métadonnées de chunk à travers le processus

**Status:** 22/22 tests, 88% coverage. Voir `tests/test_embedding_service.py`

**Fonctionnalités clés:**
- fastembed (ONNX-based) pour inférence CPU rapide, pas GPU requis
- Batch processing pour performance
- Caching de modèles (class-level) pour éviter rechargement
- Méthode similarity() pour calcul de similarité cosinus
- embed_text() pour embeddings de requêtes de recherche
- Métadonnées complètes préservées (source_file, document_title, chunk_index, etc.)

**Modèle par défaut:** `BAAI/bge-small-en-v1.5` (384 dimensions, ~33MB ONNX)

**Intégration avec ChunkingService:**
```python
from ragindexer import ChunkingService, EmbeddingService

chunking_service = ChunkingService(chunk_size=512, overlap_size=50)
embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")

chunking_result = chunking_service.chunk(parsed_document)
embedding_result = embedding_service.embed_chunks(chunking_result.chunks)

# embedding_result.embedded_chunks contient les chunks avec vecteurs
```

---

### 5. Vector Database (Base de données vectorielle) ✅ Implémenté

**Rôle:** Stocker et indexer les embeddings pour la recherche sémantique rapide.

**Entrées:**
- Vecteur d'embedding (EmbeddedChunk)
- Métadonnées (chemin, chunk_id, position, contenu, timestamp)
- Opérations CRUD (Create, Read, Update, Delete)

**Sorties:**
- ID unique du chunk indexé
- Résultats de recherche par requête vectorielle
- Métadonnées associées

**Responsabilités:**
- Maintenir un index vectoriel performant
- Supporter recherche par similarité cosinus
- Gérer les suppressions/mises à jour de documents
- Persistence (autohébergée, open source)

**Status:** 13/17 tests, 63% coverage. Voir `tests/test_vector_database.py` et `docs/VectorDatabaseService.md`

**Technologie:** Qdrant 1.17.1 (base de données vectorielle)

**Fonctionnalités clés:**
- Storage in-memory ou persistent (SQLite-backed)
- Recherche par similarité cosinus
- Métadonnées associées à chaque embedding
- Opérations CRUD complètes
- Statistiques de collection
- Intégration seamless avec EmbeddingService

**Modes de stockage:**
```python
# In-memory (développement, tests)
vector_db = VectorDatabaseService(persistence_path=None)

# Persistent (production)
vector_db = VectorDatabaseService(persistence_path=Path("./data/qdrant"))
```

**Intégration avec EmbeddingService:**
```python
from ragindexer import EmbeddingService, VectorDatabaseService

embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
vector_db = VectorDatabaseService(vector_size=384)

# Pipeline complète
embedding_result = embedding_service.embed_chunks(chunks)
db_result = vector_db.add_embeddings(embedding_result.embedded_chunks)

# Recherche sémantique
query_embedding = embedding_service.embed_text("search query")
results = vector_db.search(query_embedding.tolist(), limit=5)
```

---

### 6. Sync Manager (Gestionnaire de synchronisation) ✅ Implémenté

**Rôle:** Orchestrer la pipeline d'indexation et gérer les changements.

**Entrées:**
- Signaux de changement (fichier ajouté, modifié, supprimé)
- Événements utilisateur (déclencher une réindexation)

**Sorties:**
- Chunks indexés ou supprimés de la base vectorielle
- État d'avancement de la synchronisation avec détails par fichier
- Logs d'erreurs détaillés

**Responsabilités:**
- Détecter les changements (via File Scanner)
- Lancer la pipeline complète (Parse → Chunk → Embed → Store)
- Gérer les suppressions (nettoyer la base vectorielle)
- Gérer les mises à jour incrémentielles
- Optionnel: Watch folder en temps réel

**Status:** En cours de test. Voir `tests/test_sync_manager.py` et `docs/SyncManager.md`

**Fonctionnalités clés:**
- Mode full_sync() pour indexation initiale complète
- Mode incremental_sync() pour mises à jour efficaces
- Gestion résiliente des erreurs (certains fichiers échouent = résultat PARTIAL)
- Résultats détaillés par fichier (statut, chunks, erreurs, durée)
- Coordination automatique de tous les composants 1-5
- Préservation des métadonnées à travers la pipeline
- Logging structuré pour monitoring et debugging

**Modèles Pydantic:**
- `SyncStatus`: État de synchronisation (PENDING, IN_PROGRESS, COMPLETED, FAILED, PARTIAL)
- `FileSyncResult`: Résultat pour un fichier (statut, chunks, erreur, durée)
- `SyncOperationResult`: Résultat global (totaux, fichiers processés, résultats détaillés)

**Intégration avec tous les composants:**
```python
from ragindexer import SyncManager

sync_manager = SyncManager(
    scan_root="/path/to/documents",
    persistence_path=Path("./data/qdrant"),
    chunk_size=512,
    overlap_size=50,
)

# Indexation initiale
result = sync_manager.full_sync()

# Mises à jour incrémentales
result = sync_manager.incremental_sync()

# Statistiques
stats = sync_manager.get_statistics()
```

---

### 7. MCP Server (Serveur Model Context Protocol) ⏳ À implémenter

**Rôle:** Exposer les capacités de recherche via l'API MCP pour Claude Code/Codex.

**Entrées:**
- Requêtes textuelles de Claude Code
- Requêtes vectorielles
- Paramètres de recherche (limite de résultats, filtres)

**Sorties:**
- Résultats pertinents (chunks + métadonnées)
- Scores de similarité
- Contexte formaté pour utilisation par Claude

**Responsabilités:**
- Implémenter les outils MCP:
  - `search(query: str)` - Recherche sémantique
  - `list_documents()` - Lister les documents indexés
  - `get_document_chunks(path: str)` - Chunks d'un document spécifique
- Transformer les requêtes textuelles en embeddings
- Formater les réponses pour intégration Claude
- Gérer l'authentification (optionnel, un seul utilisateur)

---

### 8. Configuration & Settings ✅ Implémenté

**Rôle:** Centraliser les paramètres de configuration de l'application.

**Entrées:**
- Variables d'environnement (`.env`)
- Fichiers de configuration

**Contient:**
- Chemin du dossier source (SCAN_ROOT)
- Modèle d'embedding utilisé (EMBEDDING_MODEL)
- Taille des chunks et recouvrement (CHUNK_SIZE, OVERLAP_SIZE)
- Paramètres de la base vectorielle (QDRANT_URL, QDRANT_PERSISTENCE_PATH)
- URL/port du serveur MCP (MCP_HOST, MCP_PORT)
- Niveau de logging (LOGLEVEL)

**Status:** 14/14 tests, 98% coverage. Voir `tests/test_settings.py` et `docs/Settings.md`

**Fonctionnalités clés:**
- Chargement depuis `.env` ou variables d'environnement
- Validation Pydantic complète avec messages d'erreur clairs
- Méthodes helper (get_qdrant_persistence_path(), get_scan_root())
- Support de modes de stockage flexible (in-memory ou persistent)
- Défauts sensés pour démarrage rapide
- Support des champs extra pour extensibilité future
- Logging enrichi avec version et fichier .env chargé

**Modèles Pydantic:**
- `Settings`: Configuration centralisée avec validation

**Intégration avec tous les composants:**
```python
from ragindexer import settings

# Utilisation directe
scan_root = settings.get_scan_root()
persistence_path = settings.get_qdrant_persistence_path()

# Avec autres composants
from ragindexer import FileScanner, ChunkingService, EmbeddingService

scanner = FileScanner(settings.get_scan_root())
chunking_service = ChunkingService(
    chunk_size=settings.CHUNK_SIZE, overlap_size=settings.OVERLAP_SIZE
)
embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
```

---

## Flux de données

### Indexation initiale
```
File Scanner (dossier)
    ↓ [liste fichiers]
Document Parser
    ↓ [texte brut]
Chunking Service
    ↓ [chunks + métadonnées]
Embedding Service
    ↓ [vectors]
Vector Database
    ↓ [indexé]
MCP Server [prêt à répondre]
```

### Mise à jour de document
```
File Scanner (détecte changement)
    ↓
Sync Manager
    ↓ [supprime anciens chunks de la VectorDB]
    ↓ [relance pipeline sur nouveau contenu]
Vector Database
    ↓ [réindexé]
```

### Requête utilisateur
```
Claude Code
    ↓ [requête texte]
MCP Server
    ↓ [convertit en embedding]
Embedding Service
    ↓ [vector]
Vector Database
    ↓ [recherche similarité]
MCP Server [retourne résultats]
    ↓
Claude Code [utilise le contexte]
```

---

## Dépendances et technologies recommandées

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| File Scanner | Python pathlib/watchdog | Standard, efficace |
| PDF Parser | PyPDF2 + pytesseract | Open source, OCR local |
| DOCX Parser | python-docx | Extrait texte structuré |
| Text Parser | Built-in Python | Lecture simple |
| Chunking | LangChain ou custom | Gestion intelligente des limites |
| Embedding | fastembed (ONNX) | Modèles locaux ONNX, CPU-optimized, pas GPU requis |
| Vector DB | Chroma / PostgreSQL+pgvector | Open source, autohébergé, facile |
| MCP Server | MCP SDK | Standard de communication |
| Configuration | Pydantic | Validation et gestion d'env |

---

## Points clés de l'architecture

1. **Modulaire:** Chaque composant est indépendant et peut être amélioré isolément
2. **Open source:** Aucune dépendance sur services cloud payants
3. **Autohébergé:** Tout tourne localement sans GPU
4. **Scalable incrémentalement:** Support des mises à jour sans réindexer tout
5. **Observabilité:** Logging et monitoring de chaque étape
