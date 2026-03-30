# SyncManager - Gestionnaire de synchronisation

## Vue d'ensemble

Le **SyncManager** est le composant 6 de ragindexer. Il orchestre la pipeline complète d'indexation et gère la synchronisation des changements de fichiers.

## Responsabilités

### 1. Orchestration de la pipeline
Le SyncManager coordonne tous les composants pour créer une pipeline complète :
- **FileScanner**: Détecte les fichiers à traiter
- **DocumentParser**: Extrait le texte des documents
- **ChunkingService**: Découpe le texte en chunks sémantiques
- **EmbeddingService**: Génère les vecteurs d'embedding
- **VectorDatabaseService**: Stocke les embeddings pour la recherche

### 2. Gestion des changements
Détecte et traite les trois types de changements :
- **Fichiers ajoutés** : Passe complète par la pipeline
- **Fichiers modifiés** : Supprime les anciens embeddings, relance la pipeline
- **Fichiers supprimés** : Supprime les embeddings associés

### 3. Synchronisation incrémentale
Supporte deux modes de synchronisation :
- **full_sync()** : Indexe tous les documents (utiliser au démarrage)
- **incremental_sync()** : Traite seulement les changements (utiliser pour les mises à jour)

## Modèles de données

### SyncStatus
État d'une opération de synchronisation :
- `PENDING` : En attente
- `IN_PROGRESS` : En cours d'exécution
- `COMPLETED` : Complétée avec succès
- `FAILED` : Échec complètement
- `PARTIAL` : Succès partiel (certains fichiers ont échoué)

### FileSyncResult
Résultat de la synchronisation d'un seul fichier :
```python
@dataclass
class FileSyncResult:
    relative_path: str           # Chemin du fichier
    status: SyncStatus           # Statut (COMPLETED, FAILED, etc.)
    change_type: ChangeType      # Type de changement (ADDED, MODIFIED, DELETED)
    chunks_count: int            # Nombre de chunks créés
    error: Optional[str]         # Message d'erreur si échec
    duration_seconds: float      # Temps de traitement
    synced_at: datetime          # Timestamp de synchronisation
```

### SyncOperationResult
Résultat de l'opération complète de synchronisation :
```python
@dataclass
class SyncOperationResult:
    scan_root: Path              # Dossier scané
    total_files_processed: int   # Nombre de fichiers traités
    total_files_added: int       # Fichiers ajoutés
    total_files_modified: int    # Fichiers modifiés
    total_files_deleted: int     # Fichiers supprimés
    total_chunks_created: int    # Total de chunks créés/mis à jour
    total_errors: int            # Nombre d'erreurs
    file_results: Dict[str, FileSyncResult]  # Résultats par fichier
    overall_status: SyncStatus   # Statut global
    duration_seconds: float      # Temps total
    synced_at: datetime          # Timestamp de synchronisation
```

## Utilisation

### Initialisation

```python
from ragindexer import SyncManager

# Créer un SyncManager
sync_manager = SyncManager(
    scan_root="/path/to/documents",
    persistence_path=Path("./data/qdrant"),  # Optional: for persistent storage
    chunk_size=512,           # Taille des chunks en tokens
    overlap_size=50,          # Recouvrement entre chunks en tokens
    embedding_model="all-MiniLM-L6-v2",
)
```

### Full Sync (Indexation initiale)

Utilisez `full_sync()` pour indexer tous les documents d'un dossier :

```python
# Indexer tous les documents
result = sync_manager.full_sync()

if result.overall_status == SyncStatus.COMPLETED:
    print(f"✅ {result.total_files_added} fichiers indexés")
    print(f"📊 {result.total_chunks_created} chunks créés")
else:
    print(f"❌ Erreurs: {result.total_errors}")
    for path, file_result in result.file_results.items():
        if file_result.status == SyncStatus.FAILED:
            print(f"  - {path}: {file_result.error}")
```

### Incremental Sync (Mise à jour incrémentale)

Après le premier `full_sync()`, utilisez `incremental_sync()` pour traiter uniquement les changements :

```python
# Première synchronisation
sync_manager.full_sync()

# ... Plus tard, après des changements de fichiers ...

# Synchronisation incrémentale
result = sync_manager.incremental_sync()

print(f"➕ {result.total_files_added} fichiers ajoutés")
print(f"🔄 {result.total_files_modified} fichiers modifiés")
print(f"🗑️ {result.total_files_deleted} fichiers supprimés")
```

### Récupération des statistiques

```python
# Obtenir les statistiques de la base vectorielle
stats = sync_manager.get_statistics()
print(f"Embeddings stockés: {stats['point_count']}")
print(f"Dimension: {stats['vector_size']}")
```

### Récupération du dernier résultat de scan

```python
last_scan = sync_manager.get_last_scan_result()
if last_scan:
    print(f"Fichiers détectés: {last_scan.total_files}")
```

## Flux de traitement

### Synchronisation complète (full_sync)

```
1. Scan du dossier
   ↓
2. Pour chaque fichier détecté:
   a. Parse document
   b. Chunk text
   c. Embed chunks
   d. Store in VectorDB
   ↓
3. Retour du résultat global
```

### Synchronisation incrémentale (incremental_sync)

```
1. Scan du dossier
   ↓
2. Comparaison avec le dernier scan
   ↓
3. Pour chaque changement détecté:
   a. Si ADDED/MODIFIED:
      - Supprimer anciens embeddings (si modifié)
      - Parse → Chunk → Embed → Store

   b. Si DELETED:
      - Supprimer embeddings de la VectorDB
   ↓
4. Retour du résultat global
```

## Gestion des erreurs

Le SyncManager gère les erreurs de manière résiliente :

1. **Erreurs au niveau fichier** : Les erreurs de traitement d'un fichier n'arrêtent pas le traitement des autres
2. **Signalement détaillé** : Chaque fichier problématique est signalé avec un message d'erreur
3. **Statut partiel** : Si certains fichiers échouent, le statut global est `PARTIAL`
4. **Logging** : Les erreurs sont loggées avec contexte pour investigation

## Caractéristiques principales

### 1. Pipeline coordonnée
- Automatise l'ensemble du flux d'indexation
- Gère les interfaces entre composants
- Préserve les métadonnées à travers la pipeline

### 2. Synchronisation efficace
- Détecte les changements sans rechaque
- Supprime et remplace seulement les documents modifiés
- Traite les suppressions proprement

### 3. Suivi détaillé
- Résultats par fichier avec statut et durée
- Compteurs agrégés pour monitoring
- Messages d'erreur détaillés pour debugging

### 4. Robustesse
- Erreur sur un fichier n'affecte pas les autres
- Distinction claire entre COMPLETED, PARTIAL, et FAILED
- Logging détaillé à chaque étape

## Exemple complet

```python
from pathlib import Path
from ragindexer import SyncManager, SyncStatus

# Initialisation
manager = SyncManager(
    scan_root="/data/documents",
    persistence_path=Path("./qdrant_storage"),
    chunk_size=512,
    overlap_size=50,
)

# Indexation initiale
print("🚀 Starting full indexing...")
result = manager.full_sync()

if result.overall_status == SyncStatus.COMPLETED:
    print(f"✅ Indexed {result.total_files_added} files")
    print(f"📊 Created {result.total_chunks_created} chunks")
    stats = manager.get_statistics()
    print(f"💾 Database contains {stats['point_count']} embeddings")
else:
    print(f"❌ Indexing failed with {result.total_errors} errors")
    for path, file_result in result.file_results.items():
        if file_result.status == SyncStatus.FAILED:
            print(f"  Error in {path}: {file_result.error}")

# Plus tard, après des changements...
print("\n🔄 Synchronizing changes...")
result = manager.incremental_sync()

if result.total_files_processed > 0:
    print(f"➕ Added: {result.total_files_added}")
    print(f"🔄 Modified: {result.total_files_modified}")
    print(f"🗑️ Deleted: {result.total_files_deleted}")
else:
    print("✅ No changes detected")
```

## Tests

Le SyncManager est couvert par une suite complète de tests :
- Initialisation et configuration
- Full sync avec plusieurs fichiers
- Incremental sync (ajout, modification, suppression)
- Gestion des erreurs
- Cycles multiples de synchronisation

Exécutez les tests avec :
```bash
./.venv/Scripts/python.exe -m pytest tests/test_sync_manager.py -v
```

## Performance

### Facteurs affectant la performance

1. **Nombre de fichiers** : Le scan est O(n) où n = nombre de fichiers
2. **Taille des documents** : Plus les documents sont grands, plus l'embedding prend du temps
3. **Taille des chunks** : Petits chunks = plus de chunks = plus d'embeddings
4. **Modèle d'embedding** : Le modèle par défaut est équilibré pour vitesse/qualité

### Optimisations

- **Batch processing** : Les embeddings sont traités par batch (défaut: 32 chunks)
- **Caching du modèle** : Les modèles sont cachés au niveau de la classe
- **Persistent storage** : Option de persistence pour éviter les recharges

## Intégration avec les autres composants

Le SyncManager s'appuie sur tous les composants précédents :

| Composant | Rôle |
|-----------|------|
| FileScanner (1) | Détecte les fichiers et les changements |
| DocumentParser (2) | Extrait le texte des documents |
| ChunkingService (3) | Découpe le texte en chunks |
| EmbeddingService (4) | Génère les vecteurs |
| VectorDatabaseService (5) | Stocke et indexe les embeddings |

