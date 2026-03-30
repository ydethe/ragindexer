# DocumentParser - Documentation Détaillée

## Vue d'ensemble

Le `DocumentParser` est le composant 2 de ragindexer responsable de l'extraction de contenu textuel à partir de documents dans divers formats.

**Formats supportés:**
- PDF (avec extraction de texte native)
- DOCX (Microsoft Word)
- DOC (traité comme DOCX)
- TXT (texte brut)
- Markdown

## Architecture

### Modèles Pydantic

#### `DocumentMetadata`
Contient les métadonnées extraites du document:
```python
class DocumentMetadata(BaseModel):
    title: Optional[str]          # Titre du document
    author: Optional[str]         # Auteur du document
    page_count: Optional[int]     # Nombre de pages (PDF/DOCX)
    source_file: str              # Chemin du fichier source
    format: FileFormat            # Format du document
    extraction_time: datetime     # Moment de l'extraction
```

#### `ParsedDocument`
Résultat principal de l'analyse:
```python
class ParsedDocument(BaseModel):
    content: str                  # Texte extrait
    metadata: DocumentMetadata    # Métadonnées
    file_info: FileInfo          # Référence FileInfo du scanner
    character_count: int         # Nombre de caractères extraits
```

### Classe `DocumentParser`

```python
class DocumentParser:
    def __init__(self, logger_instance: Optional[logging.Logger] = None)
    def parse(self, file_info: FileInfo) -> ParsedDocument
```

## Utilisation

### Exemple basique
```python
from ragindexer.DocumentParser import DocumentParser
from ragindexer.FileScanner import FileScanner

# Initialiser le scanner
scanner = FileScanner("/path/to/documents")
scan_result = scanner.scan()

# Initialiser le parser
parser = DocumentParser()

# Parser chaque fichier détecté
for rel_path, file_info in scan_result.files.items():
    try:
        parsed_doc = parser.parse(file_info)
        print(f"Texte extrait: {len(parsed_doc.content)} caractères")
        print(f"Titre: {parsed_doc.metadata.title}")
        print(f"Auteur: {parsed_doc.metadata.author}")

        # Utiliser le contenu extrait
        # ... (passer au Chunking Service)
    except IOError as e:
        print(f"Erreur lors du parsing: {e}")
```

### Intégration avec FileScanner
Le DocumentParser accepte directement les objets `FileInfo` du FileScanner:

```python
from ragindexer.FileScanner import FileScanner, FileInfo
from ragindexer.DocumentParser import DocumentParser

scanner = FileScanner("/docs")
result = scanner.scan()

parser = DocumentParser()

# Le parser comprend les métadonnées FileInfo
for file_info in result.files.values():
    parsed = parser.parse(file_info)
    # Accès aux métadonnées du scanner
    print(file_info.file_size)    # Taille en bytes
    print(file_info.modified_time) # Date de modification
    print(file_info.file_hash)     # Hash SHA256
```

## Extraction par format

### PDF
- **Méthode:** `_parse_pdf(file_info: FileInfo)`
- **Extraction:** Texte à partir de chaque page
- **Métadonnées:** Titre, auteur, nombre de pages (si disponibles dans les propriétés PDF)
- **Notes:** Support OCR peut être ajouté à l'avenir

Exemple:
```
Page 1: "Introduction..."
Page 2: "Contenu principal..."
-> Combiné en un seul texte avec séparateurs de pages
```

### DOCX
- **Méthode:** `_parse_docx(file_info: FileInfo)`
- **Extraction:**
  - Paragraphes
  - Contenu des tableaux (format: `Col1 | Col2 | Col3`)
- **Métadonnées:** Titre et auteur depuis les propriétés du document

Exemple:
```
Paragraph 1
Paragraph 2
Table:
Header 1 | Header 2
Data 1 | Data 2
```

### TXT
- **Méthode:** `_parse_txt(file_info: FileInfo)`
- **Extraction:** Contenu brut du fichier
- **Encodage:** UTF-8 par défaut, sinon Latin-1

### Markdown
- **Méthode:** `_parse_markdown(file_info: FileInfo)`
- **Extraction:** Contenu brut avec préservation de la syntaxe Markdown
- **Encodage:** UTF-8 par défaut, sinon Latin-1

## Gestion d'erreurs

### Exceptions possibles

| Exception | Cause | Exemple |
|-----------|-------|---------|
| `IOError` | Fichier non trouvable ou illisible | Fichier supprimé après scan |
| `ValueError` | Format non supporté | Format non reconnu |
| `Exception` | Erreur lors du parsing | PDF corrompu, DOCX invalid |

### Pattern de gestion d'erreur recommandé
```python
parser = DocumentParser()
try:
    parsed = parser.parse(file_info)
except IOError as e:
    # Fichier inaccessible, peut être ignoré
    logger.warning(f"Fichier inaccessible: {e}")
except Exception as e:
    # Erreur de parsing, document peut être problématique
    logger.error(f"Erreur d'extraction: {e}")
```

## Performance

### Caractéristiques
- **Parsing en mémoire:** Aucun fichier temporaire
- **Pas de limite de taille:** Adapté aux documents volumineux
- **Pas de parallelisation:** À implémenter au niveau du Sync Manager

### Optimisations recommandées
Pour traiter de nombreux documents:
```python
from concurrent.futures import ThreadPoolExecutor

parser = DocumentParser()

with ThreadPoolExecutor(max_workers=4) as executor:
    parsed_docs = list(executor.map(parser.parse, file_infos))
```

## Logging

Le DocumentParser fournit des logs à différents niveaux:

```python
from ragindexer.DocumentParser import DocumentParser
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)

parser = DocumentParser()
# INFO: "Parsing document: file.pdf"
# INFO: "Successfully parsed file.pdf: 45000 characters"
# WARNING/ERROR en cas de problème
```

## Intégration avec le pipeline complet

```
[File Scanner]
    ↓ FileInfo
[Document Parser] ← Vous êtes ici
    ↓ ParsedDocument.content
[Chunking Service]
    ↓ chunks
[Embedding Service]
    ↓ vectors
[Vector Database]
```

Le DocumentParser transforme les métadonnées du fichier en contenu textuel exploitable pour les étapes suivantes.

## Tests

Voir `tests/test_document_parser.py` pour:
- Tests unitaires par format
- Tests d'intégration avec FileScanner
- Tests de gestion d'erreur
- Tests de préservation de contenu

Exécution:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_document_parser.py -v
```

## Améliorations futures

1. **Support OCR pour PDF scanné**
   - Intégration pytesseract
   - Détection PDF scanné vs natif

2. **Extraction de métadonnées avancée**
   - Dates de création/modification depuis PDF
   - Auteur, sujet depuis DOCX

3. **Détection de langue**
   - Amélioration du chunking multilingue

4. **Support de formats additionnels**
   - Excel (XLSX)
   - PowerPoint (PPTX)
   - RTF

5. **Validation et sanitization**
   - Suppression de caractères de contrôle
   - Normalisation des espaces blancs
