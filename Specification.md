# Spécifications - ragindexer

## Description du besoin

ragindexer vise à indexer un dossier récursivement pour lister tous les fichiers.
Pour chaque fichier :
1. On récupère le texte (avec un OCR si c'est un PDF)
2. On calcule l'embedding chunk par chunk
3. On stocke les embeddings et leurs metadonnées dans une base de données vectorielles

ragindexer expose un serveur MCP pour que claude code, codex ou équivalent puisse interroger cette base et répondre aux questions.

## Contraintes

Tout doit être open source et autohébergé
La plateforme hôte est linux sans GPU
Un seul utilisateur de la plateforme
Utilisation de python et pydantic pour les types de données
Chaque fonction ou méthode implémentée doit faire l'objet d'un ou plusieurs tests unitaires

## Cas d'usages typiques

Lors d'un ajout / modification / suppression de document, la base de données est mise à jour
Via claude code, codex ou équivalent, les questions sont répondues avec le serveur MCP hébergé sur la plateforme hôte

## Format d'entrée

Tous documents :
- pdf
- doc ou docx
- txt
- markdown
