# legalRag

RAG (Retrieval-Augmented Generation) system for legal documents.

## Context Directory

The `context/` directory contains documentation that helps Claude understand specific parts of the codebase:

- **API group docs** - Each group of related endpoints should have a context doc describing:
  - What the endpoints do
  - Request/response formats
  - Authentication requirements
  - Error handling patterns
  - Database interactions

When adding new features, create or update the relevant context doc so Claude can understand and work with that code effectively.

## Skills Available

- `/code-reviewer` - Review uncommitted changes

## Agents Available

- `code-reviewer` - Detailed code review for architecture compliance
- `privacy-reviewer` - Check for unintentional data collection/sharing
