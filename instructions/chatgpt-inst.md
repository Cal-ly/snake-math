You are an expert agent assisting with refactoring the Snake Math project (see `instructions/project_blueprint.md`, `CLAUDE.md`, and `instructions/concept_page_template.md`).

### Goal:
Restructure the content to:

1. Split long `.md` pages into smaller, topic-specific pages.
2. Generate index pages (e.g. `statistics-index.md`).
3. Suggest and apply component splits/duplications where needed.
4. Propose new folders where appropriate.
5. Update internal links accordingly.
6. Preserve existing style, tone, progressive structure (`instructions/concept_page_template.md`).

### Steps:

1. Analyze the project:
    - `.md` pages in `docs/`
    - Vue components in `components/`
    - Project style and architecture in `instructions/project_blueprint.md`, `CLAUDE.md`, and `instructions/concept_page_template.md`

2. Create and propose a **mapping table**:
    - old pages → new pages
    - component adjustments
    - proposed folder structure
    - index pages

**Pause and wait for my approval** before making changes.

3. Once approved:
    - Work **one page at a time**.
    - Log progress in `instructions/task-log.md`:
        - What was done
        - What’s next
        - Any issues/questions

4. After each update:
    - New `.md` pages use correct style & sections.
    - Vue/PyScript components are preserved and functional.
    - Internal links updated.
    - No loss of functionality.

### Notes:

- Split topics by conceptual granularity.
- Flag ambiguous cases for review.
- Keep consistent tone (informal, accessible) and interactivity.
- Maintain an up-to-date `instructions/task-log.md`.

---

Your first action:  
Propose the **mapping table** (no file edits yet).  
Wait for my confirmation.
