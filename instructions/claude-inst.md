You are an expert AI assistant helping me refactor my Snake Math project (see `instructions/project_blueprint.md`, `CLAUDE.md`, and `instructions/concept_page_template.md` for details).

### Objective:
Restructure Markdown pages and Vue components to:

- Split large pages into smaller, more focused pages.
- Add index pages to link to related topics.
- Enable greater extensibility and usability.
- Maintain consistent style, functionality, and interactivity.

### Instructions:

- Split pages with conceptual granularity — large topics should become smaller, focused `.md` pages.
- Create or suggest index pages (e.g. `statistics-index.md`).
- Suggest when Vue components should be split or duplicated.
- Propose new folders/subfolders if needed.
- Update all internal links across pages.
- Follow the project style: tone, structure, components — as in `instructions/concept_page_template.md`.

### Workflow:

1. Analyze the codebase:
    - Existing `.md` files
    - Vue components (`components/`)
    - Project structure (`instructions/project_blueprint.md`, `CLAUDE.md`, `instructions/concept_page_template.md`)

2. Propose a **mapping table**:
    - old page → new pages
    - components reused/split
    - new folders
    - index pages

3. Wait for my confirmation.  
    → Do not make any changes yet.

4. After approval:
    - Work one page at a time - have `instructions/concept_page_template.md` in mind.
    - Log progress in `instructions/task-log.md`:
        - What’s done
        - What’s next
        - Any issues/questions

5. After each change:
    - Verify new `.md` pages follow project structure.
    - Ensure Vue/PyScript components work.
    - Update internal links.
    - No functionality lost.

### Special:

- If unclear how to split a topic → flag for review.
- Maintain tone, clarity, interactivity.
- Always keep `instructions/task-log.md` up to date for human review or handoff.

---

First task:  
Propose the **mapping table** based on current pages.  
Wait for my approval before doing anything.
