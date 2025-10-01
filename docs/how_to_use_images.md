# Using Images and Diagrams in Cursor Sessions

Visual artefacts carry critical context. Treat every diagram or screenshot in `docs/`, `design/`, and `assets/` as first-class input to the agent loop.

## Recommended Patterns
- **Architecture flows:** Reference sequence diagrams or block diagrams when planning changes that affect data flow. Mention the file path in the task plan so reviewers can cross-check assumptions.
- **API schemas:** When implementing or updating endpoints, link to any Swagger captures or schema PNGs to keep contracts aligned.
- **Experiment logs & UI screenshots:** If a validation run fails, attach the screenshot (e.g., `docs/validation_branin.png`) when summarizing the issue so the reasoning agent can inspect the UI or log output.

## How to Link Images
- Store images under version control (e.g., `docs/`, `design/`, `assets/`).
- Use Markdown syntax in plans and PR descriptions: `![Driver health check](docs/validation_branin.png)`.
- Cite the image path in final summaries alongside code references so approvers have a single source of truth.

## Security & Privacy
- Never include raw personally identifiable information or export control data in screenshots.
- Blur or mask any experiment identifiers that are not meant for wide distribution before committing the image.

Embedding relevant visuals accelerates reviews and keeps context synchronized across the team and agents.
