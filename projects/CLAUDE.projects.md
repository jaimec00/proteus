# CLAUDE.projects.md

## Philosophy

- projects are config-driven — write as little code as possible
- core logic lives in the library, projects just compose it via configs
- if you're writing significant logic in a project, it probably belongs in the library instead

## Guidelines

- a project should mostly be config overrides and a thin entry point
- reuse existing configs — only override what changes
- keep experiment scripts minimal and reproducible
