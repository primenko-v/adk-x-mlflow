# Prompt versioning and workflow

The agent's instruction lives in three places:

- `_TEMPLATE` in `src/mlflow_adk/agents/simple_agent/prompts.py` — canonical text in your branch.
- MLflow Prompt Registry — versioned history.
- `prompt_frozen.json` (committed) — what production runs.

The running agent reads from MLflow (dev) or from `prompt_frozen.json` (prod). It never reads `_TEMPLATE` directly.

## Versions and aliases

- A **version** (v1, v2, v3...) is immutable. Once registered, it never changes.
- An **alias** (e.g. `@production`, `@alice`) is a mutable name pointing at exactly one version.
- Reassigning an alias moves the pointer atomically — the old binding is gone.

The only alias this project knows about by name is `@production`. Every other alias is per-developer or per-feature; you create and move them yourself in the MLflow UI.

## Choosing what the agent reads

`PROMPT_SOURCE` picks the mode:

- `registry` (default) — read from MLflow at agent startup.
- `frozen` — read `prompt_frozen.json`. No MLflow contact.

In `registry` mode, `PROMPT_REF` says which ref. Default is `@production`.

```bash
make simulate                              # @production (default)
PROMPT_REF=@alice make simulate            # by alias
PROMPT_REF=5 make simulate                 # by version number
```

## Workflow

```
1. Edit _TEMPLATE in prompts.py.
2. make register_prompt                          → new MLflow version (no alias).
3. In the MLflow UI, assign your alias to it (e.g. @your-handle).
4. PROMPT_REF=@your-handle make simulate         → iterate against your alias.
5. In the MLflow UI, move @production onto your version when ready to ship.
6. make freeze_prompt                            → @production → prompt_frozen.json.
7. Commit prompt_frozen.json. That's what production runs.
```

Aliases (including `@production`) are managed entirely through the UI — there is no command that moves them. The first time on a fresh MLflow server, step 5 just creates `@production` instead of moving it.
