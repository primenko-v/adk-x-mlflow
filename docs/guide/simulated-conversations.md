# Simulated conversations

To produce traces for inspection and evaluation, the simulator runs
conversations against the agent. There are **two kinds**, and you choose per
file just by what you put in it. Files live under
`simulations/conversations/` (read recursively), conventionally split into
`scenarios/` and `static/` subfolders. `make simulate` runs every file found
there in a single batch — the kind is detected from each file's contents, so
there are no flags to set.

| Kind | Put this in the file | What happens |
|---|---|---|
| **Scenario** (LLM-driven) | a `conversation_plan` | An LLM plays the user, improvising each turn to follow your plan. Good for open-ended, realistic, multi-turn chats. |
| **Static** (fixed input) | a `messages` list | Your exact messages are sent verbatim, in order. No LLM playing the user. Good when you want precise, repeatable inputs. |

Pick scenario when you want natural, varied conversations; pick static when you
have the exact questions you want to ask. The two are mutually exclusive within
a file — a file has either a `conversation_plan` or a `messages` list, never
both.

## Scenario conversations

A scenario file describes *what the user is trying to do*, and an LLM
improvises the actual turns to pursue that goal:

```yaml
# simulations/conversations/scenarios/curious_traveler.yaml
starting_prompt: "Hi! I'm planning a trip and want to compare the weather."
conversation_plan: |
  - Ask for the temperature in a couple of cities.
  - Then ask which one is warmer.
  - End once you've asked at least three questions.
```

- `starting_prompt` — the user's opening message, sent verbatim as turn 1.
- `conversation_plan` — free-text instructions the LLM-played user follows to
  drive the rest of the conversation. It keeps going, improvising each turn,
  until the plan says to stop (or the turn cap is hit).

### Configuring the LLM user

The user simulator is shared by every scenario in the batch and configured in
`simulations/user_simulator.yaml`:

```yaml
model: gemini-3.1-flash-lite
maxAllowedInvocations: 20
```

- `model` — which model plays the user.
- `maxAllowedInvocations` — a hard cap on turns, so a plan that never decides to
  stop can't loop forever.

This config only affects scenario files; static files replay their fixed
`messages` and ignore it.

## Static conversations

A static file is just a list of user turns, replayed verbatim, in order — no
LLM playing the user:

```yaml
# simulations/conversations/static/berlin_then_tokyo.yaml
messages:
  - "What's the temperature in Berlin?"
  - "And Tokyo?"
```

Each line is sent to the agent in order, exactly as written. Use static
conversations when you want precise, repeatable inputs.

### Starting from pre-existing context

A static file can also set up state *before* the first message, via an optional
`state` block. The agent remembers a preferred temperature unit, so you can
establish it two ways:

**Set it during the conversation** — an earlier turn changes the preference, and
later turns should respect it:

```yaml
# simulations/conversations/static/unit_preference_in_convo.yaml
messages:
  - "Show temperatures in Fahrenheit from now on."
  - "What's the temperature in Berlin?"   # expect °F
  - "And Tokyo?"                            # expect °F
```

**Seed it up front** — the preference is already in place before turn 1, without
ever being discussed:

```yaml
# simulations/conversations/static/seeded_fahrenheit.yaml
state:
  temperature_unit: fahrenheit
messages:
  - "What's the temperature in Berlin?"     # expect °F
```

Use the seeded form when you want to evaluate a single turn as if a lot had
already happened — without having to script all of it.

## Running them

```bash
# Run every conversation under simulations/conversations/ and log traces to MLflow
make simulate

# Or point at a different directory / send spans to a file instead of MLflow
uv run python -m mlflow_adk.simulate --input path/to/dir
uv run python -m mlflow_adk.simulate --input path/to/dir --output-traces traces.jsonl
```

`make simulate` also writes the produced trace IDs to `.last_trace_ids.txt`, so
[`make evaluate`](evaluation.md) can score exactly those traces.

In the MLflow UI you can filter the trace list by the `conversation_mode` tag
(`scenario` or `static`) to look at one kind at a time.
