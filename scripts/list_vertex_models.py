"""Probe which Gemini publisher models your project can actually call.

Usage:
    uv run python scripts/list_vertex_models.py <region>
    uv run python scripts/list_vertex_models.py us-central1

The catalog (``client.models.list``) shows every Gemini model Google has
published, regardless of whether *your* project can invoke it. Even
``launchStage == GA`` is only Google's claim about the model — per-project
allowlisting, quota allocation, and regional rollout waves often gate
access for weeks after a GA announcement.

This script sends a 1-token ``generateContent`` request to each Gemini
model and classifies the response — so the output reflects what's
*actually callable* from this project/region right now, not what's
*advertised*. Cost is negligible (≤25 single-token completions).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import errors as genai_errors

from mlflow_adk.settings import settings

# Models in these categories don't speak the generateContent text-chat shape;
# probing them returns 400/INVALID_ARGUMENT and just adds noise. Skip them —
# they were never judge candidates anyway.
_NON_CHAT_HINTS = (
    "embedding",
    "image",
    "tts",
    "native-audio",
    "computer-use",
    "live",
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("region", help="Vertex AI region, e.g. us-central1, europe-west1")
args = parser.parse_args()

client = genai.Client(
    vertexai=True,
    project=settings.google_cloud_project,
    location=args.region,
)

gemini_names = sorted(
    {
        model.name.rsplit("/", 1)[-1]
        for model in client.models.list(config={"page_size": 100, "query_base": True})
        if "gemini" in model.name.lower()
    }
)


def _probe(name: str) -> str:
    """Single 1-token ping → ``OK`` / ``NO_ACCESS`` / ``<error code>``."""
    if any(hint in name for hint in _NON_CHAT_HINTS):
        return "NOT_CHAT"
    try:
        client.models.generate_content(
            model=name,
            contents="ok",
            config={"max_output_tokens": 1, "temperature": 0.0},
        )
    except genai_errors.ClientError as e:
        if e.code == 404:
            return "NO_ACCESS"
        return f"ERROR_{e.code}"
    except genai_errors.ServerError as e:
        return f"ERROR_{e.code}"
    except Exception as e:
        return f"ERROR ({type(e).__name__})"
    return "OK"


print(
    f"Probing Gemini publisher models in "
    f"{settings.google_cloud_project}/{args.region}\n"
    f"(each model gets one 1-token generateContent call)\n"
)

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(zip(gemini_names, ex.map(_probe, gemini_names), strict=True))

# Sort: OK first (alphabetical), then everything else
results.sort(key=lambda r: (r[1] != "OK", r[1], r[0]))

for name, status in results:
    print(f"  {name:<45} [{status}]")
