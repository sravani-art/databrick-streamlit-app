import json
import os
import threading

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")
_lock = threading.Lock()

_DEFAULTS = {
    "OPENAI_API_KEY": "",
    "OPENAI_MODEL": "gpt-4o-mini",
    "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
    "RAG_API_KEY": "rag-SXjlAnFAdvROAPlid4M9dQMrOoAyjoplYxNZF63mzo",
    "MAX_RETRIEVALS": 6,
    "SYSTEM_PROMPT": "You are an assistant that answers questions about process maps represented in JSON files.",
    "TEMPERATURE": 0.2,
    "CHUNK_SIZE": 800,
    "CHUNK_OVERLAP": 100,
    "SHOW_REFERENCES": "Yes",
    ################# bug 1065 ########################
    "INTENT_PROMPT": "Classify user input for a process map chatbot into one of three labels: edit (modify existing process map), create (generate a new process map), or qa (ask questions). Return ONLY one word: edit, create, or qa. Examples: 'Add approval step' -> edit, 'Delete step 3' -> edit, 'Rename review to validation' -> edit, 'Create a new onboarding process' -> create, 'Generate a payment workflow' -> create, 'What is step 2?' -> qa, 'Explain this process' -> qa.",
    "COMMAND_PROMPT": "Convert user instruction into structured JSON command for process map editing. Supported actions: add_node, delete_node, rename_node, connect_nodes. Return only JSON.",
    "CLASSIFIER_SYSTEM": "You are a strict query classifier for a process map assistant.\n\nYour only job is to decide if a user query is relevant to process maps, business workflows, or process documentation.\n\nRespond with EXACTLY one word \u2014 either ALLOWED or BLOCKED.\n\nALLOWED queries are about:\n- Requests to create, build, generate, edit or document a process map or workflow\n- How-to or step-by-step questions about a process that may be loaded in the session\n- Questions about steps, phases, decision points, or flow of a business process\n- Roles, actors, systems, documents, or references in a process\n- Process improvement, analysis, or clarification questions\n- Queries referencing uploaded documents or context (\"as per the doc\", \"based on this\", etc.)\n- Any question that could reasonably relate to the loaded process map described below\n- Layman or conversational questions referring to the process, even if not using technical terms\n- Questions like:\n  \"what happens next\"\n  \"who does this\"\n  \"what is going on here\"\n  \"what do we do next\"\n- General explanation requests such as:\n  \"explain this process\"\n  \"describe this process\"\n  \"give me an overview of the process\"\n  \"summarise the process\"\n  \nBLOCKED queries are about:\n- Arithmetic, mathematics, calculations\n- Politics, elections, government, current events\n- General knowledge, trivia, history, geography unrelated to any process\n- Personal advice, jokes, stories, poems\n\n{process_context}\n\nReply ALLOWED or BLOCKED only. No explanation."
    ################# bug 1065 ########################
}

def load_config():
    """Load configuration from disk (create if missing)."""
    if not os.path.exists(CONFIG_PATH):
        save_config(_DEFAULTS)
        return _DEFAULTS.copy()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = _DEFAULTS.copy()
    merged.update(data or {})
    return merged

def save_config(data):
    """Persist configuration safely."""
    with _lock:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

def update_config(changes: dict):
    """Update only specified keys."""
    cfg = load_config()
    for k, v in changes.items():
        if k in _DEFAULTS:
            cfg[k] = v
    save_config(cfg)
    return cfg
