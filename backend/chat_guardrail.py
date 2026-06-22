################## BUG 981 ############################
import re
import asyncio
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# LAYER 1 — Fast pattern guard (no LLM cost)
# Catches obviously off-topic queries before any API call is made.
# ─────────────────────────────────────────────────────────────────

_BLOCKED_PATTERNS = [
    # Arithmetic / maths
    r'\b\d+\s*[\+\-\*\/\^]\s*\d+\b',           # e.g. "12 + 5", "8 * 3"
    # r'\b(calculate|compute|solve|integral|derivative|equation|algebra|calculus|geometry|trigonometry)\b',

    # Politics & current affairs
    r'\b(election|parliament|congress|senate|president|prime minister|political party|democrat|republican|labour|conservative|vote|referendum|geopolit)\b',

    # General knowledge / trivia
    # r'\b(capital city|population of|who invented|history of|world war|famous person|celebrity|actor|musician|sport|football|cricket|recipe|weather|stock price|currency|cryptocurrency|bitcoin)\b',
    
    ###############TASK 999 #######################
    r'\b(capital city|population of|world war|famous person|celebrity|actor|musician|sport|football|cricket|recipe|weather|stock price|currency|cryptocurrency|bitcoin)\b',
    ################## TASK 999 ####################

    # Personal / social
    r'\b(joke|tell me a story|poem|write a song|relationship advice|mental health|medical advice|diagnos|prescri)\b',

    # Competitor / unrelated AI
    r'\b(chatgpt|gemini|copilot|llama|claude|grok|bard)\b',
]

_BLOCKED_RE = re.compile(
    "|".join(_BLOCKED_PATTERNS),
    re.IGNORECASE
)

MATH_KEYWORDS = [
    "calculate", "compute", "solve",
    "integral", "derivative", "equation",
    "algebra", "calculus", "geometry", "trigonometry"
]

def is_math_query(query: str) -> bool:
    q = query.lower().strip()

    # Match full words only
    words = re.findall(r'\b\w+\b', q)

    return any(word in MATH_KEYWORDS for word in words) or bool(
        re.search(r'\b\d+\s*[\+\-\*\/\^]\s*\d+\b', q)
    )

def is_summary_query(query: str) -> bool:
    q = query.lower()

    summary_keywords = [
        "summary", "summarize", "summarise",
        "overview", "describe", "explain this process",
        "explain the process", "process summary"
    ]

    return any(keyword in q for keyword in summary_keywords)

#################### TASK 1029 ##########################
def is_analytical_query(query: str) -> bool:
    q = query.lower()

    analytical_keywords = [
        "bottleneck",
        "optimize",
        "optimization",
        "delay",
        "constraint",
        "risk",
        "dependency",
        "inefficiency",
        "improvement",
        "downstream impact",
        "demand variability",
        "manual effort",
        "rework",
        "slow step",
        "operational issue"
    ]

    return any(keyword in q for keyword in analytical_keywords)
#################### TASK 1029 ##########################

def pattern_guard(query: str) -> bool:
    """Returns True if the query is blocked by pattern matching."""
    return bool(_BLOCKED_RE.search(query))

############### TASK 999 #######################
def allowlist_guard(query: str) -> bool:
    q = query.lower()

    keywords = [
        "step", "process", "workflow", "stage",
        "next", "after", "before",
        "who", "what", "where", "how",
        "happens", "done", "goes", "works",
        "this", "that"
    ]

    return any(word in q for word in keywords)

###############TASK 999 #######################

# ─────────────────────────────────────────────────────────────────
# LAYER 2 — LLM intent classifier (gpt-4o-mini, ~50 tokens)
# Only reached if Layer 1 passes. Catches nuanced off-topic queries.
# ─────────────────────────────────────────────────────────────────

# NOTE: the classifier prompt is currently moved to cofig for easier editing and tuning without code changes, but here is the original version for reference:
# _CLASSIFIER_SYSTEM = """You are a strict query classifier for a process map assistant.

# Your only job is to decide if a user query is relevant to process maps, business workflows, or process documentation.

# Respond with EXACTLY one word — either ALLOWED or BLOCKED.

# ALLOWED queries are about:
# - Requests to create, build, generate, edit or document a process map or workflow
# - How-to or step-by-step questions about a process that may be loaded in the session
# - Questions about steps, phases, decision points, or flow of a business process
# - Roles, actors, systems, documents, or references in a process
# - Process improvement, analysis, or clarification questions
# - Queries referencing uploaded documents or context ("as per the doc", "based on this", etc.)
# - Any question that could reasonably relate to the loaded process map described below
# - Layman or conversational questions referring to the process, even if not using technical terms
# - Questions like:
#   "what happens next"
#   "who does this"
#   "what is going on here"
#   "what do we do next"
# - General explanation requests such as:
#   "explain this process"
#   "describe this process"
#   "give me an overview of the process"
#   "summarise the process"
  
# BLOCKED queries are about:
# - Arithmetic, mathematics, calculations
# - Politics, elections, government, current events
# - General knowledge, trivia, history, geography unrelated to any process
# - Personal advice, jokes, stories, poems

# {process_context}

# Reply ALLOWED or BLOCKED only. No explanation."""


async def llm_intent_guard(query: str, client: OpenAI, model: str = "gpt-4o-mini",process_summary: str = "", CLASSIFIER_SYSTEM: str = "") -> bool:
    """
    Returns True if the query is BLOCKED.
    Uses a cheap model to keep cost minimal (~50 tokens per call).
    """
    process_context = (
        f"The process map currently loaded in this session is about: {process_summary}"
        if process_summary
        else "No specific process map context is available."
    )
    system_prompt = CLASSIFIER_SYSTEM.format(process_context=process_context)
    
    try:
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": query}
                ],
                temperature=0,
                max_tokens=5   # Only needs 1 word
            )
        )
        verdict = response.choices[0].message.content.strip().upper()
        logger.info("Guardrail verdict for query '%s': %s", query[:80], verdict)
        return verdict == "BLOCKED"

    except Exception as e:
        # If classifier fails, fail open (allow the query through)
        # so a guardrail outage doesn't break the whole chat
        logger.warning("Guardrail LLM call failed: %s — failing open", e)
        return False


# ─────────────────────────────────────────────────────────────────
# REFUSAL MESSAGES — polite, role-consistent responses
# ─────────────────────────────────────────────────────────────────

_REFUSAL_MESSAGES = {
    "default": (
        "I'm a process map assistant and can only answer questions about "
        "the process maps and workflows loaded into this session. "
        "Your question appears to be outside that scope. "
        "Try asking about specific steps, decision points, phases, or documents in the process."
    ),
    "maths": (
        "I can't help with calculations or mathematical problems."
        "I'm specialised for process map analysis only."
    ),
    "politics": (
        "I'm not able to answer political or current affairs questions. "
        "I can only assist with questions about your process maps and business workflows."
    ),
    "unsupported": (
        "That type of question is outside my area of expertise. "
        "I'm here to help with process maps and workflows, so please ask something related to that."
    )
}


def get_refusal_message(query: str) -> str:
    """Return a contextually appropriate refusal message."""
    q = query.lower()
    if re.search(r'\b\d+\s*[\+\-\*\/]\s*\d+\b|calculat|mathemat', q):
        return _REFUSAL_MESSAGES["maths"]
    if re.search(r'politic|election|parliament|president', q):
        return _REFUSAL_MESSAGES["politics"]
    if re.search(r'joke|story|poem|song', q):
        return _REFUSAL_MESSAGES["unsupported"]
    return _REFUSAL_MESSAGES["default"]


# ─────────────────────────────────────────────────────────────────
# COMBINED GUARD — single entry point used by the endpoint
# ─────────────────────────────────────────────────────────────────

async def run_guardrails(query: str, client: OpenAI, model: str = "gpt-4o-mini", process_summary: str = "", CLASSIFIER_SYSTEM: str = "" ) -> dict:
    """
    Run both guardrail layers.

    Returns:
        {"blocked": False}                          — query is allowed
        {"blocked": True, "message": "..."}         — query is blocked
    """
    #################### TASK 1029 ##########################
    if is_summary_query(query) or is_analytical_query(query):
        return {"blocked": False}
    #################### TASK 1029 ##########################
    
    if is_math_query(query):
        return {
            "blocked": True,
            "message": get_refusal_message(query)
        }
    pattern_flag = pattern_guard(query)
    allow_flag = allowlist_guard(query)

    # Allowlist overrides pattern
    if allow_flag:
        pattern_flag = False

    # Always check LLM (single call only)
    is_blocked = await llm_intent_guard(query, client, model, process_summary, CLASSIFIER_SYSTEM)

    # Final decision
    if pattern_flag and is_blocked:
        logger.info("Query blocked (pattern + LLM): '%s'", query[:80])
        return {"blocked": True, "message": get_refusal_message(query)}

    if is_blocked:
        logger.info("Query blocked by LLM: '%s'", query[:80])
        return {"blocked": True, "message": get_refusal_message(query)}

    # ─────────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────────
    ######################Task 999 #######################
    query_type = "layman" if allow_flag else "technical"

    logger.info(
        "Guardrail Decision | query='%s' | pattern=%s | allowlist=%s | blocked=%s | type=%s",
        query[:80],
        pattern_flag,
        allow_flag,
        is_blocked,
        query_type
    )
    ######################Task 999 #######################

    return {"blocked": False}

################## BUG 981 ############################
