####################### TASK 1021 #######################
from enum import Enum


class FallbackType(str, Enum):
    BLOCKED = "blocked"
    NO_DATA = "no_data"
    VAGUE = "vague"
    UNSUPPORTED = "unsupported"
    ERROR = "error"
    ANALYTICAL = "analytical"


def _get_suggestions(fallback_type: FallbackType, query: str) -> str:
    """
    Generate contextual suggestions based on failure type.
    """

    if fallback_type == FallbackType.BLOCKED:
        return (
            "Try asking about specific steps, activities, documents, or workflow details "
            "from the current process."
        )

    if fallback_type == FallbackType.NO_DATA:
        return (
            "I couldn’t find matching information. Try mentioning a specific step name, "
            "activity, or part of the process."
        )

    if fallback_type == FallbackType.VAGUE:
        return (
            "Your question is a bit unclear. Try rephrasing with more detail, for example:\n"
            "- Explain this process step by step\n"
            "- What happens after approval?\n"
            "- What documents are attached to this step?"
        )
    if fallback_type == FallbackType.ANALYTICAL:
        return (
            "Try providing more operational detail such as approvals, delays, "
            "manual reviews, dependencies, or workflow constraints to improve analytical reasoning."
        )

    if fallback_type == FallbackType.UNSUPPORTED:
        return (
            "This assistant focuses on process maps. Try asking about workflows, steps, "
            "or related documents."
        )

    return ""


def build_fallback_response(fallback_type: FallbackType, message: str, query: str = ""):
    suggestions = _get_suggestions(fallback_type, query)

    final_message = message
    if suggestions:
        final_message = f"{message}\n\n{suggestions}"

    return {
        "status": True,
        "message": final_message,
        "fallback_type": fallback_type,
        "blocked": fallback_type == FallbackType.BLOCKED,
        "isAutomatedProcess": False
    }
