"""
flowchart_generator.py
----------------------
Converts natural-language process descriptions (or uploaded documents)
into the exact JSON schema consumed by flowChartMaker.

All shape types, edge types, colours, and generation rules are defined in
flowchart_schema_config.py — edit that file to extend the schema; no changes
needed here.
"""

import json
import re

import asyncio
from typing import Optional
from openai import OpenAI

from backend.flowchart_schema_config import (
    CANVAS,
    NODE_DEFAULTS,
    EDGE_COLORS,
    GENERATION_SYSTEM_PROMPT,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (derived from config — do not hard-code here)
# ─────────────────────────────────────────────────────────────────────────────

ORIGIN_X: int = 300   # diagram starts at ~300 px from canvas left edge
ORIGIN_Y: int = 200   # diagram starts at ~200 px from canvas top edge

H_GAP: int = 280   # horizontal spacing between node centres
V_GAP: int = 180   # vertical spacing when branching downward

EDGE_DEFAULT_COLOR: str = "#666666"
YES_EDGE_COLOR:     str = "#4caf50"
NO_EDGE_COLOR:      str = "#f44336"

# Maximum nodes to request in a single LLM call.
# Above this threshold the description is split into sections.
CHUNK_NODE_LIMIT: int = 25

# Hard cap on output tokens per LLM generation call.
# 25 nodes x ~200 tokens/node JSON = ~5 000 tokens; 8 000 gives headroom.
MAX_TOKENS_PER_CHUNK: int = 8_000

# Hard cap for single-call (small) generations.
MAX_TOKENS_SINGLE: int = 8_000

# ─────────────────────────────────────────────────────────────────────────────
# NODE / EDGE SCHEMA BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_node(node_id: str, label: str, node_type: str,
               x: int = 0, y: int = 0,
               description: str = "", notes: str = "",
               color: Optional[str] = None) -> dict:
    """
    Return a fully-populated node dict that matches flowchart.json schema exactly.
    x and y are placeholders; _auto_layout() will overwrite them.
    """
    dims = NODE_DEFAULTS.get(node_type, NODE_DEFAULTS["process"])

    return {
        "id":          node_id,
        "label":       label,
        "type":        node_type,
        "x":           x,
        "y":           y,
        "width":       dims["width"],
        "height":      dims["height"],
        "description": description,
        "notes":       notes,
        "documentUrls":   [],
        "color":       color,
        "labelColor":  "#000000",
        "inputSettings":  [],
        "outputSettings": [],
        "fontSize":     "19px",
        "fontFamily":   "Arial, sans-serif",
        "fontStyle":    {"bold": False, "italic": False, "underline": False},
        "isLocked":     False,
        "subProcess":   [],
        "isHidden":     False,
        "attachments":  [],
    }


def _make_edge(edge_id: str, source: str, target: str,
               label: str = "",
               source_anchor: str = "Right",
               target_anchor: str = "Left",
               edge_type: str = "directional",
               color: str = EDGE_DEFAULT_COLOR,
               thickness: int = 2,
               label_x: int = 0,
               label_y: int = 0) -> dict:
    """Return a fully-populated edge dict matching flowchart.json schema."""
    return {
        "id":           edge_id,
        "source":       source,
        "target":       target,
        "label":        label,
        "labelX":       label_x,
        "labelY":       label_y,
        "sourceAnchor": source_anchor,
        "targetAnchor": target_anchor,
        "type":         edge_type,
        "thickness":    thickness,
        "color":        color,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-LAYOUT  (topological left-to-right walk)
# ─────────────────────────────────────────────────────────────────────────────

def _auto_layout(nodes: list[dict], edges: list[dict]) -> None:
    """
    Assigns x/y to every node using a topological walk from root nodes.
    Diamond branches split downward for the second outgoing edge.
    Falls back to a row-wrap grid for disconnected subgraphs.
    After positioning, label midpoints are calculated from node centres.
    """
    adj: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        if e["source"] in adj:
            adj[e["source"]].append(e["target"])

    id_to_node: dict[str, dict] = {n["id"]: n for n in nodes}

    # Nodes with no incoming edge are roots
    targets = {e["target"] for e in edges}
    roots   = [n["id"] for n in nodes if n["id"] not in targets]
    if not roots:
        roots = [nodes[0]["id"]]

    visited:   dict[str, bool] = {}
    col_usage: dict[int, int]  = {}   # col → highest row used so far

    def place(nid: str, col: int, row: int) -> None:
        if nid in visited:
            return
        visited[nid] = True
        node = id_to_node.get(nid)
        if not node:
            return

        # Avoid overlapping with siblings already in this column
        while col_usage.get(col, -1) >= row:
            row += 1
        col_usage[col] = row

        node["x"] = ORIGIN_X + col * H_GAP
        node["y"] = ORIGIN_Y + row * V_GAP

        children = adj.get(nid, [])
        for i, child in enumerate(children):
            # First child: continue horizontally; extra children: offset downward
            place(child, col + 1, row + i * 2)

    for root in roots:
        place(root, 0, 0)

    # Grid fallback for nodes not reached by the walk (disconnected)
    unvisited = [n for n in nodes if n["id"] not in visited]
    for i, node in enumerate(unvisited):
        node["x"] = ORIGIN_X + (i % 8) * H_GAP
        node["y"] = ORIGIN_Y + (i // 8 + 12) * V_GAP

    # Update edge label positions to the geometric midpoint of source→target
    for edge in edges:
        src = id_to_node.get(edge["source"])
        tgt = id_to_node.get(edge["target"])
        if src and tgt:
            edge["labelX"] = (src["x"] + tgt["x"]) // 2
            edge["labelY"] = (src["y"] + tgt["y"]) // 2


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL + RESPONSE PARSING
# ─────────────────────────────────────────────────────────────────────────────

async def generate_flowchart_from_text(
    text: str,
    openai_client: OpenAI,
    model: str = "gpt-4o",
    temperature: float = 0.2,
) -> dict:
    """
    Takes a natural-language process description and returns a complete
    flowchart JSON ready to be loaded into flowChartMaker.

    Returns one of:
      {"status": "complete",  "flowchart": {"nodes": [...], "edges": [...]}}
      {"status": "partial",   "flowchart": {"nodes": [...], "edges": [...]}}
      {"status": "ambiguous", "clarification": "<question for the user>"}
      {"status": "error",     "message": "<error detail>"}
    """
    try:
        # ── Normalise input ───────────────────────────────────────────────────
        # Callers may pass a plain string or a dict {"query": ..., "file_context": ...}
        if isinstance(text, dict):
            query        = (text.get("query") or "").strip()
            file_context = (text.get("file_context") or "").strip()
            user_content = f"{query}\n\n{file_context}".strip() if file_context else query
        else:
            user_content = str(text).strip()

        # ── Call the LLM ──────────────────────────────────────────────────────
        messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        completion = await asyncio.to_thread(
            lambda: openai_client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
        )

        raw = completion.choices[0].message.content.strip()
        # Strip any accidental markdown fences the model may produce
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\n?```$",        "", raw)

        parsed = json.loads(raw)

        # ── Ambiguous response ────────────────────────────────────────────────
        if parsed.get("ambiguous"):
            return {
                "status": "ambiguous",
                "clarification": parsed.get(
                    "clarification",
                    "Could you describe the process steps in more detail?"
                ),
            }

        raw_nodes: list[dict] = parsed.get("nodes", [])
        raw_edges: list[dict] = parsed.get("edges", [])

        if not raw_nodes:
            return {
                "status": "ambiguous",
                "clarification": (
                    "I couldn't identify any concrete process steps. "
                    "Could you describe what happens at each stage?"
                ),
            }

        # ── Enrich nodes (fill in all schema fields) ──────────────────────────
        enriched_nodes: list[dict] = []
        for n in raw_nodes:
            node = _make_node(
                node_id=n.get("id", f"node-{len(enriched_nodes)+1}"),
                label=n.get("label", ""),
                node_type=n.get("type", "process"),
                description=n.get("description", ""),
                notes=n.get("notes", ""),
                color=n.get("color"),
            )
            enriched_nodes.append(node)

        # ── Enrich edges (fill in all schema fields) ──────────────────────────
        enriched_edges: list[dict] = []
        for e in raw_edges:
            label = (e.get("label") or "").strip()
            upper = label.upper()

            # Colour + thickness driven by YES/NO convention
            if upper == "YES":
                color, thickness = YES_EDGE_COLOR, 3
            elif upper == "NO":
                color, thickness = NO_EDGE_COLOR, 3
            else:
                color     = e.get("color", EDGE_DEFAULT_COLOR)
                thickness = e.get("thickness", 2)

            enriched_edges.append(
                _make_edge(
                    edge_id=e.get("id", f"edge-{len(enriched_edges)+1}"),
                    source=e.get("source", ""),
                    target=e.get("target", ""),
                    label=label,
                    source_anchor=e.get("sourceAnchor", "Right"),
                    target_anchor=e.get("targetAnchor", "Left"),
                    edge_type=e.get("type", "directional"),
                    color=color,
                    thickness=thickness,
                )
            )

        # ── Auto-layout ───────────────────────────────────────────────────────
        _auto_layout(enriched_nodes, enriched_edges)

        # ── Determine completeness ────────────────────────────────────────────
        # Trust the model's own verdict first; fall back to heuristic.
        model_completeness = parsed.get("completeness", "")
        if model_completeness in ("complete", "partial"):
            status = model_completeness
        else:
            type_set = {n["type"] for n in enriched_nodes}
            labels   = " ".join(n["label"].lower() for n in enriched_nodes)
            has_start = "terminator" in type_set
            has_end   = (
                "externalentity" in type_set
                or any(kw in labels for kw in ("end", "complete", "finish", "done"))
            )
            status = "complete" if (has_start and has_end) else "partial"

        return {
            "status": status,
            "flowchart": {
                "nodes":         enriched_nodes,
                "edges":         enriched_edges,
                "floatingTexts": parsed.get("floatingTexts", []),
            },
        }

    except json.JSONDecodeError as exc:
        return {
            "status": "error",
            "message": f"Could not parse LLM response as JSON: {exc}",
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }

#-------------------------------------------------------------
# ADDITIONAL FUNCTION TO CONVERT PROCESS SUMMARY TO STEPS
#-------------------------------------------------------------
async def convert_summary_to_steps(file_context: str, client):
    prompt = f"""
Convert the following process summary into explicit sequential steps.

Rules:
- Extract clear step-by-step actions
- Identify decisions explicitly
- Ensure minimum 3 steps
- Use format:

1. Step name
2. Step name
3. Decision: <question?>
   - YES → step
   - NO → step

Return ONLY steps.

Summary:
{file_context}
"""

    response = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
    )

    return response.choices[0].message.content.strip()
