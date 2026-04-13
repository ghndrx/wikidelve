"""
Topic type classification for adaptive article templates.

Detects topic type from the topic string and search results, then returns
the appropriate article structure template. This replaces the one-size-fits-all
synthesis prompt with topic-aware guidance.
"""

import re
import logging

logger = logging.getLogger("kb-service.topic_classifier")


# --- Topic type definitions --------------------------------------------------

TOPIC_TYPES = {
    "technical": {
        "description": "Software, APIs, frameworks, protocols, programming concepts",
        "structure": (
            "ARTICLE STRUCTURE (technical topic detected):\n"
            "- ## Executive Summary (what it is and why it matters)\n"
            "- ## Key Findings (organized by sub-topic)\n"
            "- ## How It Works (architecture, key concepts, data flow)\n"
            "- ## Code Examples (practical usage with language tags)\n"
            "- ## Comparison with Alternatives (only if sources support it)\n"
            "- ## Practical Recommendations (specific, actionable guidance)\n"
            "- ## Limitations\n"
            "- ## Key Takeaways (3-5 bullets)\n"
        ),
    },
    "geographic": {
        "description": "Cities, countries, islands, regions, landmarks, travel",
        "structure": (
            "ARTICLE STRUCTURE (geographic topic detected):\n"
            "- ## Executive Summary (what makes this place notable)\n"
            "- ## Key Findings:\n"
            "  - ### Why It Matters (significance, what draws people)\n"
            "  - ### Key Features (notable landmarks, natural features, culture)\n"
            "  - ### Practical Information (access, logistics, what to know)\n"
            "- ## Practical Recommendations (specific tips for visitors/researchers)\n"
            "- ## Limitations\n"
            "- ## Key Takeaways\n"
            "NOTE: Do NOT include exhaustive distance tables, electoral districts, "
            "or coordinate lists. Focus on what makes this place interesting and useful.\n"
        ),
    },
    "biographical": {
        "description": "People, organizations, companies, teams",
        "structure": (
            "ARTICLE STRUCTURE (biographical topic detected):\n"
            "- ## Executive Summary (who/what and why they matter)\n"
            "- ## Key Findings:\n"
            "  - ### Background and Context\n"
            "  - ### Major Contributions or Impact\n"
            "  - ### Current Status / Recent Developments\n"
            "- ## Practical Recommendations (how to engage, follow, or learn more)\n"
            "- ## Limitations\n"
            "- ## Key Takeaways\n"
        ),
    },
    "comparative": {
        "description": "X vs Y, comparisons, alternatives, choosing between options",
        "structure": (
            "ARTICLE STRUCTURE (comparative topic detected):\n"
            "- ## Executive Summary (the core tradeoff in 2-3 sentences)\n"
            "- ## Key Findings:\n"
            "  - ### Option A Overview\n"
            "  - ### Option B Overview\n"
            "  - ### Head-to-Head Comparison (table with SOURCED data only)\n"
            "  - ### When to Choose Each\n"
            "- ## Practical Recommendations (decision framework)\n"
            "- ## Limitations\n"
            "- ## Key Takeaways\n"
        ),
    },
    "conceptual": {
        "description": "Abstract concepts, theories, methodologies, best practices",
        "structure": (
            "ARTICLE STRUCTURE (conceptual topic detected):\n"
            "- ## Executive Summary (the concept and why it matters)\n"
            "- ## Key Findings:\n"
            "  - ### Core Concepts\n"
            "  - ### Current State of Practice\n"
            "  - ### Evidence and Research\n"
            "- ## Practical Recommendations\n"
            "- ## Limitations\n"
            "- ## Key Takeaways\n"
        ),
    },
}

# Default fallback
DEFAULT_STRUCTURE = (
    "ARTICLE STRUCTURE:\n"
    "- ## Executive Summary (2-3 sentences)\n"
    "- ## Key Findings (organized by sub-topic with ### headers)\n"
    "- ## Practical Recommendations (actionable bullet points)\n"
    "- ## Limitations (what the sources didn't cover)\n"
    "- ## Key Takeaways (3-5 bullets)\n"
)

# --- Classification patterns ------------------------------------------------

_GEOGRAPHIC_PATTERNS = re.compile(
    r'\b(city|town|island|country|region|state|province|mountain|river|lake|'
    r'beach|coast|bay|peninsula|continent|village|harbor|port|national park|'
    r'australia|queensland|victoria|nsw|tasmania|zealand|europe|asia|africa|'
    r'america|canada|mexico|brazil|japan|china|india|france|germany|italy|'
    r'spain|uk|england|scotland|ireland)\b',
    re.IGNORECASE,
)

_BIOGRAPHICAL_PATTERNS = re.compile(
    r'\b(who is|biography|founder|ceo|creator|inventor|author|musician|'
    r'artist|politician|scientist|engineer|company|organization|startup|'
    r'corporation|founded by)\b',
    re.IGNORECASE,
)

_COMPARATIVE_PATTERNS = re.compile(
    r'\b(vs\.?|versus|compared to|comparison|alternative|choosing between|'
    r'which is better|differences between|pros and cons)\b',
    re.IGNORECASE,
)

_TECHNICAL_PATTERNS = re.compile(
    r'\b(api|framework|library|protocol|algorithm|database|deployment|'
    r'kubernetes|docker|aws|terraform|python|javascript|typescript|rust|go|'
    r'react|svelte|fastapi|django|node|npm|git|cicd|devops|microservice|'
    r'encryption|authentication|oauth|jwt|webhook|websocket|graphql|rest|'
    r'llm|rag|embedding|vector|neural|machine learning|deep learning|'
    r'agent|prompt|langchain|langgraph)\b',
    re.IGNORECASE,
)


def classify_topic(topic: str, sources: list[dict] | None = None) -> str:
    """Classify a topic string into a topic type.

    Uses pattern matching on the topic text. If sources are provided,
    also checks source titles/content for additional signals.

    Returns one of: technical, geographic, biographical, comparative, conceptual
    """
    scores = {
        "technical": 0,
        "geographic": 0,
        "biographical": 0,
        "comparative": 0,
        "conceptual": 0,
    }

    # Score from topic text
    text = topic
    if sources:
        titles = " ".join(s.get("title", "") for s in sources[:10])
        text = f"{topic} {titles}"

    scores["geographic"] += len(_GEOGRAPHIC_PATTERNS.findall(text))
    scores["biographical"] += len(_BIOGRAPHICAL_PATTERNS.findall(text))
    scores["comparative"] += len(_COMPARATIVE_PATTERNS.findall(text)) * 2  # boost comparative signals
    scores["technical"] += len(_TECHNICAL_PATTERNS.findall(text))

    # If nothing matched strongly, default to conceptual
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "conceptual"

    return best


def get_article_template(topic_type: str) -> str:
    """Get the article structure template for a topic type."""
    entry = TOPIC_TYPES.get(topic_type)
    if entry:
        return entry["structure"]
    return DEFAULT_STRUCTURE
