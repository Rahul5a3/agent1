from __future__ import annotations

import re
import spacy


nlp = spacy.load("en_core_web_sm")


STOP_WORDS = {
    "find",
    "paper",
    "papers",
    "using",
    "use",
    "based",
    "approach",
    "approaches",
    "method",
    "methods",
    "study",
    "studies",
    "analysis",
    "system",
    "systems",
    "show",
    "give",
    "write",
    "generate",
    "what",
    "exists",
}


TERM_EXPANSIONS = {
    "llm": "large language model",
    "llms": "large language models",
    "nlp": "natural language processing",
    "cv": "computer vision",
}


def clean_phrase(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Strip leading articles that spaCy includes in noun chunks
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return text


def is_useful_phrase(text: str) -> bool:
    if not text:
        return False
    if len(text) < 3:
        return False
    if text in STOP_WORDS:
        return False
    if len(text.split()) > 5:
        return False
    # Reject tokens that look like two words merged without a space
    # e.g. "humevolution", "claimverification"
    if len(text) > 14 and " " not in text:
        return False
    return True


def extract_keywords(query: str) -> list[str]:
    """
    General-purpose keyword extractor using spaCy.
    Priority:
    1. noun chunks
    2. standalone meaningful tokens not already covered by phrases
    3. simple term expansion
    """
    doc = nlp(query)

    phrase_candidates: list[str] = []
    token_candidates: list[str] = []

    # 1. Noun chunks
    for chunk in doc.noun_chunks:
        phrase = clean_phrase(chunk.text)
        if is_useful_phrase(phrase):
            phrase_candidates.append(phrase)

    # Words already covered by extracted phrases
    covered_words = set()
    for phrase in phrase_candidates:
        for word in phrase.split():
            covered_words.add(word)

    # 2. Standalone technical/content tokens
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue

        term = clean_phrase(token.text)
        term = TERM_EXPANSIONS.get(term, term)

        if not is_useful_phrase(term):
            continue

        # keep only alphabetic-ish content terms
        if not any(ch.isalpha() for ch in term):
            continue

        if term in covered_words:
            continue

        token_candidates.append(term)

    # 3. Expanded abbreviations
    query_lower = query.lower()
    for short_form, expanded in TERM_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(short_form)}\b", query_lower):
            if expanded not in phrase_candidates:
                phrase_candidates.append(expanded)

    candidates = phrase_candidates + token_candidates

    # Remove duplicates while preserving order
    seen = set()
    final_keywords: list[str] = []

    for item in candidates:
        if item not in seen:
            final_keywords.append(item)
            seen.add(item)

    return final_keywords