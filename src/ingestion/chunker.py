"""
Section-aware chunker.

Design decisions (explained in DECISIONS.md):
- Chunk by h2/h3 section boundaries, NOT fixed token windows
- Only split within a section if it exceeds MAX_TOKENS
- Overlap of 100 tokens when splitting long sections
- Each chunk carries full metadata for citation

Known failure: tables get split mid-row if they exceed MAX_TOKENS.
"""

import re
import hashlib
from dataclasses import dataclass, field, asdict

MAX_TOKENS = 1200     # max tokens per chunk
OVERLAP_TOKENS = 100  # overlap when splitting long sections


def count_tokens(text: str) -> int:
    """Approximate token count (~4 chars per token for English)."""
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    token_count: int
    source_url: str
    section_title: str
    page_title: str
    section_anchor: str
    chunk_index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        anchor = f"#{self.section_anchor}" if self.section_anchor else ""
        return f"{self.source_url}{anchor} — {self.section_title}"


def split_into_paragraphs(text: str) -> list:
    paragraphs = re.split(r"\n\s*\n|\n(?=[A-Z•\-\d])", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _approx_split_words(text: str, max_tokens: int) -> list:
    """Split text by words into chunks of max_tokens approximate tokens."""
    words = text.split()
    chars_per_chunk = max_tokens * 4
    chunks = []
    current_chars = 0
    current_words = []
    for w in words:
        if current_chars + len(w) > chars_per_chunk and current_words:
            chunks.append(" ".join(current_words))
            current_words = []
            current_chars = 0
        current_words.append(w)
        current_chars += len(w) + 1
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def chunk_section(section: dict) -> list:
    text = section["content"].strip()
    if not text:
        return []

    base_id = _make_id(
    section["url"],
    section.get("section_anchor", ""),
    text
    )
    token_count = count_tokens(text)

    if token_count <= MAX_TOKENS:
        return [Chunk(
            chunk_id=f"{base_id}_0",
            text=text,
            token_count=token_count,
            source_url=section["url"],
            section_title=section["title"],
            page_title=section.get("page_title", ""),
            section_anchor=section.get("section_anchor", ""),
            chunk_index=0,
        )]

    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_paras = []
    current_tokens = 0
    chunk_idx = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > MAX_TOKENS:
            if current_paras:
                chunks.append(_make_chunk(base_id, chunk_idx, "\n\n".join(current_paras), section, current_tokens))
                chunk_idx += 1
                current_paras = []
                current_tokens = 0
            for sub_text in _approx_split_words(para, MAX_TOKENS):
                chunks.append(_make_chunk(base_id, chunk_idx, sub_text, section, count_tokens(sub_text)))
                chunk_idx += 1
            continue

        if current_tokens + para_tokens > MAX_TOKENS:
            chunks.append(_make_chunk(base_id, chunk_idx, "\n\n".join(current_paras), section, current_tokens))
            chunk_idx += 1
            overlap_paras = current_paras[-1:] if current_paras else []
            current_paras = overlap_paras + [para]
            current_tokens = sum(count_tokens(p) for p in current_paras)
        else:
            current_paras.append(para)
            current_tokens += para_tokens

    if current_paras:
        chunks.append(_make_chunk(base_id, chunk_idx, "\n\n".join(current_paras), section, current_tokens))

    return chunks


def _make_id(url: str, anchor: str, title: str) -> str:
    return hashlib.md5(
        f"{url}#{anchor}#{title}".encode()
    ).hexdigest()[:12]


def _make_chunk(base_id, idx, text, section, tokens) -> Chunk:
    return Chunk(
        chunk_id=f"{base_id}_{idx}",
        text=text,
        token_count=tokens,
        source_url=section["url"],
        section_title=section["title"],
        page_title=section.get("page_title", ""),
        section_anchor=section.get("section_anchor", ""),
        chunk_index=idx,
    )


def chunk_sections(sections: list) -> list:
    all_chunks = []
    for section in sections:
        all_chunks.extend(chunk_section(section))
    return all_chunks
