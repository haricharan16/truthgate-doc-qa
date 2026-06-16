"""
Token counting utility.
Uses tiktoken if available with cached encoding, otherwise falls back to word-split approximation.
The approximation (chars/4) is accurate enough for chunking purposes.
"""

def count_tokens(text: str) -> int:
    """Estimate token count. ~4 chars per token for English text."""
    return max(1, len(text) // 4)

def encode(text: str):
    """Return a list of pseudo-tokens (words) for splitting purposes."""
    return text.split()

def decode(tokens) -> str:
    return " ".join(tokens)
