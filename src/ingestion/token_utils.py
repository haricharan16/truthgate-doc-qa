def count_tokens(text: str) -> int:
    """Estimate token count. ~4 chars per token for English text."""
    return max(1, len(text) // 4)

def encode(text: str):
    """Return a list of pseudo-tokens (words) for splitting purposes."""
    return text.split()

def decode(tokens) -> str:
    return " ".join(tokens)
