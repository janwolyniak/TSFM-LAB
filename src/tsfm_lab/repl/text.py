"""Display untrusted text without allowing terminal control sequences."""


def literal(value) -> str:
    """Keep ordinary Unicode/newlines and make C0, C1, and DEL controls visible."""
    named = {"\t": r"\t", "\r": r"\r", "\b": r"\b", "\f": r"\f", "\v": r"\v"}
    parts = []
    for character in str(value):
        code = ord(character)
        if character != "\n" and (code < 0x20 or 0x7F <= code <= 0x9F):
            parts.append(named.get(character, f"\\x{code:02x}"))
        else:
            parts.append(character)
    return "".join(parts)
