import unicodedata

def normalize(s: str) -> str:
    """lower, sem acento, strip, espaços->underscore, remove chars estranhos."""
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    # separadores e pontuações comuns viram underscore
    for ch in [" ", "-", ".", "/", "\\", "(", ")", "[", "]"]:
        s = s.replace(ch, "_")
    # colapsa underscores duplicados
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

