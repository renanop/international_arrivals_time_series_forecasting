import unicodedata
import Levenshtein

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

def levenshtein_substitute(word, cats=["aereo", "maritimo","fluvial", "terrestre"]):
    """Returns the most similar string in {cats} to {word} according to the levenshtein distance."""
    distances={}
    for cat in cats:
        distances[cat] = Levenshtein.distance(word, cat)

    min_distance = min(distances.values())

    for k,v in distances.items():
        if v == min_distance:
            return k