import re


SEMANTIC_SIMILARITY_THRESHOLD = 0.45


def split_fixed_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def split_into_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]


def split_long_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    result = []
    for sentence in sentences:
        words = sentence.split()
        result.extend(
            " ".join(words[start : start + chunk_size])
            for start in range(0, len(words), chunk_size)
        )
    return result


def split_by_sentence_boundaries(
    sentences: list[str],
    chunk_size: int,
    overlap: int,
    adjacent_similarities: list[float] | None = None,
) -> list[str]:
    if not sentences:
        return []

    groups: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for index, sentence in enumerate(sentences):
        sentence_words = sentence.split()
        semantic_break = (
            adjacent_similarities is not None
            and index > 0
            and adjacent_similarities[index - 1] < SEMANTIC_SIMILARITY_THRESHOLD
        )
        size_break = current and current_words + len(sentence_words) > chunk_size
        if current and (semantic_break or size_break):
            groups.append(current)
            current = []
            current_words = 0
        current.append(sentence)
        current_words += len(sentence_words)
    if current:
        groups.append(current)

    chunks = []
    previous_words: list[str] = []
    for group in groups:
        words = " ".join(group).split()
        prefix = previous_words[-overlap:] if overlap else []
        chunks.append(" ".join([*prefix, *words]))
        previous_words = words
    return chunks
