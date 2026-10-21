import unittest

from chunking import (
    split_by_sentence_boundaries,
    split_fixed_size,
    split_into_sentences,
    split_long_sentences,
)


class ChunkingTests(unittest.TestCase):
    def test_fixed_size_chunks_use_word_overlap(self) -> None:
        chunks = split_fixed_size("one two three four five six seven", 4, 1)

        self.assertEqual(
            chunks,
            ["one two three four", "four five six seven"],
        )

    def test_sentence_chunks_preserve_boundaries_and_overlap(self) -> None:
        sentences = split_into_sentences("One two three. Four five six. Seven eight.")

        chunks = split_by_sentence_boundaries(sentences, 4, 2)

        self.assertEqual(
            chunks,
            ["One two three.", "two three. Four five six.", "five six. Seven eight."],
        )

    def test_semantic_chunks_split_at_low_similarity_boundaries(self) -> None:
        sentences = ["Cats purr.", "Kittens meow.", "Rain falls."]

        chunks = split_by_sentence_boundaries(
            sentences, 20, 0, adjacent_similarities=[0.8, 0.2]
        )

        self.assertEqual(chunks, ["Cats purr. Kittens meow.", "Rain falls."])

    def test_long_sentences_are_bounded_by_chunk_size(self) -> None:
        sentences = split_long_sentences(["one two three four five"], 2)

        self.assertEqual(sentences, ["one two", "three four", "five"])

    def test_empty_text_has_no_chunks(self) -> None:
        self.assertEqual(split_fixed_size("", 4, 1), [])
        self.assertEqual(split_into_sentences("  \n "), [])


if __name__ == "__main__":
    unittest.main()
