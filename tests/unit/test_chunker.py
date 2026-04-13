"""Unit tests for app.chat._chunk_for_retrieval."""

import pytest

from app.chat import _chunk_for_retrieval


class TestChunker:
    def test_empty_body(self):
        assert _chunk_for_retrieval("") == []
        assert _chunk_for_retrieval("   ") == []

    def test_small_body_stays_whole(self):
        body = "Short paragraph.\n\nAnother line."
        chunks = _chunk_for_retrieval(body, max_chars=2000)
        assert len(chunks) == 1
        assert "Short paragraph." in chunks[0]

    def test_h2_sections_split_into_chunks(self):
        body = (
            "## Section A\n\nContent A paragraph one.\n\n"
            "## Section B\n\nContent B paragraph.\n\n"
            "## Section C\n\nContent C paragraph."
        )
        chunks = _chunk_for_retrieval(body, max_chars=2000, overlap_chars=0)
        assert len(chunks) == 3
        assert "Section A" in chunks[0]
        assert "Section B" in chunks[1]
        assert "Section C" in chunks[2]

    def test_section_title_stays_with_body(self):
        body = "## Intro\n\nBody goes here."
        chunks = _chunk_for_retrieval(body, overlap_chars=0)
        assert chunks == ["## Intro\n\nBody goes here."]

    def test_section_exceeding_max_chars_paragraph_packs(self):
        para = "This is a paragraph sentence. " * 20  # ~600 chars
        body = f"## Big\n\n{para}\n\n{para}\n\n{para}\n\n{para}"
        chunks = _chunk_for_retrieval(body, max_chars=800, overlap_chars=0)
        assert len(chunks) >= 2
        # Every chunk should belong to the same section (no spurious "##")
        assert all("##" not in c.splitlines()[1:] for c in chunks)

    def test_huge_paragraph_hard_split(self):
        # A single 5000-char blob with no paragraph breaks (e.g. a code block).
        body = "## Code\n\n" + "X" * 5000
        chunks = _chunk_for_retrieval(body, max_chars=1000, overlap_chars=0)
        # Should produce at least 5 chunks via hard split.
        assert len(chunks) >= 5

    def test_overlap_prefix_added(self):
        body = (
            "## A\n\n" + "word " * 200 + "\n\n"
            "## B\n\n" + "word " * 200
        )
        with_overlap = _chunk_for_retrieval(body, max_chars=1200, overlap_chars=100)
        without_overlap = _chunk_for_retrieval(body, max_chars=1200, overlap_chars=0)
        assert len(with_overlap) == len(without_overlap)
        # First chunk never has an overlap prefix.
        assert with_overlap[0] == without_overlap[0]
        # Later chunks are longer because of the prefix tail.
        if len(with_overlap) > 1:
            assert len(with_overlap[1]) > len(without_overlap[1])

    def test_no_section_headers_still_chunks(self):
        body = ("Paragraph one is short.\n\n" * 50)
        chunks = _chunk_for_retrieval(body, max_chars=500, overlap_chars=0)
        assert len(chunks) > 1
        assert all(len(c) <= 500 for c in chunks)

    def test_single_section_single_chunk_no_overlap(self):
        body = "## Only\n\nOne short body."
        chunks = _chunk_for_retrieval(body, overlap_chars=200)
        assert len(chunks) == 1
        assert not chunks[0].startswith("body.")
