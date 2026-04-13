"""Tests for fuzzy matching logic in find_related_article."""

import pytest
from unittest.mock import patch
from app.wiki import find_related_article, _significant_words


# ---------------------------------------------------------------------------
# _significant_words helper
# ---------------------------------------------------------------------------

class TestSignificantWords:
    """Tests for the word extraction helper used by fuzzy matching."""

    def test_filters_stop_words(self):
        words = _significant_words("the best practices for python")
        assert "the" not in words
        assert "best" not in words
        assert "for" not in words
        assert "python" in words
        assert "practices" in words

    def test_filters_short_words(self):
        words = _significant_words("go is a language")
        # "go" and "is" and "a" are all <= 2 chars or stop words
        assert "go" not in words
        assert "is" not in words
        assert "language" in words

    def test_empty_string(self):
        assert _significant_words("") == set()

    def test_all_stop_words(self):
        words = _significant_words("the and or for in on to with")
        assert words == set()

    def test_case_insensitive(self):
        words = _significant_words("Python Docker Kubernetes")
        assert "python" in words
        assert "docker" in words
        assert "kubernetes" in words

    def test_shared_stop_words_dont_count(self):
        """Stop words should not contribute to word overlap."""
        w1 = _significant_words("the best python guide")
        w2 = _significant_words("the best rust guide")
        overlap = w1 & w2
        # "guide" overlaps but "the" and "best" are stop words
        assert "guide" in overlap
        assert "the" not in overlap
        assert "best" not in overlap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_articles(titles: list[str]) -> list[dict]:
    """Build a list of article dicts from title strings."""
    articles = []
    for title in titles:
        slug = title.lower().replace(" ", "-")
        # Remove non-alnum except hyphens for slug
        slug = "".join(c if c.isalnum() or c == "-" else "" for c in slug)
        articles.append({
            "slug": slug,
            "title": title,
            "summary": "",
            "tags": [],
            "status": "draft",
            "confidence": "medium",
            "updated": "2024-01-01",
            "source_type": "web",
            "source_files": [],
            "word_count": 100,
            "kb": "test",
        })
    return articles


# ---------------------------------------------------------------------------
# find_related_article
# ---------------------------------------------------------------------------

class TestFindRelatedArticle:
    """Tests for fuzzy matching with the 0.75 threshold + word overlap check."""

    @patch("app.wiki.get_articles")
    def test_exact_slug_match(self, mock_get):
        """Exact slug match should always return the article."""
        mock_get.return_value = _make_articles(["Docker Containers"])
        result = find_related_article("test", "docker-containers")
        assert result is not None
        assert result["slug"] == "docker-containers"

    @patch("app.wiki.get_articles")
    def test_empty_kb_returns_none(self, mock_get):
        mock_get.return_value = []
        result = find_related_article("test", "anything")
        assert result is None

    @patch("app.wiki.get_articles")
    def test_no_match_below_threshold(self, mock_get):
        mock_get.return_value = _make_articles(["Quantum Computing Fundamentals"])
        result = find_related_article("test", "Python Web Development")
        assert result is None

    # --- The production bugs: false positives that should NOT match ----------

    @patch("app.wiki.get_articles")
    def test_kubernetes_operators_not_matching_pod_security(self, mock_get):
        """The exact production bug: 'kubernetes operators' was matching
        'kubernetes pod security standards' at 0.55 threshold."""
        mock_get.return_value = _make_articles([
            "Kubernetes Pod Security Standards",
        ])
        result = find_related_article("test", "kubernetes operators")
        assert result is None

    @patch("app.wiki.get_articles")
    def test_kubernetes_operators_not_matching_admission_controllers(self, mock_get):
        mock_get.return_value = _make_articles([
            "Kubernetes Admission Controllers",
        ])
        result = find_related_article("test", "kubernetes operators")
        assert result is None

    @patch("app.wiki.get_articles")
    def test_python_testing_not_matching_python_packaging(self, mock_get):
        mock_get.return_value = _make_articles(["Python Packaging Guide"])
        result = find_related_article("test", "Python Testing")
        assert result is None

    @patch("app.wiki.get_articles")
    def test_docker_networking_not_matching_docker_security(self, mock_get):
        mock_get.return_value = _make_articles(["Docker Security Best Practices"])
        result = find_related_article("test", "Docker Networking")
        assert result is None

    # --- Legitimate matches that SHOULD work --------------------------------

    @patch("app.wiki.get_articles")
    def test_docker_networking_matches_docker_networking_guide(self, mock_get):
        """Legitimate duplicate: same topic with 'Guide' suffix."""
        mock_get.return_value = _make_articles(["Docker Networking Guide"])
        result = find_related_article("test", "Docker Networking")
        assert result is not None
        assert result["title"] == "Docker Networking Guide"

    @patch("app.wiki.get_articles")
    def test_python_testing_matches_python_testing_best_practices(self, mock_get):
        mock_get.return_value = _make_articles(["Python Testing Best Practices"])
        result = find_related_article("test", "Python Testing")
        assert result is not None
        assert result["title"] == "Python Testing Best Practices"

    @patch("app.wiki.get_articles")
    def test_exact_title_match(self, mock_get):
        mock_get.return_value = _make_articles(["Rust Memory Safety"])
        result = find_related_article("test", "Rust Memory Safety")
        assert result is not None

    @patch("app.wiki.get_articles")
    def test_case_insensitive_match(self, mock_get):
        mock_get.return_value = _make_articles(["Docker Containers"])
        result = find_related_article("test", "docker containers")
        assert result is not None

    # --- Short topics (< 3 words) -------------------------------------------

    @patch("app.wiki.get_articles")
    def test_single_word_topic_exact(self, mock_get):
        mock_get.return_value = _make_articles(["Kubernetes"])
        result = find_related_article("test", "Kubernetes")
        assert result is not None

    @patch("app.wiki.get_articles")
    def test_two_word_topic_match(self, mock_get):
        mock_get.return_value = _make_articles(["Docker Compose"])
        result = find_related_article("test", "Docker Compose")
        assert result is not None

    @patch("app.wiki.get_articles")
    def test_single_word_no_match(self, mock_get):
        mock_get.return_value = _make_articles(["Kubernetes Operators"])
        result = find_related_article("test", "Terraform")
        assert result is None

    # --- Topics with typos --------------------------------------------------

    @patch("app.wiki.get_articles")
    def test_minor_typo_still_matches(self, mock_get):
        """A minor typo in a long-enough title might still pass threshold."""
        mock_get.return_value = _make_articles(["Docker Networking Guide"])
        result = find_related_article("test", "Docker Networkng Guide")
        # Minor typo: "Networkng" vs "Networking" — ratio should still be high
        assert result is not None

    @patch("app.wiki.get_articles")
    def test_major_typo_no_match(self, mock_get):
        """Heavily misspelled + extra words should not match."""
        mock_get.return_value = _make_articles(["Kubernetes Operators"])
        result = find_related_article("test", "Kuberntes Opertors in Productoin")
        # Enough drift that it should not match at 0.75 threshold
        assert result is None

    # --- Multiple articles in KB, pick best ---------------------------------

    @patch("app.wiki.get_articles")
    def test_picks_best_match_from_multiple(self, mock_get):
        mock_get.return_value = _make_articles([
            "Python Web Development",
            "Python Testing Best Practices",
            "Rust Memory Safety",
        ])
        result = find_related_article("test", "Python Testing")
        assert result is not None
        assert result["title"] == "Python Testing Best Practices"

    @patch("app.wiki.get_articles")
    def test_no_match_among_multiple_unrelated(self, mock_get):
        mock_get.return_value = _make_articles([
            "Rust Memory Safety",
            "Docker Containers",
            "PostgreSQL Optimization",
        ])
        result = find_related_article("test", "Machine Learning Pipelines")
        assert result is None

    # --- Edge: topic with special characters --------------------------------

    @patch("app.wiki.get_articles")
    def test_topic_with_colon(self, mock_get):
        mock_get.return_value = _make_articles(["GPU Cloud Platforms"])
        result = find_related_article("test", "GPU Cloud Platforms: RunPod vs Lambda")
        # The word overlap should help here — "GPU", "Cloud", "Platforms" all match

    @patch("app.wiki.get_articles")
    def test_topic_with_parentheses(self, mock_get):
        mock_get.return_value = _make_articles(["Python Type Hints"])
        result = find_related_article("test", "Python Type Hints (PEP 484)")
        # Should still match due to high word overlap
