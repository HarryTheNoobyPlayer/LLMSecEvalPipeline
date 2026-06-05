"""Unit tests for the Data Loader module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from llmseceval.config import DataLoaderConfig
from llmseceval.data_loader.loader import DataLoader

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixture_dir() -> Path:
    """Path to the test fixtures directory containing sample_sharing.json."""
    assert (FIXTURES_DIR / "sample_sharing.json").exists(), (
        "Fixture missing, run `python tests/create_fixtures.py` first."
    )
    return FIXTURES_DIR


@pytest.fixture()
def default_config(fixture_dir: Path) -> DataLoaderConfig:
    """A DataLoaderConfig pointing at the fixtures directory."""
    return DataLoaderConfig(
        dataset_path=str(fixture_dir),
        language="en",
        min_char_length=10,
        deduplicate=True,
        sample_size=None,
        random_seed=42,
    )


@pytest.fixture()
def loader(default_config: DataLoaderConfig) -> DataLoader:
    return DataLoader(default_config)


# ---------------------------------------------------------------------------
# Step-level tests
# ---------------------------------------------------------------------------


class TestLoadDevGPTJSON:
    def test_loads_and_extracts_fixture(self, fixture_dir: Path) -> None:
        df = DataLoader._load_devgpt_json(fixture_dir)
        
        # Item 0 has 0, Item 1 has 1, Item 2 has 5, Item 3 has 1,
        # Item 4 has 3, Item 5 has 4 → total = 14
        assert len(df) == 14
        assert "prompt_text" in df.columns
        assert "url" in df.columns
        assert "source_file" in df.columns
        assert "original_index" in df.columns
        
        # Check an extracted URL
        assert "https://chatgpt.com/share/1" in df["url"].values

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            DataLoader._load_devgpt_json(tmp_path / "nonexistent")

    def test_no_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / "not_json.txt").write_text("hello")
        with pytest.raises(FileNotFoundError, match="No .json files"):
            DataLoader._load_devgpt_json(tmp_path)


class TestFilterEmpty:
    def test_removes_non_string_and_blank(self) -> None:
        df = pd.DataFrame({
            "prompt_text": ["valid text here", "", "   ", "\n\t", None, 42, "another valid"],
            "original_index": range(7),
            "source_file": ["f"] * 7,
        })
        result = DataLoader._filter_empty(df)
        assert len(result) == 2
        assert result["prompt_text"].tolist() == ["valid text here", "another valid"]


class TestFilterByLength:
    def test_removes_short_prompts(self) -> None:
        df = pd.DataFrame({
            "prompt_text": ["short", "a" * 9, "a" * 10, "this is long enough"],
            "original_index": range(4),
            "source_file": ["f"] * 4,
        })
        result = DataLoader._filter_by_length(df, min_chars=10)
        assert len(result) == 2
        assert all(len(t) >= 10 for t in result["prompt_text"])


class TestFilterByCodeGenPattern:
    """Verb/noun anchor filter, keeps prompts that ask for code, drops meta."""

    PATTERNS = [
        "write", "create", "implement", "build", "generate",
        "code", "function", "script", "class", "def", "algorithm", "program",
    ]

    def test_keeps_prompt_with_anchor(self) -> None:
        df = pd.DataFrame({
            "prompt_text": [
                "write a python function to sort a list",
                "fix typos in my resume",
            ],
            "original_index": range(2),
            "source_file": ["f"] * 2,
        })
        result = DataLoader._filter_by_code_gen_pattern(df, self.PATTERNS)
        assert result["prompt_text"].tolist() == ["write a python function to sort a list"]

    def test_drops_meta_prompt_with_no_anchor(self) -> None:
        df = pd.DataFrame({
            "prompt_text": [
                "Summarize my python project more briefly. Two bullet points.",
            ],
            "original_index": [0],
            "source_file": ["f"],
        })
        result = DataLoader._filter_by_code_gen_pattern(df, self.PATTERNS)
        assert len(result) == 0

    def test_whole_word_only(self) -> None:
        # "implementation" alone must NOT match "implement"; "classroom" must
        # NOT match "class". A bare verb in another prompt should still work.
        df = pd.DataFrame({
            "prompt_text": [
                "Review my implementation strategy for the classroom system",
                "write a parser",
            ],
            "original_index": range(2),
            "source_file": ["f"] * 2,
        })
        result = DataLoader._filter_by_code_gen_pattern(df, self.PATTERNS)
        assert result["prompt_text"].tolist() == ["write a parser"]

    def test_case_insensitive(self) -> None:
        df = pd.DataFrame({
            "prompt_text": ["WRITE A SCRIPT that prints hello"],
            "original_index": [0],
            "source_file": ["f"],
        })
        result = DataLoader._filter_by_code_gen_pattern(df, self.PATTERNS)
        assert len(result) == 1

    def test_empty_patterns_no_filtering(self) -> None:
        df = pd.DataFrame({
            "prompt_text": ["anything goes here"],
            "original_index": [0],
            "source_file": ["f"],
        })
        result = DataLoader._filter_by_code_gen_pattern(df, [])
        assert len(result) == 1

    def test_empty_df_returns_empty(self) -> None:
        df = pd.DataFrame(columns=["prompt_text", "original_index", "source_file"])
        result = DataLoader._filter_by_code_gen_pattern(df, self.PATTERNS)
        assert len(result) == 0


class TestFilterByLanguage:
    """Language filtering tests, mocks lingua to avoid heavy dependency in CI."""

    def test_keeps_english_only(self, loader: DataLoader) -> None:
        df = pd.DataFrame({
            "prompt_text": [
                "Write a Python function",
                "用Python写一个排序算法",
                "Implement binary search in Python",
            ],
            "original_index": range(3),
            "source_file": ["f"] * 3,
        })

        # Mock lingua: first and third are English, second is not
        mock_detector = MagicMock()
        mock_en = MagicMock()

        def detect_side_effect(text: str) -> MagicMock:
            if "Python" in text and "用" not in text:
                return mock_en
            return MagicMock()  # different object → not equal to target

        mock_detector.detect_language_of = detect_side_effect

        mock_builder_cls = MagicMock()
        mock_builder_instance = MagicMock()
        mock_builder_cls.from_languages.return_value = mock_builder_instance
        mock_builder_instance.with_low_accuracy_mode.return_value = mock_builder_instance
        mock_builder_instance.build.return_value = mock_detector

        with (
            patch("lingua.LanguageDetectorBuilder", mock_builder_cls, create=True),
            patch("llmseceval.data_loader.loader._get_lingua_language", return_value=mock_en),
        ):
            result = loader._filter_by_language(df, "en")

        assert len(result) == 2
        assert "用Python" not in result["prompt_text"].values

    def test_empty_df_returns_empty(self, loader: DataLoader) -> None:
        df = pd.DataFrame(columns=["prompt_text", "original_index", "source_file"])
        result = loader._filter_by_language(df, "en")
        assert len(result) == 0


class TestDeduplicate:
    def test_removes_exact_dupes(self) -> None:
        df = pd.DataFrame({
            "prompt_text": ["aaa bbb ccc", "ddd eee fff", "aaa bbb ccc"],
            "original_index": [0, 1, 2],
            "source_file": ["f"] * 3,
        })
        result = DataLoader._deduplicate(df)
        assert len(result) == 2
        # First occurrence kept
        assert result["original_index"].tolist() == [0, 1]


class TestAssignIds:
    def test_deterministic_ids(self) -> None:
        df = pd.DataFrame({
            "prompt_text": ["hello world", "foo bar"],
            "original_index": [0, 1],
            "source_file": ["f"] * 2,
        })
        result = DataLoader._assign_ids(df)
        assert "prompt_id" in result.columns

        expected_id = hashlib.sha256("hello world".encode("utf-8")).hexdigest()[:12]
        assert result.iloc[0]["prompt_id"] == expected_id

    def test_same_text_same_id(self) -> None:
        df = pd.DataFrame({
            "prompt_text": ["hello world", "hello world"],
            "original_index": [0, 1],
            "source_file": ["f"] * 2,
        })
        result = DataLoader._assign_ids(df)
        assert result.iloc[0]["prompt_id"] == result.iloc[1]["prompt_id"]


class TestSampling:
    def test_sample_correct_count(self) -> None:
        df = pd.DataFrame({
            "prompt_text": [f"prompt {i}" for i in range(20)],
            "original_index": range(20),
            "source_file": ["f"] * 20,
        })
        result = DataLoader._sample(df, n=5, seed=42)
        assert len(result) == 5

    def test_sample_reproducible(self) -> None:
        df = pd.DataFrame({
            "prompt_text": [f"prompt {i}" for i in range(20)],
            "original_index": range(20),
            "source_file": ["f"] * 20,
        })
        r1 = DataLoader._sample(df, n=5, seed=42)
        r2 = DataLoader._sample(df, n=5, seed=42)
        assert r1["prompt_text"].tolist() == r2["prompt_text"].tolist()

    def test_sample_larger_than_df(self) -> None:
        df = pd.DataFrame({
            "prompt_text": ["a", "b"],
            "original_index": [0, 1],
            "source_file": ["f"] * 2,
        })
        result = DataLoader._sample(df, n=100, seed=42)
        assert len(result) == 2  # returns all


# ---------------------------------------------------------------------------
# Integration-level tests (end-to-end with mocked language detection)
# ---------------------------------------------------------------------------

def _make_lingua_mocks():
    """Create a set of mocks that simulate lingua's LanguageDetectorBuilder.

    Returns (mock_builder_cls, mock_en) where mock_en is the language object
    returned by detect_language_of for all prompts (i.e. treats everything as
    English).
    """
    mock_detector = MagicMock()
    mock_en = MagicMock()
    mock_detector.detect_language_of.return_value = mock_en

    mock_builder_cls = MagicMock()
    mock_builder_instance = MagicMock()
    mock_builder_cls.from_languages.return_value = mock_builder_instance
    mock_builder_instance.with_low_accuracy_mode.return_value = mock_builder_instance
    mock_builder_instance.build.return_value = mock_detector

    return mock_builder_cls, mock_en


class TestDisableFilters:
    """Verify disable_filters bypasses opinionated filters."""

    def test_bypasses_keyword_codegen_and_language(self, fixture_dir: Path, tmp_path: Path) -> None:
        config = DataLoaderConfig(
            dataset_path=str(fixture_dir),
            language="en",
            min_char_length=10,
            required_keyword="python",
            required_code_gen_pattern=["write", "implement"],
            deduplicate=True,
            disable_filters=True,
            sample_size=None,
            random_seed=42,
        )
        loader = DataLoader(config)
        mock_builder_cls, mock_en = _make_lingua_mocks()

        with (
            patch("lingua.LanguageDetectorBuilder", mock_builder_cls, create=True),
            patch("llmseceval.data_loader.loader._get_lingua_language", return_value=mock_en),
        ):
            loader.run(output_dir=tmp_path)

        # Lingua's builder should NOT have been called when filters are bypassed.
        mock_builder_cls.from_languages.assert_not_called()

        report = json.loads((tmp_path / "data_loader_report.json").read_text(encoding="utf-8"))
        assert report["filters_bypassed"] is True
        assert report["filtered_by_keyword"] == 0
        assert report["filtered_by_code_gen_pattern"] == 0
        assert report["filtered_by_language"] == 0
        # Empty-string filter and length filter still apply (validity, not filtering).
        assert "filtered_empty" in report
        assert "filtered_by_length" in report

    def test_default_filters_enabled_keeps_filters_bypassed_false(
        self, fixture_dir: Path, tmp_path: Path,
    ) -> None:
        config = DataLoaderConfig(
            dataset_path=str(fixture_dir),
            language="en",
            min_char_length=10,
            required_keyword="python",
            deduplicate=True,
            sample_size=None,
            random_seed=42,
        )
        loader = DataLoader(config)
        mock_builder_cls, mock_en = _make_lingua_mocks()

        with (
            patch("lingua.LanguageDetectorBuilder", mock_builder_cls, create=True),
            patch("llmseceval.data_loader.loader._get_lingua_language", return_value=mock_en),
        ):
            loader.run(output_dir=tmp_path)

        report = json.loads((tmp_path / "data_loader_report.json").read_text(encoding="utf-8"))
        assert report["filters_bypassed"] is False


class TestFullRun:
    """End-to-end test of DataLoader.run() with mocked lingua."""

    def test_run_produces_outputs(self, fixture_dir: Path, tmp_path: Path) -> None:
        config = DataLoaderConfig(
            dataset_path=str(fixture_dir),
            language="en",
            min_char_length=10,
            deduplicate=True,
            sample_size=None,
            random_seed=42,
        )
        loader = DataLoader(config)
        mock_builder_cls, mock_en = _make_lingua_mocks()

        with (
            patch("lingua.LanguageDetectorBuilder", mock_builder_cls, create=True),
            patch("llmseceval.data_loader.loader._get_lingua_language", return_value=mock_en),
        ):
            jsonl_path = loader.run(output_dir=tmp_path)

        # Check output files exist
        assert jsonl_path.exists()
        assert (tmp_path / "data_loader_report.json").exists()

        # Check JSONL format
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        assert len(lines) > 0

        first = json.loads(lines[0])
        assert "prompt_id" in first
        assert "url" in first
        assert "text" in first
        assert "char_count" in first
        assert "source_file" in first
        assert len(first["prompt_id"]) == 12
        assert first["char_count"] == len(first["text"])

    def test_run_with_sampling(self, fixture_dir: Path, tmp_path: Path) -> None:
        config = DataLoaderConfig(
            dataset_path=str(fixture_dir),
            language="en",
            min_char_length=10,
            deduplicate=True,
            sample_size=3,
            random_seed=42,
        )
        loader = DataLoader(config)
        mock_builder_cls, mock_en = _make_lingua_mocks()

        with (
            patch("lingua.LanguageDetectorBuilder", mock_builder_cls, create=True),
            patch("llmseceval.data_loader.loader._get_lingua_language", return_value=mock_en),
        ):
            jsonl_path = loader.run(output_dir=tmp_path)

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        assert len(lines) == 3

    def test_report_has_correct_keys(self, fixture_dir: Path, tmp_path: Path) -> None:
        config = DataLoaderConfig(
            dataset_path=str(fixture_dir),
            language="en",
            min_char_length=10,
            deduplicate=True,
            sample_size=None,
            random_seed=42,
        )
        loader = DataLoader(config)
        mock_builder_cls, mock_en = _make_lingua_mocks()

        with (
            patch("lingua.LanguageDetectorBuilder", mock_builder_cls, create=True),
            patch("llmseceval.data_loader.loader._get_lingua_language", return_value=mock_en),
        ):
            loader.run(output_dir=tmp_path)

        with open(tmp_path / "data_loader_report.json", "r", encoding="utf-8") as fh:
            report = json.load(fh)

        expected_keys = {
            "total_prompts_extracted",
            "filtered_empty",
            "filtered_by_length",
            "filtered_by_keyword",
            "filtered_by_code_gen_pattern",
            "filtered_by_language",
            "duplicates_removed",
            "final_count",
            "processing_time_s",
        }
        assert expected_keys.issubset(report.keys())
        assert report["final_count"] > 0

    def test_idempotent_rerun(self, fixture_dir: Path, tmp_path: Path) -> None:
        """Running twice with the same config produces identical output."""
        config = DataLoaderConfig(
            dataset_path=str(fixture_dir),
            language="en",
            min_char_length=10,
            deduplicate=True,
            sample_size=None,
            random_seed=42,
        )
        mock_builder_cls, mock_en = _make_lingua_mocks()

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"

        for out_dir in (out1, out2):
            loader = DataLoader(config)
            with (
                patch("lingua.LanguageDetectorBuilder", mock_builder_cls, create=True),
                patch("llmseceval.data_loader.loader._get_lingua_language", return_value=mock_en),
            ):
                loader.run(output_dir=out_dir)

        content1 = (out1 / "prompts_clean.jsonl").read_text(encoding="utf-8")
        content2 = (out2 / "prompts_clean.jsonl").read_text(encoding="utf-8")
        assert content1 == content2
