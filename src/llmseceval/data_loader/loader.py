"""Data Loader, loads raw PromptSet Parquet files, cleans, filters, and
outputs a deduplicated, English-only prompt list as JSONL.

Processing pipeline
-------------------
4. Remove prompts shorter than ``min_char_length``.
5. Detect language with *lingua* and retain only the target language.
6. Remove exact-duplicate prompt texts.
7. Assign deterministic ``prompt_id`` (SHA-256 prefix).
8. Optionally subsample.
9. Write ``prompts_clean.jsonl`` and ``data_loader_report.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from llmseceval.config import DataLoaderConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language code → lingua Language mapping
# ---------------------------------------------------------------------------

# We lazily import lingua so the module can be imported even when the
# (heavy) lingua library is not installed, useful for tests that mock it.
_LINGUA_LANG_MAP: dict[str, Any] | None = None


def _get_lingua_language(iso_code: str) -> Any:
    """Return the ``lingua.Language`` member for an ISO 639-1 code."""
    global _LINGUA_LANG_MAP  # noqa: PLW0603
    if _LINGUA_LANG_MAP is None:
        from lingua import Language  # type: ignore[import-untyped]

        _LINGUA_LANG_MAP = {lang.iso_code_639_1.name.lower(): lang for lang in Language.all()}
    lang = _LINGUA_LANG_MAP.get(iso_code.lower())
    if lang is None:
        raise ValueError(
            f"Unsupported language code '{iso_code}'. "
            f"Available: {sorted(_LINGUA_LANG_MAP.keys())}"
        )
    return lang


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class DataLoader:
    """Loads, cleans, and filters a PromptSet dataset.

    Parameters
    ----------
    config:
        A validated ``DataLoaderConfig`` instance.
    """

    # Languages to include in the lingua detector for speed (covers >99% of
    # the non-English content in PromptSet).
    _DETECTOR_LANGUAGES = [
        "en", "zh", "ko", "ja", "ru", "de", "fr", "es", "pt",
    ]

    def __init__(self, config: DataLoaderConfig) -> None:
        self.config = config

    # ---- public API -------------------------------------------------------

    def run(self, output_dir: str | Path) -> Path:
        """Execute the full data-loading pipeline.

        Parameters
        ----------
        output_dir:
            Directory to write output files into.  Created if it doesn't exist.

        Returns
        -------
        Path
            Path to the generated ``prompts_clean.jsonl``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        stats: dict[str, Any] = {}

        # 1. Load & Extract
        logger.info("Loading DevGPT JSON files from %s", self.config.dataset_path)
        df_prompts = self._load_devgpt_json(self.config.dataset_path)
        stats["total_prompts_extracted"] = len(df_prompts)

        # 3. Remove non-string / empty
        logger.info("Filtering empty / non-string prompts")
        before = len(df_prompts)
        df_prompts = self._filter_empty(df_prompts)
        stats["filtered_empty"] = before - len(df_prompts)

        # 4. Length filter
        logger.info("Filtering prompts shorter than %d chars", self.config.min_char_length)
        before = len(df_prompts)
        df_prompts = self._filter_by_length(df_prompts, self.config.min_char_length)
        stats["filtered_by_length"] = before - len(df_prompts)

        if self.config.disable_filters:
            logger.warning(
                "disable_filters=True, bypassing keyword, code-gen-pattern, "
                "and language filters. All prompts will pass through.",
            )
            stats["filtered_by_keyword"] = 0
            stats["filtered_by_code_gen_pattern"] = 0
            stats["filtered_by_language"] = 0
            stats["filters_bypassed"] = True
        else:
            stats["filters_bypassed"] = False

            # Keyword filter
            if self.config.required_keyword:
                logger.info("Filtering prompts missing keyword '%s'", self.config.required_keyword)
                before = len(df_prompts)
                df_prompts = self._filter_by_keyword(df_prompts, self.config.required_keyword)
                stats["filtered_by_keyword"] = before - len(df_prompts)
            else:
                stats["filtered_by_keyword"] = 0

            # Code-gen anchor filter, drops meta/editing prompts that mention
            # Python but don't ask for code (e.g. "summarize my python project").
            if self.config.required_code_gen_pattern:
                logger.info(
                    "Filtering prompts missing code-gen anchor (%s)",
                    ", ".join(self.config.required_code_gen_pattern),
                )
                before = len(df_prompts)
                df_prompts = self._filter_by_code_gen_pattern(
                    df_prompts, self.config.required_code_gen_pattern,
                )
                stats["filtered_by_code_gen_pattern"] = before - len(df_prompts)
            else:
                stats["filtered_by_code_gen_pattern"] = 0

            # 5. Language filter
            logger.info("Detecting language (target: %s)", self.config.language)
            before = len(df_prompts)
            df_prompts = self._filter_by_language(df_prompts, self.config.language)
            stats["filtered_by_language"] = before - len(df_prompts)

        # 6. Deduplication
        if self.config.deduplicate:
            logger.info("Removing duplicate prompts")
            before = len(df_prompts)
            df_prompts = self._deduplicate(df_prompts)
            stats["duplicates_removed"] = before - len(df_prompts)
        else:
            stats["duplicates_removed"] = 0

        # 7. Assign IDs
        logger.info("Assigning prompt IDs")
        df_prompts = self._assign_ids(df_prompts)

        # 8. Optional sampling
        if self.config.sample_size is not None:
            logger.info("Sampling %d prompts (seed=%d)", self.config.sample_size, self.config.random_seed)
            before = len(df_prompts)
            df_prompts = self._sample(df_prompts, self.config.sample_size, self.config.random_seed)
            stats["sampled_from"] = before
            stats["sample_size"] = self.config.sample_size
        else:
            stats["sampled_from"] = None
            stats["sample_size"] = None

        stats["final_count"] = len(df_prompts)

        # 9. Write outputs
        jsonl_path = output_dir / "prompts_clean.jsonl"
        logger.info("Writing %d prompts to %s", len(df_prompts), jsonl_path)
        self._write_jsonl(df_prompts, jsonl_path)

        elapsed = time.perf_counter() - t0
        stats["processing_time_s"] = round(elapsed, 2)

        report_path = output_dir / "data_loader_report.json"
        logger.info("Writing report to %s", report_path)
        self._write_report(stats, report_path)

        logger.info(
            "Data Loader complete, %d prompts in %.1fs",
            stats["final_count"],
            elapsed,
        )
        return jsonl_path

    # ---- pipeline steps (each is independently testable) ------------------

    @staticmethod
    def _load_devgpt_json(dataset_path: str | Path) -> pd.DataFrame:
        """Load DevGPT JSON snapshot files and extract prompts into a DataFrame."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        json_files = sorted(path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in {path}")

        logger.info("Found %d JSON file(s)", len(json_files))
        
        records = []
        original_index = 0
        
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    
                items = data.get("Sources", []) if isinstance(data, dict) else data
                    
                if not isinstance(items, list):
                    logger.warning("Expected list of items in %s, got %s. Skipping.", file_path.name, type(items))
                    continue
                    
                for item in items:
                    chatgpt_sharings = item.get("ChatgptSharing", [])
                    if not isinstance(chatgpt_sharings, list):
                        continue
                        
                    for sharing in chatgpt_sharings:
                        url = sharing.get("URL", "")
                        conversations = sharing.get("Conversations", [])
                        if not isinstance(conversations, list):
                            continue
                            
                        for conv in conversations:
                            prompt_text = conv.get("Prompt")
                            if prompt_text is not None:
                                records.append({
                                    "prompt_text": prompt_text,
                                    "url": url,
                                    "source_file": file_path.name,
                                    "original_index": original_index
                                })
                                original_index += 1
            except Exception as e:
                logger.error("Failed to parse %s: %s", file_path.name, e)

        if not records:
            return pd.DataFrame(columns=["prompt_text", "url", "source_file", "original_index"])
            
        return pd.DataFrame(records)

    @staticmethod
    def _filter_empty(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where ``prompt_text`` is not a string or is blank."""
        mask = df["prompt_text"].apply(
            lambda x: isinstance(x, str) and len(x.strip()) > 0,
        )
        return df.loc[mask].reset_index(drop=True)

    @staticmethod
    def _filter_by_length(df: pd.DataFrame, min_chars: int) -> pd.DataFrame:
        """Remove prompts shorter than *min_chars* characters."""
        mask = df["prompt_text"].str.len() >= min_chars
        return df.loc[mask].reset_index(drop=True)

    @staticmethod
    def _filter_by_keyword(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
        """Keep only prompts that contain *keyword* (case-insensitive)."""
        mask = df["prompt_text"].str.contains(keyword, case=False, na=False)
        return df.loc[mask].reset_index(drop=True)

    @staticmethod
    def _filter_by_code_gen_pattern(df: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
        """Keep only prompts where at least one of *patterns* appears as a
        whole word (case-insensitive). Used to require a code-gen anchor in
        addition to the language keyword, e.g. ``"implement"`` matches but
        ``"implementation"`` alone does not.
        """
        if df.empty or not patterns:
            return df
        regex = r"\b(?:" + "|".join(re.escape(p) for p in patterns) + r")\b"
        mask = df["prompt_text"].str.contains(regex, case=False, na=False, regex=True)
        return df.loc[mask].reset_index(drop=True)

    def _filter_by_language(self, df: pd.DataFrame, target_lang: str) -> pd.DataFrame:
        """Detect the language of each prompt and keep only *target_lang*.

        Uses ``lingua`` in low-accuracy mode for speed, restricted to a small
        set of languages.
        """
        if df.empty:
            return df

        from lingua import LanguageDetectorBuilder  # type: ignore[import-untyped]

        # Build a restricted detector for speed
        languages = [_get_lingua_language(code) for code in self._DETECTOR_LANGUAGES]
        detector = (
            LanguageDetectorBuilder.from_languages(*languages)
            .with_low_accuracy_mode()
            .build()
        )

        target = _get_lingua_language(target_lang)
        texts = df["prompt_text"].tolist()

        keep_mask: list[bool] = []
        with Progress(
            TextColumn("[bold blue]Language detection"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Detecting...", total=len(texts))
            for text in texts:
                detected = detector.detect_language_of(text)
                keep_mask.append(detected == target)
                progress.advance(task)

        return df.loc[keep_mask].reset_index(drop=True)

    @staticmethod
    def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact-duplicate ``prompt_text`` values, keeping first."""
        return df.drop_duplicates(subset="prompt_text", keep="first").reset_index(drop=True)

    @staticmethod
    def _assign_ids(df: pd.DataFrame) -> pd.DataFrame:
        """Add a deterministic ``prompt_id`` column (SHA-256 prefix, 12 hex chars)."""
        df = df.copy()
        df["prompt_id"] = df["prompt_text"].apply(
            lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()[:12],
        )
        return df

    @staticmethod
    def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        """Randomly subsample *n* rows from *df*."""
        if n >= len(df):
            logger.warning(
                "sample_size (%d) >= available prompts (%d); returning all.",
                n,
                len(df),
            )
            return df
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    # ---- serialisation ----------------------------------------------------

    @staticmethod
    def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
        """Write the DataFrame as JSONL (one JSON object per line)."""
        records = []
        for _, row in df.iterrows():
            url = row.get("url", "")
            if pd.isna(url):
                url = ""
            records.append(
                {
                    "prompt_id": row["prompt_id"],
                    "url": str(url),
                    "text": str(row["prompt_text"]),
                    "char_count": len(str(row["prompt_text"])),
                    "source_file": str(row["source_file"]),
                }
            )

        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_report(stats: dict[str, Any], path: Path) -> None:
        """Write the filtering statistics report as JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Standalone entry point (for manual testing / CLI usage before full CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from llmseceval.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)
    loader = DataLoader(cfg.data_loader)
    loader.run(output_dir="./results/data_loader_output")
