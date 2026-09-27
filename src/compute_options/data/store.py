"""Parquet storage: one raw file per source (upserted), plus named processed frames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from compute_options.config import DATA_DIR
from compute_options.data.schemas import KEY, empty_observations, validate_observations


class ObservationStore:
    def __init__(self, root: str | Path = DATA_DIR):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"

    def _raw_path(self, source: str) -> Path:
        return self.raw_dir / f"{source}.parquet"

    def upsert(self, source: str, df: pd.DataFrame) -> int:
        """Merge new rows into the source's file; newer rows win on ``KEY``. Returns row count."""
        path = self._raw_path(source)
        existing = pd.read_parquet(path) if path.exists() else empty_observations()
        merged = pd.concat([existing, df], ignore_index=True)
        merged = merged.drop_duplicates(KEY, keep="last")
        merged = validate_observations(merged)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
        return len(merged)

    def last_date(self, source: str) -> pd.Timestamp | None:
        path = self._raw_path(source)
        if not path.exists():
            return None
        dates = pd.read_parquet(path, columns=["date"])["date"]
        return None if dates.empty else dates.max()

    def sources(self) -> list[str]:
        return sorted(p.stem for p in self.raw_dir.glob("*.parquet"))

    def read(self, sources: list[str] | None = None) -> pd.DataFrame:
        names = sources or self.sources()
        frames = [pd.read_parquet(self._raw_path(s)) for s in names if self._raw_path(s).exists()]
        return (
            validate_observations(pd.concat(frames, ignore_index=True))
            if frames
            else empty_observations()
        )

    def write_frame(self, name: str, df: pd.DataFrame) -> Path:
        path = self.processed_dir / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return path

    def read_frame(self, name: str) -> pd.DataFrame:
        path = self.processed_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `compute-options aggregate` first")
        return pd.read_parquet(path)
