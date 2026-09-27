"""Base class and registry for data source adapters."""

from __future__ import annotations

import io
import time
from abc import ABC, abstractmethod
from datetime import date
from typing import Any, ClassVar

import pandas as pd
import requests

from compute_options.data.schemas import validate_observations

USER_AGENT = "compute-options/0.1 (+https://github.com/talhajamal11/compute-options)"

REGISTRY: dict[str, type[Source]] = {}


def register(cls: type[Source]) -> type[Source]:
    REGISTRY[cls.name] = cls
    return cls


class Source(ABC):
    """A provider of daily GPU rental prices.

    Subclasses implement ``_fetch`` returning rows in the canonical observation schema;
    ``fetch`` validates them. ``since`` lets incremental sources skip dates already stored.
    """

    name: ClassVar[str]
    url: ClassVar[str]
    license: ClassVar[str] = "unspecified"
    # True for published benchmarks: kept as their own series rather than averaged with quotes.
    is_index: ClassVar[bool] = False
    # False when the source only exposes today's snapshot; history accrues by running daily.
    has_history: ClassVar[bool] = True

    def __init__(self, session: requests.Session | None = None, timeout: float = 30.0):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.timeout = timeout

    def fetch(self, since: date | None = None) -> pd.DataFrame:
        df = self._fetch(since)
        df = df.assign(source=self.name)
        out = validate_observations(df)
        if since is not None:
            out = out[out["date"] >= pd.Timestamp(since)]
        return out.reset_index(drop=True)

    @abstractmethod
    def _fetch(self, since: date | None) -> pd.DataFrame: ...

    def _get(self, url: str, retries: int = 3, **kwargs: Any) -> requests.Response:
        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _get_json(self, url: str, **kwargs: Any) -> Any:
        return self._get(url, **kwargs).json()

    def _get_csv(self, url: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(io.StringIO(self._get(url, **kwargs).text))
