"""Plain-text formatting for the CLI."""

from __future__ import annotations

from typing import Any

import pandas as pd


def fmt_kv(d: dict[str, Any], pct_keys: tuple[str, ...] = ()) -> str:
    width = max(len(k) for k in d) if d else 0
    lines = []
    for k, v in d.items():
        if isinstance(v, float):
            v = f"{v:.2%}" if any(p in k for p in pct_keys) else f"{v:.6g}"
        lines.append(f"  {k:<{width}}  {v}")
    return "\n".join(lines)


def fmt_frame(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    return df.to_string(float_format=lambda x: floatfmt.format(x))
