"""Model bundle: LightGBM classifier + the vocabulary needed to score raw applications."""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_LABELS, FEATURES, build_features


@dataclass
class ModelBundle:
    booster: object                       # lightgbm.Booster
    categories: dict[str, list[str]]
    threshold: float = config.REVIEW_THRESHOLD
    meta: dict = field(default_factory=dict)

    def score(self, raw: pd.DataFrame, reasons: int = 3) -> tuple[np.ndarray, list[list[dict]]]:
        """Return (default probability per row, top risk-increasing drivers per row)."""
        X = build_features(raw, self.categories, now=pd.Timestamp.now().normalize())
        p = self.booster.predict(X)
        if not reasons:
            return p, [[] for _ in range(len(X))]
        contrib = self.booster.predict(X, pred_contrib=True)[:, :-1]  # last col = bias
        out = []
        for i in range(len(X)):
            top = np.argsort(contrib[i])[::-1][:reasons]
            out.append([
                {"feature": FEATURE_LABELS[FEATURES[j]], "value": _fmt(X.iloc[i, j]),
                 "impact": round(float(contrib[i, j]), 3)}
                for j in top if contrib[i, j] > 0
            ])
        return p, out


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return "n/a"
    return f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, (float, np.floating)) else str(v)


def save(bundle: ModelBundle, path=config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load(path=config.MODEL_PATH) -> ModelBundle:
    return joblib.load(path)
