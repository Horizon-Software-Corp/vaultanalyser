from typing import Dict, Tuple
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np


class MetricsStyler:

    # ────────── 色定義 ──────────
    _RGB: Dict[str, Tuple[int, int, int]] = {
        "green": (46, 125, 50),
        "red": (198, 40, 40),
    }

    def __init__(self, *, p_th: float = 0.05, zmax: float = 3.0):
        self.p_th = p_th
        self.zmax = zmax
        self._cache: Dict[Tuple[str, float], str] = {}

    # ────────── ユーティリティ ──────────
    @staticmethod
    def _blend(rgb: tuple[int, int, int], inten: float) -> str:
        """
        inten ∈ [0,1] で白→rgb を補間し、
        文字色は常に黒にする
        """
        r, g, b = rgb
        r = int(255 * (1 - inten) + r * inten)
        g = int(255 * (1 - inten) + g * inten)
        b = int(255 * (1 - inten) + b * inten)
        return f"background-color: #{r:02x}{g:02x}{b:02x}; color: black;"

    # ────────── p 値の強度 ──────────
    def _p_intensity(self, p: float) -> Tuple[float, bool]:
        """log10 スケールの強度と有意側判定を返す"""
        if p <= self.p_th:  # 有意側
            inten = (np.log10(self.p_th) - np.log10(max(p, 1e-300))) / (
                np.log10(self.p_th) - np.log10(1e-3)
            )
            return min(inten, 1.0), True
        inten = (np.log10(min(p, 1.0)) - np.log10(self.p_th)) / (
            np.log10(1.0) - np.log10(self.p_th)
        )
        return min(inten, 1.0), False

    def _style_p(self, p: float, sig_color: str) -> str:
        if pd.isna(p):  # ★ 追加
            return "background-color: #000000; color: white;"  # 黒地・白字

        try:
            p = float(p)
        except Exception:
            return ""

        inten, is_sig = self._p_intensity(p)
        base = self._RGB[
            sig_color if is_sig else ("red" if sig_color == "green" else "green")
        ]
        return self._blend(base, inten)

    # ────────── z-score のスタイル ──────────
    def _style_z(
        self, x: float, mu: float, sigma: float, *, higher_is_better: bool
    ) -> str:
        # ── NaN は背景＝黒に固定 ─────────────────────────
        if pd.isna(x):
            return "background-color: #000000; color: white;"  # 黒地に白字

        try:
            x = float(x)
        except Exception:
            return ""  # 数値以外（str など）は無装飾

        if sigma == 0 or np.isnan(sigma):
            return ""

        z = (x - mu) / sigma

        good = (z >= 0) if higher_is_better else (z <= 0)
        if good:
            inten = np.clip(abs(z) / self.zmax, 0.0, 1.0)
        else:
            inten = np.clip(abs(z) / (self.zmax * 0.05), 0.0, 1.0)

        base = self._RGB["green" if good else "red"]

        return self._blend(base, inten)

    def _robust_stats(self, series: pd.Series) -> tuple[float, float]:
        """中央値と MAD 由来のロバスト σ を返す"""
        s = series.dropna().astype(float)
        if s.empty:
            return np.nan, np.nan

        mu = s.median()
        mad = np.median(np.abs(s - mu))
        sigma = 1.4826 * mad  # 正規換算

        # 万一 MAD=0（定数列）なら fallback = 通常の std
        if sigma == 0 or np.isnan(sigma):
            sigma = s.std(ddof=0)

        return mu, sigma

    # ────────── メインエントリ ──────────
    def generate_style(
        self, df: pd.DataFrame, df_all: pd.DataFrame, data_range: str = "month"
    ):
        styler = df.style

        for col in df.columns:
            if col == "User Id" or not is_numeric_dtype(df[col]):
                print(f"Skipping non-numeric column: {col}")
                continue  # 数値列以外は無視

            # ---------- 1) rekt 列：0→緑, 1↑→赤 ----------
            if "Rekt" in col:
                g_full = self._blend(self._RGB["green"], 1.0)  # 濃い緑
                r_full = self._blend(self._RGB["red"], 1.0)  # 濃い赤
                styler = styler.map(
                    lambda v, _g=g_full, _r=r_full: (
                        _g if (pd.notna(v) and float(v) == 0) else _r
                    ),
                    subset=[col],
                )
                print(f"Styled 'Rekt' column: {col}")
                continue

            # ── p 値列 ────────────────────────────────────
            if col.startswith("p_"):
                sig_color = (
                    "green" if "up" in col else "red" if "down" in col else "green"
                )

                styler = styler.map(
                    lambda p, _c=sig_color: self._style_p(p, _c),
                    subset=[col],
                )
                print(f"Skipping non-numeric column: {col}")
                continue

            # ── z-score 列 ───────────────────────────────

            # ① “value” → log スケール
            log_mode = "Value" in col
            eps = 10e-10
            scaling_func = lambda x: (np.log10(x + eps) if log_mode else x)

            if not any(k in col for k in ["Weight"]):
                series_for_stats = df_all[col].apply(scaling_func)
                print(f"Calculating robust stats for {col}...")
                mu, sigma = self._robust_stats(series_for_stats)

            # ② “ratio / gain / apr” → µ を 0 に固定
            if any(k in col for k in ["Annualized", "APR"]):
                # 典型的ドル金利
                mu = 9.0
                sigma = 100.0
            elif any(k in col for k in ["Gain", "Ratio"]):
                # 予言能力なし
                mu = 0.0
            elif any(k in col for k in ["Days"]):
                # 予言能力なし
                if data_range == "allTime":
                    mu = 365
                    sigma = 180
                elif data_range == "month":
                    mu = 20
                    sigma = 10
            elif any(k in col for k in ["Fraction"]):
                # 予言能力なし
                mu = 30
                sigma = 20
            elif any(k in col for k in ["DD"]):
                mu = 10
                sigma = 10
            elif any(k in col for k in ["Value"]):
                mu = scaling_func(100000)
                sigma = 2
            elif any(k in col for k in ["Weight"]):
                mu = 0
                sigma = 0.1

            if sigma == 0 or pd.isna(sigma):
                continue  # 定数列はスキップ

            # ③ 指標の「良し悪し」判定
            hib = not any(k in col for k in ("DD", "Rekt"))

            # ④ セルごとのスタイル適用
            styler = styler.map(
                lambda x, _m=mu, _s=sigma, _hib=hib, _log_mode=log_mode: self._style_z(
                    np.log10(x + eps) if _log_mode else x, _m, _s, higher_is_better=_hib
                ),
                subset=[col],
            )
        float_cols = df.select_dtypes(include=["float"]).columns
        if len(float_cols):
            styler = styler.format(precision=3, subset=float_cols)

        return styler
