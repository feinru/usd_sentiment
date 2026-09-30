"""
Penilaian sentimen berbasis lexicon (TANPA embeddings/transformer).

Dua pendekatan sesuai ketentuan proposal:
  1. VADER (vaderSentiment) — lexicon umum + aturan intensifier/negasi.
     Cocok untuk judul berita yang singkat dan tajam.
  2. Loughran-McDonald (LM) — lexicon KHUSUS keuangan.
     VADER sering salah di teks finansial ("liability", "risk" dsb),
     LM mengoreksinya dengan daftar kata positif/negatif akuntansi-keuangan.

Cara pakai:
  score_articles(df, text_col) -> df + kolom skor per ARTIKEL
  aggregate_daily(scored, date_col) -> SATU BARIS per hari trading

Agregasi harian memakai mean/std/min/max/count agar distribusi
sentimen intraday (banyak berita/hari) tidak hilang jadi satu angka.
Hari tanpa berita diisi netral (0) dengan news_count=0 di build_dataset.py.
"""

import re
import pandas as pd

# ---------------------------------------------------------------------------
# Loughran-McDonald: subset kata inti (dieksplisitkan agar repo mandiri tanpa
# file eksternal). Daftar penuh LM tersedia publik (mis. Univ. Notre Dame);
# untuk reproduksi penuh, ganti list ini dengan file LM lengkap — API di bawah
# tidak berubah.
# ---------------------------------------------------------------------------
LM_POSITIVE = {
    "achieve", "achieved", "achievement", "advance", "advancing", "advantage",
    "beat", "beats", "benefit", "beneficial", "better", "boom", "booming",
    "boost", "boosted", "breakthrough", "confident", "confidence", "efficient",
    "efficiency", "gain", "gains", "gained", "good", "great", "growth",
    "grow", "growing", "improve", "improved", "improvement", "improving",
    "innovative", "innovation", "opportunity", "opportunities", "optimistic",
    "optimism", "outperform", "profit", "profitable", "profits", "progress",
    "prosper", "prosperity", "rebound", "recovered", "recovery", "resilient",
    "resilience", "rise", "rising", "rose", "stable", "stability", "strength",
    "strong", "stronger", "strongest", "success", "successful", "surge",
    "surged", "upbeat", "upside",
}

LM_NEGATIVE = {
    "adverse", "adversely", "bad", "bankrupt", "bankruptcy", "bearish",
    "blow", "breach", "collapse", "collapsed", "concern", "concerns",
    "concerned", "concerning", "crash", "crisis", "critical", "cut",
    "cuts", "damage", "damaged", "decline", "declined", "declining",
    "default", "deficit", "depressed", "depressing", "deteriorate",
    "deteriorated", "disappoint", "disappointed", "disappointing", "dispute",
    "disruption", "disrupted", "doubt", "doubts", "downgrade", "downgraded",
    "down", "downside", "downturn", "drop", "dropped", "fail", "failed",
    "failing", "failure", "fall", "fallen", "falling", "fear", "fears",
    "feared", "fragile", "fraud", "friction", "grim", "halt", "halted",
    "hard", "harm", "harmful", "hit", "hurdle", "hurdles", "instability",
    "jitter", "jitters", "lag", "lagging", "layoff", "layoffs", "leak",
    "leaked", "liability", "liabilities", "loss", "losses", "lost",
    "miss", "missed", "missing", "negative", "notch", "penalty", "plunge",
    "plunged", "poor", "pressure", "pressured", "recession", "recessionary",
    "restructure", "restructuring", "risk", "risks", "risky", "shock",
    "shocked", "shortfall", "shrink", "shrinking", "slack", "slid",
    "slide", "sliding", "slip", "slipped", "slow", "slowed", "slowing",
    "slowdown", "slump", "slumped", "strain", "strained", "stress",
    "stressed", "struggle", "struggled", "struggling", "tension",
    "tensions", "threat", "threats", "turmoil", "turbulent", "uncertain",
    "uncertainty", "underperform", "unfavorable", "volatile", "volatility",
    "vulnerable", "warn", "warned", "warning", "warnings", "weak",
    "weaken", "weakened", "weakening", "weaker", "weakest", "worry",
    "worried", "worries", "worse", "worsen", "worsened", "worst",
}

_TOKEN_RE = re.compile(r"[a-z]+")

_VADER = None


def _get_vader():
    """Lazy-load VADER agar modul tetap bisa diimpor tanpa dependency."""
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
    return _VADER


def clean_text(text: str) -> str:
    """Lowercase + buang non-huruf (cukup untuk lexicon counting)."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-zA-Z\s]", " ", text).lower()


def vader_scores(texts) -> pd.DataFrame:
    """Skor VADER per teks -> DataFrame [neg, neu, pos, compound]."""
    analyzer = _get_vader()
    rows = []
    for t in texts:
        s = analyzer.polarity_scores(t if isinstance(t, str) else "")
        rows.append({"vader_neg": s["neg"], "vader_neu": s["neu"],
                     "vader_pos": s["pos"], "vader_compound": s["compound"]})
    return pd.DataFrame(rows)


def lm_scores(texts) -> pd.DataFrame:
    """Skor Loughran-McDonald per teks via counting kata.

    Kolom: lm_pos (# kata positif), lm_neg (# kata negatif),
           lm_polarity = (pos-neg)/(pos+neg+eps) in [-1, 1],
           lm_hit = 1 bila ada kata LM sama sekali (proksi subjektivitas finansial).
    """
    rows = []
    for t in texts:
        toks = _TOKEN_RE.findall(clean_text(t))
        pos = sum(1 for w in toks if w in LM_POSITIVE)
        neg = sum(1 for w in toks if w in LM_NEGATIVE)
        denom = pos + neg
        pol = (pos - neg) / denom if denom else 0.0
        rows.append({"lm_pos": pos, "lm_neg": neg,
                     "lm_polarity": pol, "lm_hit": int(denom > 0)})
    return pd.DataFrame(rows)


def score_articles(df: pd.DataFrame, text_col: str, prefix: str = "") -> pd.DataFrame:
    """Skor VADER+LM untuk tiap artikel.

    Args:
        df: dataframe berita (satu baris = satu artikel).
        text_col: kolom teks ('title' / 'full_text').
        prefix: mis. 't_' untuk judul, 'f_' untuk isi.

    Returns:
        Copy df + 8 kolom skor ber-prefix.
    """
    texts = df[text_col].fillna("").astype(str).tolist()
    v = vader_scores(texts).add_prefix(prefix)
    l = lm_scores(texts).add_prefix(prefix)
    scored = pd.concat([df.reset_index(drop=True), v, l], axis=1)
    return scored


def aggregate_daily(scored: pd.DataFrame, date_col: str, prefix: str,
                    score_cols: list) -> pd.DataFrame:
    """Agregasi skor artikel -> level harian (satu baris per tanggal).

    Statistik: mean, std, min, max untuk tiap skor + news_count.
    Kolom output memakai prefix yang sama, mis. 't_vader_compound_mean'.
    """
    g = scored.groupby(date_col)
    parts = []
    for c in score_cols:
        # c SUDAH ber-prefix (mis. 't_vader_compound'), jadi nama output
        # langsung '<c>_<stat>' tanpa menambah prefix lagi.
        agg = g[c].agg(["mean", "std", "min", "max"])
        agg.columns = [f"{c}_{s}" for s in ("mean", "std", "min", "max")]
        parts.append(agg)
    daily = pd.concat(parts, axis=1)
    daily[f"{prefix}news_count"] = g.size()
    daily = daily.reset_index().rename(columns={date_col: "tanggal"})
    daily["tanggal"] = pd.to_datetime(daily["tanggal"])
    # std NaN bila hanya 1 berita/hari -> isi 0 (tidak ada variasi teramati)
    daily = daily.fillna(0)
    return daily


# Kolom skor mentah (tanpa prefix) yang dihasilkan score_articles()
SCORE_COLS = ["vader_neg", "vader_neu", "vader_pos", "vader_compound",
              "lm_pos", "lm_neg", "lm_polarity", "lm_hit"]
