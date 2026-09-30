import re
import pandas as pd
from typing import List

# Loughran-McDonald simple subset for financial sentiment
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

LM_UNCERTAINTY = {
    "ambiguous", "appear", "appears", "approximately", "believe", "conceivable",
    "contingent", "could", "depend", "doubt", "either", "estimate", "estimated",
    "eventual", "fluctuate", "fluctuating", "if", "improbable", "indefinite",
    "likely", "may", "maybe", "might", "pending", "possible", "possibly",
    "probable", "probably", "question", "rough", "seem", "speculate", "uncertain",
    "uncertainty", "unclear", "unknown", "unpredictable", "unsure", "whether",
}

_TOKEN_RE = re.compile(r"[a-z]+")

_VADER_ANALYZER = None

def _get_vader():
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-zA-Z\s]", " ", text).lower()

def compute_vader_scores(texts: List[str]) -> pd.DataFrame:
    analyzer = _get_vader()
    scores = []
    for text in texts:
        text_str = text if isinstance(text, str) else ""
        s = analyzer.polarity_scores(text_str)
        scores.append({
            "vader_neg": s["neg"],
            "vader_neu": s["neu"],
            "vader_pos": s["pos"],
            "vader_compound": s["compound"]
        })
    return pd.DataFrame(scores)

def compute_lm_scores(texts: List[str]) -> pd.DataFrame:
    scores = []
    for text in texts:
        tokens = _TOKEN_RE.findall(clean_text(text))
        pos_count = sum(1 for w in tokens if w in LM_POSITIVE)
        neg_count = sum(1 for w in tokens if w in LM_NEGATIVE)
        unc_count = sum(1 for w in tokens if w in LM_UNCERTAINTY)
        
        total_lm = pos_count + neg_count
        total_words = max(len(tokens), 1)
        
        polarity = (pos_count - neg_count) / total_lm if total_lm > 0 else 0.0
        
        scores.append({
            "lm_pos": pos_count,
            "lm_neg": neg_count,
            "lm_unc": unc_count,
            "lm_polarity": polarity,
            "lm_unc_ratio": unc_count / total_words,
            "lm_hit": int(total_lm > 0)
        })
    return pd.DataFrame(scores)

def score_dataframe(df: pd.DataFrame, text_col: str, prefix: str = "") -> pd.DataFrame:
    texts = df[text_col].fillna("").astype(str).tolist()
    
    vader_df = compute_vader_scores(texts).add_prefix(prefix)
    lm_df = compute_lm_scores(texts).add_prefix(prefix)
    
    return pd.concat([df.reset_index(drop=True), vader_df, lm_df], axis=1)

def aggregate_daily(df: pd.DataFrame, date_col: str, prefix: str, score_cols: List[str]) -> pd.DataFrame:
    grouped = df.groupby(date_col)
    
    aggs = []
    for col in score_cols:
        agg = grouped[col].agg(["mean", "std", "min", "max"])
        agg.columns = [f"{col}_{stat}" for stat in ["mean", "std", "min", "max"]]
        aggs.append(agg)
        
    daily = pd.concat(aggs, axis=1)
    daily[f"{prefix}news_count"] = grouped.size()
    daily = daily.reset_index().rename(columns={date_col: "tanggal"})
    daily["tanggal"] = pd.to_datetime(daily["tanggal"])
    daily = daily.fillna(0)
    
    return daily
