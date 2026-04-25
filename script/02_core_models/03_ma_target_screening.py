"""
03_ma_target_screening.py

This script implements a target screening model for M&A deal sourcing, using the BC matrix and financial features to 
predict which targets are most likely to be chosen by an acquirer in a given year. It trains a ranking model (XGBoost 
or logistic regression) on historical deals and evaluates its performance on held-out test years.

The script also includes diagnostic functions to analyze feature importance, score distributions, and the relationship 
between predicted scores and BC synergy. The final output is a ranked list of candidate targets for the specified acquirer, 
along with their predicted scores and synergy features. This can be used as a practical tool for deal sourcing, as well as 
a framework for further analysis and validation of the BC-based synergy measure. 

Usage:
  python script/02_core_models/03_ma_target_screening.py --acquirer-gvkey 001632 --train-year 2020 --model xgb --top-n 10
    (This example trains an XGBoost ranker on 2020 deals for acquirer with gvkey 007257 and outputs the top 10 candidates.)

Input data requirements:
    - data/processed/modeling_dataset_with_synergy.csv: historical M&A deal data with acquirer and target gvkeys and deal years.
    - data/raw/clean_financials.csv: Compustat financial data with gvkey, year, and relevant financial metrics.
    - data/raw/corrected/firm_gmm_parameters_k15.parquet: GMM parameters for each firm used in BC calculation.
    - data/raw/corrected/bc_matrix_all_k15_dedup_linear.npz: Precomputed BC matrix and corresponding gvkeys.
    - data/raw/kmax_sweep/coassignment_audit.parquet: Data on shared patents between firms for additional features.
    - data/raw/compustat_names.csv: Mapping of gvkeys to company names for interpretability.

Output:
    - Ranked list of candidate targets for the specified acquirer, with predicted scores and synergy features.
    - Optional CSV output with detailed candidate information and scores.
    - Diagnostic reports on feature importance and score distributions.

"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", None) or str(exc)
    raise SystemExit(
        "\n".join(
            [
                f"Missing Python dependency: {missing}",
                "",
                "Setup (recommended):",
                "  python3 -m venv .venv",
                "  source .venv/bin/activate",
                "  pip install -r requirements.txt",
                "  # optional for XGBoost-based scripts:",
                "  pip install -r requirements-xgboost.txt",
                "",
                "Then rerun:",
                "  python3 03_ma_target_screening.py",
            ]
        )
    ) from exc

XGBRanker = None
_xgb_import_error: Exception | None = None
try:
    from xgboost import XGBRanker as _XGBRanker

    XGBRanker = _XGBRanker
except Exception as exc:  # noqa: BLE001
    _xgb_import_error = exc

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", None) or str(exc)
    raise SystemExit(
        "\n".join(
            [
                f"Missing Python dependency: {missing}",
                "",
                "Fix:",
                "  pip install -r requirements.txt",
            ]
        )
    ) from exc

# ---------------- CONFIG ---------------- #

DEALS_PATH = Path("data/processed/modeling_dataset_with_synergy.csv")
FIN_PATH = Path("data/raw/clean_financials.csv")
GMM_K15 = Path("data/raw/corrected/firm_gmm_parameters_k15.parquet")
BC_K15 = Path("data/raw/corrected/bc_matrix_all_k15_dedup_linear.npz")
COASSIGN_PARQUET = Path("data/raw/kmax_sweep/coassignment_audit.parquet")
COMPUSTAT_NAMES_PATH = Path("data/raw/compustat_names.csv")
PATENTS_META_PATH = Path("data/raw/patents/firm_patents_text_metadata_techbio.parquet")

TRAIN_YEAR = 2020
TEST_YEARS = [2021, 2022, 2023]

ACQUIRER_GVKEY = "007257"  # 
TOP_N = 10
MOTIVE_TOP_K = 50
WRITE_CSV = True
MODEL_KIND = "auto"  # auto | xgb | lr
SYNERGY_WEIGHT = 0.0  # optional post-hoc blend for LR
PRIVATE_BC_MIN = 0.001
PRIVATE_TOP_N = 10
PRIVATE_OVERLAP_TOP_K = 100
CANDIDATE_TOP_K = 200
ZOMBIE_WINDOW_YEARS = 3
FILTER_ZOMBIES = True

# ---------------- HELPERS ---------------- #

def require_file(path: Path) -> None:
    if path.exists():
        return
    raise SystemExit(
        "\n".join(
            [
                f"Missing input file: {path}",
                "Double-check your working directory and that the data pipeline has produced this file.",
            ]
        )
    )

def require_columns(df: "pd.DataFrame", name: str, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if not missing:
        return
    raise SystemExit(f"{name} is missing required columns: {missing}")

def load_bc_matrix(path):
    data = np.load(path, allow_pickle=True)
    if "gvkeys" not in data or "bc_matrix" not in data:
        raise SystemExit(f"{path} must contain arrays 'gvkeys' and 'bc_matrix'.")
    gvkeys = [str(g) for g in data["gvkeys"]]
    return gvkeys, data["bc_matrix"]

def make_pair_key(a, b):
    return tuple(sorted([str(a), str(b)]))

def load_gvkey_to_name(path: Path, *, year: int) -> dict[str, str]:
    if not path.exists():
        print(f"Warning: names file not found: {path}")
        return {}

    names = pd.read_csv(path, usecols=["gvkey", "conm", "year1", "year2"], low_memory=False)
    names["gvkey"] = (
        names["gvkey"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    names["year1"] = pd.to_numeric(names["year1"], errors="coerce")
    names["year2"] = pd.to_numeric(names["year2"], errors="coerce")

    # Keep names active during the requested year; treat missing year2 as open-ended.
    y = int(year)
    mask = (names["year1"].isna() | (names["year1"] <= y)) & (names["year2"].isna() | (names["year2"] >= y))
    names = names.loc[mask].copy()

    if names.empty:
        return {}

    # Resolve duplicates: prefer the entry with the latest year2 (or open-ended), then latest year1.
    names["_year2"] = names["year2"].fillna(9999)
    names["_year1"] = names["year1"].fillna(-9999)
    names = names.sort_values(["gvkey", "_year2", "_year1"], ascending=[True, False, False]).drop_duplicates("gvkey")

    return dict(zip(names["gvkey"], names["conm"]))

def is_numeric_gvkey(g: str) -> bool:
    return isinstance(g, str) and g.isdigit() and len(g) <= 6

def recent_patent_counts(
    path: Path,
    gvkeys: list[str],
    *,
    year: int,
    window_years: int,
) -> dict[str, int]:
    """
    Counts patents filed in [year-window_years, year) for the given gvkeys.
    Uses a parquet scan with pushdown filters when possible.
    """
    if not path.exists():
        print(f"Warning: patents metadata not found; zombie filter disabled: {path}")
        return {}

    numeric = [str(g).zfill(6) for g in gvkeys if is_numeric_gvkey(str(g))]
    if not numeric:
        return {}

    start = f"{int(year - window_years)}-01-01"
    end = f"{int(year)}-01-01"

    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(path), format="parquet")
        filt = (
            ds.field("gvkey").isin(numeric)
            & (ds.field("patent_date") >= start)
            & (ds.field("patent_date") < end)
        )
        table = dataset.to_table(columns=["gvkey"], filter=filt)
        values = [str(x).replace(".0", "").zfill(6) for x in table.column("gvkey").to_pylist()]
        return dict(Counter(values))
    except Exception:
        patents = pd.read_parquet(path, columns=["gvkey", "patent_date"])
        patents["gvkey"] = patents["gvkey"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        patents = patents[patents["gvkey"].isin(numeric)]
        patents = patents[(patents["patent_date"] >= start) & (patents["patent_date"] < end)]
        return patents["gvkey"].value_counts().to_dict()

def sic2(x) -> float:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return np.nan
    return float(int(v) // 100)

def bc_component_matrix(mu_a, var_a, mu_b, var_b):
    var_a = np.clip(var_a, a_min=1e-9, a_max=None)
    var_b = np.clip(var_b, a_min=1e-9, a_max=None)
    sigma_avg = (var_a[:, None, :] + var_b[None, :, :]) / 2.0
    diff = mu_a[:, None, :] - mu_b[None, :, :]
    mahal = 0.125 * np.sum(diff**2 / sigma_avg, axis=2)

    log_det_avg = np.sum(np.log(sigma_avg), axis=2)
    log_det_a = np.sum(np.log(var_a), axis=1)
    log_det_b = np.sum(np.log(var_b), axis=1)
    det_term = 0.5 * (log_det_avg - 0.5 * (log_det_a[:, None] + log_det_b[None, :]))

    return np.exp(-(mahal + det_term))

def gmm_params_from_row(row) -> dict[str, np.ndarray]:
    k = int(row["n_components"])
    means = np.frombuffer(row["means"], dtype=np.float64).reshape(k, 50)
    covariances = np.frombuffer(row["covariances"], dtype=np.float64).reshape(k, 50)
    return {"means": means, "covariances": covariances}

def compute_overlap_features(
    gmm_params_lookup: dict[str, dict[str, np.ndarray]],
    acq: str,
    targets: list[str],
) -> pd.DataFrame:
    out_rows = []
    if acq not in gmm_params_lookup:
        return pd.DataFrame(columns=["target", "max_single_component_overlap", "component_coverage"])

    gmm_a = gmm_params_lookup[acq]
    for tgt in targets:
        if tgt not in gmm_params_lookup:
            continue
        gmm_b = gmm_params_lookup[tgt]
        bc_grid = bc_component_matrix(gmm_a["means"], gmm_a["covariances"], gmm_b["means"], gmm_b["covariances"])
        max_overlap = float(np.max(bc_grid))
        # share of target components that have a good match in the acquirer
        target_has_match = (bc_grid.max(axis=0) > float(np.quantile(bc_grid, 0.90))).sum()
        coverage = float(target_has_match / bc_grid.shape[1]) if bc_grid.shape[1] > 0 else 0.0
        out_rows.append(
            {
                "target": tgt,
                "max_single_component_overlap": max_overlap,
                "component_coverage": coverage,
            }
        )
    return pd.DataFrame(out_rows)

def classify_candidate_motives(screen_df: pd.DataFrame, *, acq_sic2_val: float) -> pd.DataFrame:
    """
    Post-hoc (ex-ante) motive classification for screened targets.
    Produces:
      - tech_motive: SUBSTITUTE / COMPLEMENTARY / NICHE_OVERLAP / UNRELATED
      - potential_motive: HORIZONTAL_CONSOLIDATION / VERTICAL_INTEGRATION / KILLER_ACQUISITION / FIRE_SALE / UNKNOWN
    """
    df = screen_df.copy()
    df["tgt_sic2"] = df["tgt_sic"].apply(sic2)
    df["same_sic2"] = ((df["tgt_sic2"] == acq_sic2_val) & df["tgt_sic2"].notna()).astype(int)

    bc = pd.to_numeric(df["bc"], errors="coerce")
    bc_q_low = float(bc.quantile(0.25))
    bc_q_high = float(bc.quantile(0.75))

    df["tech_motive"] = "UNRELATED"
    # High synergy targets should not be labeled "UNRELATED" just because SIC differs.
    df.loc[(bc >= bc_q_high) & (df["same_sic2"] == 1), "tech_motive"] = "SUBSTITUTE"
    df.loc[(bc >= bc_q_high) & (df["same_sic2"] == 0), "tech_motive"] = "COMPLEMENTARY"
    df.loc[bc.between(bc_q_low, bc_q_high, inclusive="both"), "tech_motive"] = "COMPLEMENTARY"
    # NICHE_OVERLAP is optionally refined later when overlap features are available

    df["target_distressed"] = ((pd.to_numeric(df["tgt_fcf"], errors="coerce") < 0) | (pd.to_numeric(df["tgt_q"], errors="coerce") < 1.0)).fillna(False)

    df["potential_motive"] = "UNKNOWN"
    df.loc[df["target_distressed"], "potential_motive"] = "FIRE_SALE"
    df.loc[df["tech_motive"] == "COMPLEMENTARY", "potential_motive"] = "VERTICAL_INTEGRATION"
    df.loc[df["tech_motive"] == "SUBSTITUTE", "potential_motive"] = "HORIZONTAL_CONSOLIDATION"
    return df

def apply_niche_override(df: pd.DataFrame) -> pd.DataFrame:
    if "max_single_component_overlap" not in df.columns:
        return df
    out = df.copy()
    bc = pd.to_numeric(out["bc"], errors="coerce")
    bc_q_low = float(bc.quantile(0.25))
    overlap = pd.to_numeric(out["max_single_component_overlap"], errors="coerce")
    if overlap.notna().sum() >= 10:
        overlap_hi = float(overlap.quantile(0.90))
    else:
        overlap_hi = float(overlap.max(skipna=True)) if overlap.notna().any() else np.nan

    niche_mask = (bc <= bc_q_low) & (overlap >= overlap_hi)
    out.loc[niche_mask, "tech_motive"] = "NICHE_OVERLAP"
    out.loc[niche_mask, "potential_motive"] = "KILLER_ACQUISITION"
    return out

def diagnostic_feature_report(df: pd.DataFrame, cols: list[str], label: str) -> None:
    if df.empty:
        return
    report = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        report.append((c, int(s.nunique(dropna=True)), float(s.std(skipna=True) if s.notna().any() else 0.0)))
    constantish = [c for c, nunique, std in report if nunique <= 1 or std == 0.0]
    if constantish:
        print(f"Diagnostic warning ({label}): near-constant columns: {constantish}")

def diagnostic_score_report(scores: pd.Series) -> None:
    s = pd.to_numeric(scores, errors="coerce")
    nunique = int(s.nunique(dropna=True))
    if nunique <= 1:
        print("Diagnostic warning: model produced a constant score across candidates.")
    elif nunique < 10:
        counts = s.round(6).value_counts().head(10)
        print("Diagnostic: score has few unique values; top duplicates (rounded):")
        print(counts.to_string())

def drop_constant_and_collinear_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    keep: list[str] = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)

    if len(keep) <= 1:
        return keep

    corr = df[keep].corr(numeric_only=True).abs()
    to_drop = set()
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            a, b = keep[i], keep[j]
            v = corr.iloc[i, j]
            if pd.notna(v) and v >= 0.999:
                # Prefer keeping synergy features when possible
                prefer = {"bc_norm", "bc", "shared_patents", "target_patents", "bc_x_shared", "bc_x_size"}
                if a in prefer and b not in prefer:
                    to_drop.add(b)
                elif b in prefer and a not in prefer:
                    to_drop.add(a)
                else:
                    to_drop.add(b)
    return [c for c in keep if c not in to_drop]

def print_synergy_diagnostics(df: pd.DataFrame, *, label: str) -> None:
    if df.empty:
        return
    if "score" in df.columns and "bc" in df.columns:
        s = pd.to_numeric(df["score"], errors="coerce")
        bc = pd.to_numeric(df["bc"], errors="coerce")
        if s.notna().sum() >= 5 and bc.notna().sum() >= 5:
            corr = float(s.corr(bc))
            print(f"Diagnostic ({label}): corr(score, bc) = {corr:.4f}")

# ---------------- CLI ---------------- #

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--acquirer-gvkey", default=ACQUIRER_GVKEY)
parser.add_argument("--top-n", type=int, default=TOP_N)
parser.add_argument("--train-year", type=int, default=TRAIN_YEAR)
parser.add_argument("--motive-top-k", type=int, default=MOTIVE_TOP_K, help="Compute component-overlap features for top K candidates (slow).")
parser.add_argument("--no-csv", action="store_true", help="Do not write output CSV.")
parser.add_argument("--model", choices=["auto", "xgb", "lr"], default=MODEL_KIND, help="Model for ranking: XGBoost ranker or logistic regression.")
parser.add_argument("--synergy-weight", type=float, default=SYNERGY_WEIGHT, help="Optional post-hoc boost: final_score = model_score + w * rank_pct(bc).")
parser.add_argument("--candidate-top-k", type=int, default=CANDIDATE_TOP_K, help="Initial candidate pool size (by BC).")
parser.add_argument("--private-bc-min", type=float, default=PRIVATE_BC_MIN, help="Minimum BC to show in private radar.")
parser.add_argument("--private-top-n", type=int, default=PRIVATE_TOP_N, help="Number of private targets to print.")
parser.add_argument("--private-overlap-top-k", type=int, default=PRIVATE_OVERLAP_TOP_K, help="Compute component-overlap for top K private-by-BC candidates.")
parser.add_argument("--zombie-window-years", type=int, default=ZOMBIE_WINDOW_YEARS, help="Recent patent activity window for zombie filter.")
parser.add_argument("--no-zombie-filter", action="store_true", help="Disable private radar zombie filtering.")
args = parser.parse_args()

ACQUIRER_GVKEY = str(args.acquirer_gvkey).zfill(6)
TOP_N = int(args.top_n)
TRAIN_YEAR = int(args.train_year)
MOTIVE_TOP_K = int(args.motive_top_k)
WRITE_CSV = not args.no_csv
MODEL_KIND = str(args.model)
SYNERGY_WEIGHT = float(args.synergy_weight)
CANDIDATE_TOP_K = int(args.candidate_top_k)
PRIVATE_BC_MIN = float(args.private_bc_min)
PRIVATE_TOP_N = int(args.private_top_n)
PRIVATE_OVERLAP_TOP_K = int(args.private_overlap_top_k)
ZOMBIE_WINDOW_YEARS = int(args.zombie_window_years)
FILTER_ZOMBIES = not bool(args.no_zombie_filter)

# ---------------- LOAD DATA ---------------- #

print("Loading data...")

for p in [DEALS_PATH, FIN_PATH, GMM_K15, BC_K15, COASSIGN_PARQUET]:
    require_file(p)

deals = pd.read_csv(DEALS_PATH)
fin = pd.read_csv(FIN_PATH, low_memory=False)
gmm_df = pd.read_parquet(GMM_K15)
audit = pd.read_parquet(COASSIGN_PARQUET)

require_columns(deals, "deals", ["acquiror_id", "target_id", "year"])
require_columns(
    fin,
    "fin",
    ["gvkey", "year", "at", "sich", "tobins_q_filled", "free_cash_flow"],
)
require_columns(
    gmm_df,
    "gmm_df",
    ["gvkey", "n_patents", "n_components", "means", "covariances"],
)
require_columns(audit, "audit", ["gvkey_a", "gvkey_b", "n_shared"])

gvkeys, bc_matrix = load_bc_matrix(BC_K15)
gvkey_to_idx = {g: i for i, g in enumerate(gvkeys)}

# clean gvkeys
fin['gvkey'] = fin['gvkey'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
fin['year'] = pd.to_numeric(fin['year'], errors='coerce')

gvkey_to_name = load_gvkey_to_name(COMPUSTAT_NAMES_PATH, year=TRAIN_YEAR)

patent_counts = dict(zip(gmm_df['gvkey'].astype(str), gmm_df['n_patents']))

audit['pair_key'] = audit.apply(lambda r: make_pair_key(r['gvkey_a'], r['gvkey_b']), axis=1)
audit_lookup = audit.set_index('pair_key')['n_shared'].to_dict()

# ---------------- BUILD DATASET ---------------- #

def build_dataset(deals_subset):
    rows = []
    np.random.seed(42)

    for deal_id, d in deals_subset.iterrows():
        acq = str(d['acquiror_id']).zfill(6)
        tgt = str(d['target_id']).zfill(6)
        year = d['year']
        feature_year = year - 1

        if acq not in gvkey_to_idx or tgt not in gvkey_to_idx:
            continue

        # get acquirer financials
        acq_fin = fin[(fin['gvkey'] == acq) & (fin['year'] == feature_year)]
        if acq_fin.empty:
            continue

        acq_at = float(acq_fin['at'].iloc[0])
        acq_q = float(acq_fin['tobins_q_filled'].iloc[0])
        acq_fcf = float(acq_fin['free_cash_flow'].iloc[0])
        acq_sic = acq_fin['sich'].iloc[0]

        acq_idx = gvkey_to_idx[acq]

        # --- POSITIVE --- #
        tgt_fin = fin[(fin['gvkey'] == tgt) & (fin['year'] == feature_year)]

        if not tgt_fin.empty:
            tgt_at = float(tgt_fin['at'].iloc[0])
            tgt_q = float(tgt_fin['tobins_q_filled'].iloc[0])
        else:
            tgt_at, tgt_q = np.nan, np.nan

        bc_score = float(bc_matrix[acq_idx, gvkey_to_idx[tgt]])
        pkey = make_pair_key(acq, tgt)

        rows.append({
            'deal_id': deal_id,
            'acquirer': acq,
            'candidate': tgt,
            'bc': bc_score,
            'shared_patents': audit_lookup.get(pkey, 0),
            'target_patents': patent_counts.get(tgt, 0),
            'size_ratio': tgt_at / acq_at if acq_at != 0 else np.nan,
            'valuation_gap': tgt_q - acq_q,
            'affordability': acq_fcf / tgt_at if (pd.notna(tgt_at) and tgt_at != 0) else np.nan,
            'chosen': 1
        })

        # --- HARD NEGATIVES (TOP BC) --- #
        bc_row = bc_matrix[acq_idx]
        top_idx = np.argsort(-bc_row)[1:101]
        candidates = [gvkeys[i] for i in top_idx if gvkeys[i] != tgt]

        for c in np.random.choice(candidates, min(20, len(candidates)), replace=False):
            if c not in gvkey_to_idx:
                continue

            tgt_fin = fin[(fin['gvkey'] == c) & (fin['year'] == feature_year)]

            if tgt_fin.empty:
                continue

            tgt_at = float(tgt_fin['at'].iloc[0])
            tgt_q = float(tgt_fin['tobins_q_filled'].iloc[0])

            bc_score = float(bc_matrix[acq_idx, gvkey_to_idx[c]])
            pkey = make_pair_key(acq, c)

            rows.append({
                'deal_id': deal_id,
                'acquirer': acq,
                'candidate': c,
                'bc': bc_score,
                'shared_patents': audit_lookup.get(pkey, 0),
                'target_patents': patent_counts.get(c, 0),
                'size_ratio': tgt_at / acq_at if acq_at != 0 else np.nan,
                'valuation_gap': tgt_q - acq_q,
                'affordability': acq_fcf / tgt_at if (pd.notna(tgt_at) and tgt_at != 0) else np.nan,
                'chosen': 0
            })

    df = pd.DataFrame(rows).dropna()

    # normalize BC
    bc_std = float(df["bc"].std()) if "bc" in df.columns else 0.0
    if bc_std == 0 or pd.isna(bc_std):
        df["bc_norm"] = 0.0
    else:
        df["bc_norm"] = (df["bc"] - df["bc"].mean()) / bc_std

    # interactions
    df['bc_x_shared'] = df['bc_norm'] * df['shared_patents']
    df['bc_x_size'] = df['bc_norm'] * df['size_ratio']

    return df

# ---------------- SPLIT ---------------- #

train_deals = deals[deals['year'] == TRAIN_YEAR]
test_deals = deals[deals['year'].isin(TEST_YEARS)]

print("Building training data...")
train_df = build_dataset(train_deals)

print("Building test data...")
test_df = build_dataset(test_deals)

base_features = [
    "bc_norm",
    "shared_patents",
    "target_patents",
    "size_ratio",
    "valuation_gap",
    "affordability",
    "bc_x_shared",
    "bc_x_size",
]

# ---------------- TRAIN RANKER ---------------- #

print("Training ranking model...")

if train_df.empty:
    raise SystemExit("Training dataset is empty after filtering; cannot train a model.")
if len(pd.unique(train_df["chosen"])) < 2:
    raise SystemExit("Training labels have only one class after filtering; cannot train a classifier.")

features = drop_constant_and_collinear_features(train_df, base_features)
if "bc_norm" not in features:
    features = ["bc_norm"] + [c for c in features if c != "bc_norm"]

print(f"Using features ({len(features)}): {features}")

def train_model(train_df_: pd.DataFrame):
    if MODEL_KIND in {"auto", "xgb"} and XGBRanker is not None:
        print("Training XGBoost ranker...")
        group_train = train_df_.groupby("deal_id").size().values
        # Enforce that higher BC synergy cannot reduce score (addresses 'penalizing synergy').
        constraints = ["0"] * len(features)
        if "bc_norm" in features:
            constraints[features.index("bc_norm")] = "1"
        mono = "(" + ",".join(constraints) + ")"

        xgb = XGBRanker(
            objective="rank:pairwise",
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            monotone_constraints=mono,
        )
        xgb.fit(train_df_[features], train_df_["chosen"], group=group_train)
        return xgb, "xgb"

    if MODEL_KIND == "xgb" and XGBRanker is None:
        msg = str(_xgb_import_error) if _xgb_import_error is not None else "xgboost import failed"
        raise SystemExit(
            "\n".join(
                [
                    "Requested --model xgb but XGBoost is unavailable.",
                    msg,
                    "",
                    "macOS fix (common):",
                    "  brew install libomp",
                    "  pip install -r requirements-xgboost.txt",
                ]
            )
        )

    print("Training Logistic Regression model (impute + standardize)...")
    lr = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
        ]
    )
    lr.fit(train_df_[features], train_df_["chosen"])
    return lr, "lr"

model, trained_kind = train_model(train_df)

def score(df: "pd.DataFrame") -> "np.ndarray":
    if trained_kind == "xgb":
        return model.predict(df[features])
    return model.predict_proba(df[features])[:, 1]

def lr_bc_coef_sign() -> float | None:
    if trained_kind != "lr":
        return None
    if "bc_norm" not in features:
        return None
    try:
        coef = float(model.named_steps["clf"].coef_[0][features.index("bc_norm")])
        return coef
    except Exception:
        return None

# ---------------- EVALUATION ---------------- #

print("\nEvaluating model...")

if test_df.empty:
    raise SystemExit("Test dataset is empty after filtering; cannot evaluate.")

test_df["pred"] = score(test_df)

def evaluate(df):
    hits5, mrr = [], []

    for _, g in df.groupby('deal_id'):
        # Shuffle randomly to break the stable-sort bias!
        g = g.sample(frac=1, random_state=42)
        g = g.sort_values('pred', ascending=False).reset_index(drop=True)
        true_targets = g.index[g['chosen'] == 1]
    
        if len(true_targets) == 0:
            # Skip this deal if the true target was dropped during data cleaning
            continue  
        
        rank = true_targets[0] + 1

        hits5.append(int(rank <= 5))
        mrr.append(1 / rank)

    print("Hit@5:", np.mean(hits5))
    print("MRR:", np.mean(mrr))

evaluate(test_df)

# ---------------- SCREENING ---------------- #

print(f"\nScreening targets for {ACQUIRER_GVKEY}...")

acq_fin = fin[(fin['gvkey'] == ACQUIRER_GVKEY) & (fin['year'] == TRAIN_YEAR)]
if acq_fin.empty:
    raise SystemExit(
        f"No acquirer financials found for gvkey={ACQUIRER_GVKEY} year={TRAIN_YEAR} in {FIN_PATH}."
    )

acq_at = float(acq_fin['at'].iloc[0])
acq_q = float(acq_fin['tobins_q_filled'].iloc[0])
acq_fcf = float(acq_fin['free_cash_flow'].iloc[0])
acq_sic = acq_fin['sich'].iloc[0]
acq_sic2_val = sic2(acq_sic)

acq_idx = gvkey_to_idx[ACQUIRER_GVKEY]

rows = []

# 1. Pre-filter to the top 200 tech-overlap candidates
bc_row = bc_matrix[acq_idx]
top_idx = np.argsort(-bc_row)[:CANDIDATE_TOP_K]
candidate_gvkeys = [gvkeys[i] for i in top_idx]

# 2. Build candidate table (keep targets even if missing financials; they may be "private radar")
for tgt in candidate_gvkeys:
    if tgt == ACQUIRER_GVKEY:
        continue

    if tgt not in gvkey_to_idx:
        continue

    tgt_fin = fin[(fin['gvkey'] == tgt) & (fin['year'] == TRAIN_YEAR)]
    if tgt_fin.empty:
        tgt_at = np.nan
        tgt_q = np.nan
        tgt_fcf = np.nan
        tgt_sic = np.nan
    else:
        tgt_at = float(tgt_fin['at'].iloc[0])
        tgt_q = float(tgt_fin['tobins_q_filled'].iloc[0])
        tgt_fcf = float(tgt_fin['free_cash_flow'].iloc[0])
        tgt_sic = tgt_fin['sich'].iloc[0]

    bc = float(bc_matrix[acq_idx, gvkey_to_idx[tgt]])
    pkey = make_pair_key(ACQUIRER_GVKEY, tgt)

    rows.append({
        'target': tgt,
        "target_name": gvkey_to_name.get(tgt),
        'bc': bc,
        'shared_patents': audit_lookup.get(pkey, 0),
        'target_patents': patent_counts.get(tgt, 0),
        'size_ratio': tgt_at / acq_at if (pd.notna(tgt_at) and acq_at != 0) else np.nan,
        'valuation_gap': tgt_q - acq_q if pd.notna(tgt_q) else np.nan,
        'affordability': acq_fcf / tgt_at if (pd.notna(tgt_at) and tgt_at != 0) else np.nan,
        "tgt_at": tgt_at,
        "tgt_q": tgt_q,
        "tgt_fcf": tgt_fcf,
        "tgt_sic": tgt_sic,
    })

screen_df = pd.DataFrame(rows)
if screen_df.empty:
    raise SystemExit("No screening candidates found after filtering; cannot score targets.")

screen_df = screen_df.dropna(subset=["target", "bc"])

# same feature engineering
bc_std_train = float(train_df["bc"].std()) if "bc" in train_df.columns else 0.0
if bc_std_train == 0 or pd.isna(bc_std_train):
    screen_df["bc_norm"] = 0.0
else:
    screen_df["bc_norm"] = (screen_df["bc"] - train_df["bc"].mean()) / bc_std_train
screen_df['bc_x_shared'] = screen_df['bc_norm'] * screen_df['shared_patents']
screen_df['bc_x_size'] = screen_df['bc_norm'] * screen_df['size_ratio']
screen_df["score"] = np.nan
screen_df["ranking_path"] = "unscored"

# Split: public (has financials) vs private/no-financials targets.
public_mask = screen_df["tgt_q"].notna()
public_df = screen_df.loc[public_mask].copy()
private_df = screen_df.loc[~public_mask].copy()

print(f"Candidates: total={len(screen_df)} public={len(public_df)} private_or_missing_fin={len(private_df)}")

# Public companies go through the ML pipeline.
score_bc_corr = None
if not public_df.empty:
    public_df["score_model"] = score(public_df)

    effective_synergy_weight = SYNERGY_WEIGHT
    bc_coef = lr_bc_coef_sign()
    try:
        score_bc_corr = float(
            pd.to_numeric(public_df["score_model"], errors="coerce").corr(pd.to_numeric(public_df["bc"], errors="coerce"))
        )
    except Exception:
        score_bc_corr = None

    if effective_synergy_weight == 0.0 and MODEL_KIND == "auto" and trained_kind == "lr":
        if (bc_coef is not None and bc_coef < 0) or (score_bc_corr is not None and score_bc_corr < 0):
            effective_synergy_weight = 0.25
            print(
                "Diagnostic: LR is not synergy-respecting (negative bc effect detected); applying automatic synergy boost "
                f"(synergy_weight={effective_synergy_weight}). Use `--synergy-weight 0` to disable."
            )

    bc_rank_public = pd.to_numeric(public_df["bc"], errors="coerce").rank(pct=True)
    public_df["score"] = public_df["score_model"] + effective_synergy_weight * bc_rank_public
    public_df["ranking_path"] = "public_model"

    diagnostic_feature_report(public_df, features + ["bc", "shared_patents"], label="public_df")
    diagnostic_score_report(public_df["score"])
    if score_bc_corr is not None:
        print(f"Diagnostic (public_df): corr(score_model, bc) = {score_bc_corr:.4f}")
    print_synergy_diagnostics(public_df, label="public_df")

    # Write back
    for col in ["score_model", "score", "ranking_path"]:
        screen_df.loc[public_df.index, col] = public_df[col]

# Private targets bypass the ML model and get an NLP-only ranking later (after overlap features are merged).
if not private_df.empty:
    private_df["ranking_path"] = "private_nlp"
    screen_df.loc[private_df.index, "ranking_path"] = "private_nlp"

# Zombie filter inputs for private radar (recent patent activity)
screen_df["target_patents_pre3yr"] = np.nan
if FILTER_ZOMBIES and not private_df.empty:
    counts = recent_patent_counts(
        PATENTS_META_PATH,
        private_df["target"].astype(str).tolist(),
        year=TRAIN_YEAR,
        window_years=ZOMBIE_WINDOW_YEARS,
    )
    if counts:
        private_counts = private_df["target"].astype(str).map(lambda g: int(counts.get(g, 0)))
        screen_df.loc[private_df.index, "target_patents_pre3yr"] = private_counts
        n_alive = int((private_counts > 0).sum())
        print(
            f"Private radar zombie filter: alive={n_alive} zombies={len(private_counts) - n_alive} "
            f"(patents in last {ZOMBIE_WINDOW_YEARS}y > 0)."
        )
    else:
        # If we couldn't compute counts, default to 0 (conservative: treat as zombie).
        screen_df.loc[private_df.index, "target_patents_pre3yr"] = 0
        print("Private radar zombie filter: could not compute counts; treating private targets as zombies (0). Use --no-zombie-filter to bypass.")

# Motive classification uses synergy signals; it works for both public + private.
screen_df = classify_candidate_motives(screen_df, acq_sic2_val=acq_sic2_val)

# Optional: compute component-overlap features for (a) top public targets and (b) top private-by-BC targets.
targets_for_overlap: list[str] = []
if MOTIVE_TOP_K > 0:
    public_ranked = screen_df[screen_df["ranking_path"] == "public_model"].sort_values(["score", "bc"], ascending=[False, False])
    targets_for_overlap.extend(public_ranked.head(MOTIVE_TOP_K)["target"].astype(str).tolist())

if PRIVATE_OVERLAP_TOP_K > 0:
    private_ranked = screen_df[screen_df["ranking_path"] == "private_nlp"].sort_values(["bc"], ascending=[False])
    private_ranked = private_ranked[private_ranked["bc"] >= PRIVATE_BC_MIN]
    if FILTER_ZOMBIES and "target_patents_pre3yr" in private_ranked.columns:
        private_ranked = private_ranked[private_ranked["target_patents_pre3yr"].fillna(0) > 0]
    targets_for_overlap.extend(private_ranked.head(PRIVATE_OVERLAP_TOP_K)["target"].astype(str).tolist())

targets_for_overlap = list(dict.fromkeys([t for t in targets_for_overlap if isinstance(t, str) and t]))
if targets_for_overlap:
    needed = set([ACQUIRER_GVKEY, *targets_for_overlap])
    gmm_subset = gmm_df[
        gmm_df["gvkey"].astype(str).str.replace(r"\\.0$", "", regex=True).str.zfill(6).isin(needed)
    ].copy()
    gmm_subset["gvkey"] = gmm_subset["gvkey"].astype(str).str.replace(r"\\.0$", "", regex=True).str.zfill(6)
    gmm_params_lookup = {r["gvkey"]: gmm_params_from_row(r) for _, r in gmm_subset.iterrows()}
    overlap_df = compute_overlap_features(gmm_params_lookup, ACQUIRER_GVKEY, targets_for_overlap)
    if not overlap_df.empty:
        screen_df = screen_df.merge(overlap_df, on="target", how="left")
        screen_df = apply_niche_override(screen_df)

# Private radar final score (NLP-only): rank by overlap then BC.
private_mask = screen_df["ranking_path"] == "private_nlp"
if private_mask.any():
    priv = screen_df.loc[private_mask].copy()
    if FILTER_ZOMBIES and "target_patents_pre3yr" in priv.columns:
        priv = priv[priv["target_patents_pre3yr"].fillna(0) > 0]
    priv = priv[priv["bc"] >= PRIVATE_BC_MIN]
    if not priv.empty:
        bc_rank_priv = pd.to_numeric(priv["bc"], errors="coerce").rank(pct=True)
        if "max_single_component_overlap" in priv.columns:
            ov = pd.to_numeric(priv["max_single_component_overlap"], errors="coerce").fillna(0.0)
            ov_rank = ov.rank(pct=True)
            priv["score"] = 0.7 * ov_rank + 0.3 * bc_rank_priv
        else:
            priv["score"] = bc_rank_priv
        screen_df.loc[priv.index, "score"] = priv["score"]

# Filters (avoid printing completely unrelated, zero-synergy noise)
if "max_single_component_overlap" in screen_df.columns:
    screen_df = screen_df[(screen_df["bc"] > PRIVATE_BC_MIN) | (screen_df["max_single_component_overlap"] > 0.05)]
else:
    screen_df = screen_df[screen_df["bc"] > PRIVATE_BC_MIN]

public_out = screen_df[screen_df["ranking_path"] == "public_model"].sort_values(["score", "bc"], ascending=[False, False]).head(TOP_N)
private_out = (
    screen_df[screen_df["ranking_path"] == "private_nlp"]
    .loc[lambda d: d["score"].notna()]
    .pipe(lambda d: d[d["target_patents_pre3yr"].fillna(0) > 0] if (FILTER_ZOMBIES and "target_patents_pre3yr" in d.columns) else d)
    .sort_values(
        ["score", "max_single_component_overlap", "bc"] if "max_single_component_overlap" in screen_df.columns else ["score", "bc"],
        ascending=[False, False, False] if "max_single_component_overlap" in screen_df.columns else [False, False],
    )
    .head(PRIVATE_TOP_N)
)

print("\nTOP PUBLIC TARGETS:")
if public_out.empty:
    print("(none)")
else:
    print(
        public_out[
            [
                "target",
                "target_name",
                "score",
                "bc",
                "shared_patents",
                "tech_motive",
                "potential_motive",
            ]
        ].to_string(index=False)
    )

print("\n--- Top Private/Non-financial Acquisition Targets ---")
if private_out.empty:
    print("(none)")
else:
    cols = ["target", "target_name", "score", "bc"]
    if "max_single_component_overlap" in private_out.columns:
        cols.append("max_single_component_overlap")
    if "target_patents_pre3yr" in private_out.columns:
        cols.append("target_patents_pre3yr")
    cols += ["tech_motive", "potential_motive"]
    print(private_out[cols].to_string(index=False))

if WRITE_CSV:
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ma_target_screening_{ACQUIRER_GVKEY}_y{TRAIN_YEAR}.csv"
    screen_df.sort_values(["score", "bc"], ascending=[False, False]).to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")
