""" 
04_ma_motive.py 

Description: 
This script builds on the M&A deal dataset by incorporating the MASS synergy scores and flags derived from 
patent vector embeddings. It classifies each deal into technology motive buckets (Substitute, Complementary, Niche Overlap, 
Unrelated) based on a combination of industry similarity and the MASS synergy score. The script then infers strategic motives 
(Killer Acquisition, Vertical Integration, Fire Sale, Horizontal Consolidation) using a combination of the technology motive, 
financial distress indicators, and patent discontinuation metrics. Finally, it saves the enriched dataset with all these 
features for further analysis. 

Inputs:
- data/processed/modeling_dataset_with_synergy.csv: The base M&A deal dataset with basic synergy scores and flags.
- data/processed/patentbert_mass_outcomes.csv: The dataset containing MASS synergy scores and product synergy flags for each deal.

Outputs:
- data/processed/ma_motive_dataset.csv: The enriched M&A deal dataset with technology and strategic motive classifications, 
along with patent discontinuation metrics and product synergy flags.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import argparse
import warnings

warnings.filterwarnings("ignore")

# --- Config ---
DEALS_PATH = Path("data/processed/modeling_dataset_with_synergy.csv")
FIN_PATH = Path("data/raw/clean_financials.csv")
GMM_K15 = Path("data/raw/corrected/firm_gmm_parameters_k15.parquet")
PATENTS_PATH = Path("data/raw/patents/firm_patents_text_metadata_techbio.parquet")
CITATIONS_PATH = Path("data/raw/patents/citation_network_techbio.parquet")
TEXT_OUTCOMES_PATH = Path("data/processed/patentbert_mass_outcomes.csv")

OUTPUT_DIR = Path("output/ma_motives")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def require_file(path: Path) -> None:
    if path.exists():
        return
    raise SystemExit(f"Missing required file: {path}")

def require_columns(df: pd.DataFrame, name: str, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{name} is missing required columns: {missing}")

# --- Math Helpers ---
def bc_component_matrix(mu_a, var_a, mu_b, var_b):
    """Pairwise BC between all GMM components."""
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

def load_gmm_lookup(path):
    df = pd.read_parquet(path)
    lookup = {}
    for _, row in df.iterrows():
        k = int(row["n_components"])
        gvkey = str(row["gvkey"]).replace(".0", "").zfill(6)
        lookup[gvkey] = {
            "means": np.frombuffer(row["means"], dtype=np.float64).reshape(k, 50),
            "covariances": np.frombuffer(row["covariances"], dtype=np.float64).reshape(k, 50),
            "weights": np.frombuffer(row["weights"], dtype=np.float64).reshape(k),
        }
    return lookup

# --- Stage 1: Technology Classification ---
def compute_technology_features(df, gmm_lookup):
    max_overlaps = []
    component_coverages = []
    
    for _, row in df.iterrows():
        acq_id, tgt_id = row['acquiror_id'], row['target_id']
        
        if acq_id in gmm_lookup and tgt_id in gmm_lookup:
            gmm_a = gmm_lookup[acq_id]
            gmm_b = gmm_lookup[tgt_id]
            bc_grid = bc_component_matrix(
                gmm_a["means"], gmm_a["covariances"], 
                gmm_b["means"], gmm_b["covariances"]
            )
            max_overlaps.append(float(np.max(bc_grid)))
            
            target_has_match = (bc_grid.max(axis=0) > 0.5).sum()
            coverage = target_has_match / bc_grid.shape[1] if bc_grid.shape[1] > 0 else 0
            component_coverages.append(coverage)
        else:
            max_overlaps.append(0.0)
            component_coverages.append(0.0)
            
    df['max_single_component_overlap'] = max_overlaps
    df['component_coverage'] = component_coverages
    return df

def classify_technology_motive(df):
    """
    Stage-1: data-driven technology relationship buckets.
    Replaces hard thresholds with quantile-based cutoffs for better robustness.
    """
    # SIC matching (prefer 2-digit sector match; keep 4-digit exact as diagnostic)
    valid_sic = df["acq_sic"].notna() & df["tgt_sic"].notna()
    df["same_sic2"] = 0
    df["same_sic4"] = 0
    df.loc[valid_sic, "same_sic2"] = (
        (df.loc[valid_sic, "acq_sic"] // 100).astype(int)
        == (df.loc[valid_sic, "tgt_sic"] // 100).astype(int)
    ).astype(int)
    df.loc[valid_sic, "same_sic4"] = (
        (df.loc[valid_sic, "acq_sic"]).astype(int)
        == (df.loc[valid_sic, "tgt_sic"]).astype(int)
    ).astype(int)

    df["same_industry"] = df["same_sic2"]
    df['tech_motive'] = 'UNRELATED'

    bc = df["bc_synergy_score"].astype(float)
    bc_q_high = float(bc.quantile(0.75))
    bc_q_low = float(bc.quantile(0.25))
    overlap = df["max_single_component_overlap"].astype(float)
    overlap_q_high = float(overlap.quantile(0.90))

    print(
        "Stage-1 thresholds (data-driven): "
        f"bc_q_low={bc_q_low:.3e}, bc_q_high={bc_q_high:.3e}, overlap_q_high={overlap_q_high:.3e}"
    )

    substitute_mask = (df["same_industry"] == 1) & (bc >= bc_q_high)
    complementary_mask = (df["same_industry"] == 0) & (bc.between(bc_q_low, bc_q_high, inclusive="both"))
    niche_mask = (bc <= bc_q_low) & (overlap >= overlap_q_high)

    df.loc[substitute_mask, "tech_motive"] = "SUBSTITUTE"
    df.loc[complementary_mask, "tech_motive"] = "COMPLEMENTARY"
    df.loc[niche_mask, "tech_motive"] = "NICHE_OVERLAP"
    
    return df

def validate_stage1(df):
    print("\n--- Stage 1 Validation ---")
    contingency = pd.crosstab(df['tech_motive'], df['same_industry'])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    print(f"Chi2 (tech_motive vs same_industry): p = {p_value:.4e}")
    
    groups = [df[df['tech_motive'] == cat]['bc_synergy_score'].dropna() for cat in df['tech_motive'].unique()]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"ANOVA (BC across motives): p = {p_anova:.4e}")

# --- Stage 2: Strategic Inference ---

def calculate_patent_discontinuation(deals_df, patents_df, citations_df):
    """
    Calculates the drop in forward citations for a target's pre-merger patents.
    Compares a 3-year pre-merger window to a 3-year post-merger window.
    """
    print("Calculating patent discontinuation rates...")
    
    # 1. Standardize column names
    col_patent_id = "patent_id"
    col_firm_id = "gvkey"
    col_date = "patent_date"
    col_citing_id = "patent_id"   # The new patent that is citing
    col_cited_id = "citation_id"  # The old patent being cited

    # Extract just the year from the patent dates & create a lookup dictionary for patent_id -> patent_year
    patents_df["patent_year"] = pd.to_datetime(patents_df[col_date], errors="coerce").dt.year
    patent_year_dict = dict(zip(patents_df[col_patent_id], patents_df["patent_year"]))
    
    # Map the year the citation was made onto the citations dataframe
    citations_df["citing_year"] = citations_df[col_citing_id].map(patent_year_dict)
    citations_df = citations_df.dropna(subset=["citing_year"])

    # 3. Pre-group the citations for fast lookup: cited_patent -> list of years it was cited
    # This prevents us from having to search the massive citation dataframe in a loop
    cited_grouped = citations_df.groupby(col_cited_id)["citing_year"].apply(list).to_dict()
    
    # Group patents by firm: gvkey -> list of (patent_id, patent_year)
    firm_patents_dict = patents_df.groupby(col_firm_id)[col_patent_id].apply(list).to_dict()
    
    discontinuation_rates = []
    pre_rates = []
    post_rates = []
    pre_patent_counts = []

    # 4. Iterate through each deal
    for _, deal in deals_df.iterrows():
        if pd.isna(deal['year']):
            discontinuation_rates.append(np.nan)
            pre_rates.append(np.nan)
            post_rates.append(np.nan)
            pre_patent_counts.append(0)
            continue
        
        
        target_id = str(deal['target_id']).zfill(6)
        merger_year = int(deal['year'])
        
        # Get all patents owned by the target
        target_patents = firm_patents_dict.get(target_id, [])
        
        # Filter to only include patents filed BEFORE the merger year
        pre_merger_patents = [
            pid for pid in target_patents 
            if patent_year_dict.get(pid, 9999) < merger_year
        ]
        
        if not pre_merger_patents:
            # Target had no patents before the merger, cannot be discontinued
            discontinuation_rates.append(0.0)
            pre_rates.append(0.0)
            post_rates.append(0.0)
            pre_patent_counts.append(0)
            continue

        pre_patent_counts.append(len(pre_merger_patents))
            
        pre_merger_cites = 0
        post_merger_cites = 0
        
        # Count citations in the 3-year windows
        for pid in pre_merger_patents:
            cite_years = cited_grouped.get(pid, [])
            
            for cy in cite_years:
                if (merger_year - 3) <= cy < merger_year:
                    pre_merger_cites += 1
                elif merger_year <= cy < (merger_year + 3):
                    post_merger_cites += 1
                    
        # Annualize the rates
        pre_rate = pre_merger_cites / 3.0
        post_rate = post_merger_cites / 3.0
        
        # Calculate the drop-off percentage
        if pre_rate == 0:
            rate = 0.0 # It was already dead
        else:
            drop = (pre_rate - post_rate) / pre_rate
            rate = max(0.0, min(1.0, drop)) # Bound between 0 (no drop) and 1 (100% drop)
            
        discontinuation_rates.append(rate)
        pre_rates.append(pre_rate)
        post_rates.append(post_rate)

    deals_df['patents_discontinued_pct'] = discontinuation_rates
    deals_df["pre_cite_rate"] = pre_rates
    deals_df["post_cite_rate"] = post_rates
    deals_df["pre_patent_count"] = pre_patent_counts
    return deals_df

def add_patent_output_metrics(deals_df: pd.DataFrame, patents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy for "inventor retention"/continued innovation using target patent output.
    Counts patents in [-3,0) vs [0,+3) years around merger year.
    """
    patents = patents_df[["gvkey", "patent_date", "patent_id"]].copy()
    patents["gvkey"] = patents["gvkey"].astype(str).str.zfill(6)
    patents["patent_year"] = pd.to_datetime(patents["patent_date"], errors="coerce").dt.year
    patents = patents.dropna(subset=["patent_year"])

    # firm-year patent counts
    firm_year = patents.groupby(["gvkey", "patent_year"]).size().rename("patent_count").reset_index()
    firm_year = firm_year.sort_values(["gvkey", "patent_year"])

    pre_counts = []
    post_counts = []
    output_drop = []

    # Create a dict for fast lookup (gvkey -> DataFrame slice)
    by_firm = {g: gdf[["patent_year", "patent_count"]].to_numpy() for g, gdf in firm_year.groupby("gvkey")}

    for _, deal in deals_df.iterrows():
        if pd.isna(deal["year"]):
            pre_counts.append(np.nan)
            post_counts.append(np.nan)
            output_drop.append(np.nan)
            continue

        tgt = str(deal["target_id"]).zfill(6)
        merger_year = int(deal["year"])

        arr = by_firm.get(tgt)
        if arr is None:
            pre_counts.append(0)
            post_counts.append(0)
            output_drop.append(0.0)
            continue

        years = arr[:, 0].astype(int)
        counts = arr[:, 1].astype(int)

        pre_mask = (years >= merger_year - 3) & (years < merger_year)
        post_mask = (years >= merger_year) & (years < merger_year + 3)

        pre = int(counts[pre_mask].sum()) if pre_mask.any() else 0
        post = int(counts[post_mask].sum()) if post_mask.any() else 0

        pre_counts.append(pre)
        post_counts.append(post)
        if pre == 0:
            output_drop.append(0.0)
        else:
            drop = (pre - post) / pre
            output_drop.append(float(max(0.0, min(1.0, drop))))

    deals_df["target_patents_pre3yr"] = pre_counts
    deals_df["target_patents_post3yr"] = post_counts
    deals_df["target_patent_output_drop"] = output_drop
    return deals_df

def load_outcome_data(df: pd.DataFrame, *, skip_citations: bool) -> pd.DataFrame:
    # Optional text outcomes (product integration proxy)
    if TEXT_OUTCOMES_PATH.exists():
        text_outcomes = pd.read_csv(TEXT_OUTCOMES_PATH, dtype={"acquiror_id": str, "target_id": str})
        require_columns(text_outcomes, "text_outcomes", ["acquiror_id", "target_id", "year"])
        if "has_product_synergy" not in text_outcomes.columns:
            print(f"Warning: {TEXT_OUTCOMES_PATH} missing 'has_product_synergy'; product proxy disabled.")
            df["product_lines_combined"] = np.nan
        else:
            text_outcomes["acquiror_id"] = text_outcomes["acquiror_id"].str.zfill(6)
            text_outcomes["target_id"] = text_outcomes["target_id"].str.zfill(6)
            df = df.merge(text_outcomes, on=["acquiror_id", "target_id", "year"], how="left")
            df["product_lines_combined"] = df["has_product_synergy"]
    else:
        print(f"Warning: {TEXT_OUTCOMES_PATH} not found; product proxy disabled.")
        df["product_lines_combined"] = np.nan

    # Patent outcomes
    if PATENTS_PATH.exists():
        patents_df = pd.read_parquet(PATENTS_PATH, columns=["gvkey", "patent_id", "patent_date"])
        df = add_patent_output_metrics(df, patents_df)
    else:
        print(f"Warning: {PATENTS_PATH} not found; patent output proxy disabled.")
        df["target_patent_output_drop"] = np.nan

    if skip_citations:
        print("Skipping citation-based discontinuation (--skip-citations).")
        df["patents_discontinued_pct"] = np.nan
        df["pre_cite_rate"] = np.nan
        df["post_cite_rate"] = np.nan
        df["pre_patent_count"] = np.nan
        return df

    if CITATIONS_PATH.exists() and PATENTS_PATH.exists():
        citations_df = pd.read_parquet(CITATIONS_PATH, columns=["patent_id", "citation_id"])
        # Reuse patents_df already loaded above if possible
        try:
            patents_df  # noqa: B018
        except NameError:
            patents_df = pd.read_parquet(PATENTS_PATH, columns=["gvkey", "patent_id", "patent_date"])

        df = calculate_patent_discontinuation(df, patents_df, citations_df)
    else:
        print("Warning: citation discontinuation disabled (missing citations or patents parquet).")
        df["patents_discontinued_pct"] = np.nan
        df["pre_cite_rate"] = np.nan
        df["post_cite_rate"] = np.nan
        df["pre_patent_count"] = np.nan

    return df

    

def infer_strategic_motive(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    df['strategic_motive'] = 'UNKNOWN'
    df['target_distressed'] = ((df['tgt_fcf'] < 0) | (df['tgt_q'] < 1.0)).fillna(False)

    # Counterfactual-lite: "excess" outcomes vs within-year & within-industry baselines.
    df["tgt_sic2"] = pd.to_numeric(df["tgt_sic"], errors="coerce") // 100
    df["disc_med_year"] = df.groupby("year")["patents_discontinued_pct"].transform("median")
    df["disc_med_ind_year"] = df.groupby(["year", "tgt_sic2"])["patents_discontinued_pct"].transform("median")
    df["disc_excess"] = df["patents_discontinued_pct"] - df["disc_med_ind_year"]

    df["out_med_year"] = df.groupby("year")["target_patent_output_drop"].transform("median")
    df["out_med_ind_year"] = df.groupby(["year", "tgt_sic2"])["target_patent_output_drop"].transform("median")
    df["out_excess"] = df["target_patent_output_drop"] - df["out_med_ind_year"]

    disc_hi = df["disc_excess"].quantile(0.75) if df["disc_excess"].notna().any() else np.nan
    out_hi = df["out_excess"].quantile(0.75) if df["out_excess"].notna().any() else np.nan
    if verbose and (pd.notna(disc_hi) or pd.notna(out_hi)):
        print(f"Stage-2 thresholds (data-driven): disc_excess_p75={disc_hi}, out_excess_p75={out_hi}")

    killer_mask = (
        df['tech_motive'].isin(['SUBSTITUTE', 'NICHE_OVERLAP']) &
        (df["pre_patent_count"].fillna(0) >= 10) &
        (df["pre_cite_rate"].fillna(0) >= 1.0) &
        (df["disc_excess"] >= disc_hi) &
        (df["patents_discontinued_pct"] > 0.40) & # hard code patent drop to 40%
        (df["out_excess"] >= out_hi)
    )
    df.loc[killer_mask, 'strategic_motive'] = 'KILLER_ACQUISITION'
    
    vertical_mask = (
        (df['tech_motive'] == 'COMPLEMENTARY') &
        (df['same_sic2'] == 0) & # DIFFERENT industry (e.g., Software buying Hardware)
        (df['patents_discontinued_pct'].fillna(0) < 0.3) & # Target R&D stays alive
        ~killer_mask
    )
    df.loc[vertical_mask, 'strategic_motive'] = 'VERTICAL_INTEGRATION'
    
    firesale_mask = (
        df['target_distressed'] &
        (pd.to_numeric(df.get("deal_value"), errors="coerce").notna()) &
        ((pd.to_numeric(df.get("deal_value"), errors="coerce") / df["tgt_at"]).fillna(np.inf) <= ((pd.to_numeric(df.get("deal_value"), errors="coerce") / df["tgt_at"]).quantile(0.25))) &
        (df['tgt_fcf'] < 0) & # Distressed by cash flow, not just low Quick ratio
        ~killer_mask & ~vertical_mask
    )
    df.loc[firesale_mask, 'strategic_motive'] = 'FIRE_SALE'
    
    horizontal_mask = (
        (df['tech_motive'] == 'SUBSTITUTE') &
        (df['same_sic2'] == 1) & # SAME industry (e.g., Bank buying a Bank)
        (df['patents_discontinued_pct'].fillna(0) < 0.3) &
        ~killer_mask & ~vertical_mask & ~firesale_mask
    )
    df.loc[horizontal_mask, 'strategic_motive'] = 'HORIZONTAL_CONSOLIDATION'
    
    return df

def write_diagnostics_summary(df: pd.DataFrame) -> None:
    numeric_cols = [
        "bc_synergy_score",
        "max_single_component_overlap",
        "component_coverage",
        "patents_discontinued_pct",
        "disc_excess",
        "target_patent_output_drop",
        "out_excess",
        "pre_cite_rate",
        "post_cite_rate",
        "pre_patent_count",
        "target_patents_pre3yr",
        "target_patents_post3yr",
    ]
    existing = [c for c in numeric_cols if c in df.columns]
    if not existing:
        return

    summary_rows = []
    for col in existing:
        s = pd.to_numeric(df[col], errors="coerce")
        summary_rows.append(
            {
                "column": col,
                "n": int(s.shape[0]),
                "missing": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean()),
                "nunique": int(s.nunique(dropna=True)),
                "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
                "std": float(s.std(skipna=True)) if s.notna().any() else np.nan,
                "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
                "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
            }
        )

    diag = pd.DataFrame(summary_rows).sort_values(["missing_pct", "nunique"], ascending=[False, True])
    diag.to_csv(OUTPUT_DIR / "diagnostics_summary.csv", index=False)

    constantish = diag[(diag["nunique"] <= 1) | (diag["std"].fillna(0) == 0)]
    if not constantish.empty:
        cols = ", ".join(constantish["column"].tolist())
        print(f"Diagnostic warning: near-constant numeric columns detected: {cols}")

def run_falsification_tests(df):
    print("\n--- Falsification Tests ---")
    
    # Placebo
    df_placebo = df.copy()
    df_placebo['patents_discontinued_pct'] = np.random.permutation(df_placebo['patents_discontinued_pct'].values)
    if "target_patent_output_drop" in df_placebo.columns:
        df_placebo["target_patent_output_drop"] = np.random.permutation(df_placebo["target_patent_output_drop"].values)
    df_placebo = infer_strategic_motive(df_placebo, verbose=False)
    
    real_killer = (df['strategic_motive'] == 'KILLER_ACQUISITION').sum()
    placebo_killer = (df_placebo['strategic_motive'] == 'KILLER_ACQUISITION').sum()
    print(f"Placebo test (Killer Acquisitions): Real = {real_killer}, Placebo = {placebo_killer}")
    
    # Same-firm
    same_firm = df[df['acquiror_id'] == df['target_id']]
    if not same_firm.empty:
        same_firm_killers = (same_firm['strategic_motive'] == 'KILLER_ACQUISITION').sum()
        print(f"Same-firm killer acquisitions: {same_firm_killers} (Should be 0)")

# --- Visualization & Export ---
def generate_diagnostic_plots(df):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    crosstab = pd.crosstab(df['tech_motive'], df['strategic_motive'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('Tech Motive vs Strategic Motive')
    
    killer = df[df['strategic_motive'] == 'KILLER_ACQUISITION']
    other = df[df['strategic_motive'].isin(['HORIZONTAL_CONSOLIDATION', 'VERTICAL_INTEGRATION'])]
    
    axes[1].scatter(other['bc_synergy_score'], other['patents_discontinued_pct'], alpha=0.5, label='Other', s=20)
    axes[1].scatter(killer['bc_synergy_score'], killer['patents_discontinued_pct'], alpha=0.7, label='Killer', s=20, color='red')
    axes[1].set_xlabel('BC Synergy Score')
    axes[1].set_ylabel('Patents Discontinued (%)')
    axes[1].set_title('Killer Acquisitions vs Other')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'diagnostic_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Running M&A Motive Classification...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-deals", type=int, default=None, help="Limit number of deals for quick runs.")
    parser.add_argument("--skip-citations", action="store_true", help="Skip citation-based discontinuation (fast).")
    parser.add_argument("--skip-plots", action="store_true", help="Skip saving diagnostic plots.")
    args = parser.parse_args()

    require_file(DEALS_PATH)
    require_file(FIN_PATH)
    require_file(GMM_K15)

    deals = pd.read_csv(DEALS_PATH)
    fin = pd.read_csv(FIN_PATH, low_memory=False)

    require_columns(deals, "deals", ["acquiror_id", "target_id", "year", "bc_synergy_score"])
    require_columns(fin, "fin", ["gvkey", "year", "at", "sich", "free_cash_flow", "tobins_q_filled"])
    
    fin['gvkey'] = fin['gvkey'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
    fin['year'] = pd.to_numeric(fin['year'], errors='coerce')
    deals['join_year'] = deals['year'] - 1
    deals['acquiror_id'] = deals['acquiror_id'].astype(str).str.zfill(6)
    deals['target_id'] = deals['target_id'].astype(str).str.zfill(6)

    if args.max_deals is not None:
        deals = deals.sort_values("year").head(args.max_deals)
    
    fin_subset = fin[['gvkey', 'year', 'at', 'sich', 'free_cash_flow', 'tobins_q_filled']]
    
    df = deals.merge(fin_subset, left_on=['acquiror_id', 'join_year'], right_on=['gvkey', 'year'], how='inner')
    df = df.rename(columns={'at': 'acq_at', 'sich': 'acq_sic', 'free_cash_flow': 'acq_fcf', 'tobins_q_filled': 'acq_q'})
    
    df = df.merge(fin_subset, left_on=['target_id', 'join_year'], right_on=['gvkey', 'year'], how='left', suffixes=('', '_tgt'))
    df = df.rename(columns={'at': 'tgt_at', 'sich': 'tgt_sic', 'free_cash_flow': 'tgt_fcf', 'tobins_q_filled': 'tgt_q'})

    gmm_lookup = load_gmm_lookup(GMM_K15)
    
    # Execute pipeline
    df = compute_technology_features(df, gmm_lookup)
    df = classify_technology_motive(df)
    validate_stage1(df)
    
    df = load_outcome_data(df, skip_citations=args.skip_citations)
    df = infer_strategic_motive(df)
    write_diagnostics_summary(df)
    run_falsification_tests(df)
    if not args.skip_plots:
        generate_diagnostic_plots(df)

    # Diagnostics export
    diag_cols = [
        "acquiror_id",
        "target_id",
        "year",
        "tech_motive",
        "strategic_motive",
        "bc_synergy_score",
        "max_single_component_overlap",
        "component_coverage",
        "patents_discontinued_pct",
        "disc_excess",
        "target_patent_output_drop",
        "out_excess",
        "pre_cite_rate",
        "post_cite_rate",
        "pre_patent_count",
        "target_patents_pre3yr",
        "target_patents_post3yr",
        "product_lines_combined",
        "tgt_sic",
        "same_sic2",
        "same_sic4",
    ]
    existing = [c for c in diag_cols if c in df.columns]
    df[existing].to_csv(OUTPUT_DIR / "diagnostics_deal_level.csv", index=False)

    # Main outputs (kept stable for downstream use)
    df.to_csv(OUTPUT_DIR / "classified_deals_two_stage.csv", index=False)
    with open(OUTPUT_DIR / "classification_summary.txt", "w", encoding="utf-8") as f:
        f.write("Tech motive counts:\n")
        f.write(df["tech_motive"].value_counts(dropna=False).to_string())
        f.write("\n\nStrategic motive counts:\n")
        f.write(df["strategic_motive"].value_counts(dropna=False).to_string())
        f.write("\n")
    
    print(f"\nPipeline complete. Outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
