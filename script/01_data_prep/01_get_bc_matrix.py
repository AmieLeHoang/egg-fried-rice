"""
01_get_bc_matrix.py

This script loads the corrected BC matrix and GMM parameters, verifies their integrity, and prepares them for use in 
later analysis and validation steps. It includes functions to compute the BC score between any two firms based on their 
GMM parameters, and it maps these scores to the M&A deals dataset for validation. 

The script also includes sanity checks to ensure the BC matrix is correctly structured and that the pre-computed BC scores 
match the mathematical formula. The final output is a set of functions and data structures that can be used in the validation 
script to test the predictive power of the BC scores against real-world M&A outcomes.

Inputs:
- data/raw/corrected/bc_matrix_all_k15_dedup_linear.npz: The corrected BC matrix for K=15, post-deduplication.
- data/raw/corrected/firm_gmm_parameters_k15.parquet: The GMM parameters for each firm corresponding to the BC matrix. 
- data/raw/teammate_deal_roster.csv: The M&A deals dataset to which we will map the BC scores for validation.

Outputs:
- data/processed/modeling_dataset_with_synergy.csv: The M&A deals dataset with mapped BC synergy scores and co-assignment controls

"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


if not Path("output/kmax_sweep").exists() and Path("../output/kmax_sweep").exists():
    os.chdir("..")
    print(f"Changed working directory to repo root: {os.getcwd()}")

# ---------------------------- CONFIGURATION & PATHS ----------------------------
CORRECTED = Path("data/raw/corrected")
SWEEP = Path("data/raw/kmax_sweep")
GMM_K15 = CORRECTED / "firm_gmm_parameters_k15.parquet"
BC_K15 = CORRECTED / "bc_matrix_all_k15_dedup_linear.npz"
DEDUP_CSV = SWEEP / "deduplication_decisions.csv"
EXCLUDED_CSV = SWEEP / "excluded_firms.csv"
COASSIGN_PARQUET = SWEEP / "coassignment_audit.parquet"
DEALS_PATH = Path("data/raw/teammate_deal_roster.csv")
deals = pd.read_csv(DEALS_PATH, low_memory=False)
OUTPUT_PATH = Path("data/processed/modeling_dataset_with_synergy.csv")


# Fail early with a helpful message if teammates haven't placed the bundle files yet
for p in [GMM_K15, BC_K15, DEDUP_CSV, EXCLUDED_CSV, COASSIGN_PARQUET]:
    assert p.exists(), f"Missing {p}. Did you place the bundle files per Section 1?"

print("All bundle files present. Ready to load.")


# --------- FIRM NAME MAPPING ---------
FIRM_NAMES = {
    "006066": "IBM",
    "012141": "Microsoft", 
    "024800": "Qualcomm",
    "160329": "Google / Alphabet",
    "020779": "Cisco Systems",
    "006008": "Intel Corp",
    "005606": "HP Inc",
    "007343": "Micron Technology Inc",
    "001690": "Apple Inc",
    "011636": "Xerox Holdings Corp",
    "007585": "Motorola Solutions Inc",
    "001300": "Honeywell International Inc",
    "007435": "3M CO",
    "009899": "AT&T Inc",
    "001161": "AMD",
    "004060": "Dupont de Nemours Inc",
    "006266": "Johnson & Johnson",
    "PRIV_BAYERHEALTHCARE": "Bayer Healthcare (private)",
    "007257": "Merk & Co Inc",
    "001704": "Applied Materials Inc",
    "012142": "Oracle Corp",
    "PRIV_GENERALMETERS": "General Meters (private)",
    "007228": "Medtronic PLC",
    "066708": "Broadcom Corp",
    "003532": "Corning Inc",
    "PRIV_SCHLUMBERGERTECHNOLOGY": "Schlumberger Technology (private)",
    "170617": "Meta Platforms Inc",
    "002136": "Verizon Communications Inc",
    "001078": "Abbott Laboratories",
    "137310": "Marvell Technology Inc"

}

# ------------------- DATA LOADING FUNCTIONS --------------------
def load_bc_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Load a corrected BC matrix .npz archive.

    Returns (gvkeys, bc_matrix). The matrix is symmetric float64 with diagonal 1.0.
    """
    data = np.load(path, allow_pickle=True)
    gvkeys = [str(g) for g in data["gvkeys"]]
    bc_matrix = data["bc_matrix"]
    assert bc_matrix.shape == (len(gvkeys), len(gvkeys)), "gvkey/matrix shape mismatch"
    return gvkeys, bc_matrix


def load_gmm_results(path: Path) -> dict[str, dict]:
    """Load per-firm GMM parameters from parquet; return dict keyed by gvkey."""
    df = pd.read_parquet(path)
    d = 50  # UMAP output dimensionality
    lookup = {}
    for _, row in df.iterrows():
        k = int(row["n_components"])
        lookup[str(row["gvkey"])] = {
            "gvkey": str(row["gvkey"]),
            "n_patents": int(row["n_patents"]),
            "n_components": k,
            "tier": row["tier"],
            "means": np.frombuffer(row["means"], dtype=np.float64).reshape(k, d),
            "covariances": np.frombuffer(row["covariances"], dtype=np.float64).reshape(k, d),
            "weights": np.frombuffer(row["weights"], dtype=np.float64).reshape(k),
        }
    return lookup


gvkeys, bc_matrix = load_bc_matrix(BC_K15)
gvkey_to_idx = {gv: i for i, gv in enumerate(gvkeys)}
gmm_lookup = load_gmm_results(GMM_K15)

print(f"BC matrix: {bc_matrix.shape}, diagonal mean = {np.diag(bc_matrix).mean():.6f}")
print(f"GMM parameters: {len(gmm_lookup):,} firms pre-dedup")
print(f"Deduplicated set (BC matrix rows): {len(gvkeys):,} firms")

# Sanity checks — if any fail, the artifacts are corrupted or mismatched.
assert bc_matrix.shape[0] == bc_matrix.shape[1] == len(gvkeys), "BC matrix is not square"
assert np.allclose(bc_matrix, bc_matrix.T, atol=1e-12), "BC matrix is not symmetric"
assert np.allclose(np.diag(bc_matrix), 1.0, atol=1e-9), "BC diagonal is not all 1.0"
assert (bc_matrix >= 0).all() and (bc_matrix <= 1 + 1e-9).all(), "BC values outside [0, 1]"

# Every BC-matrix firm should have GMM parameters
missing = [gv for gv in gvkeys if gv not in gmm_lookup]
assert not missing, f"{len(missing)} firms in BC matrix but missing GMM params"
print("All sanity checks pass.")


# ---------------------- BC CALCULATION -----------------------
def bc_component_matrix(mu_a: np.ndarray, var_a: np.ndarray,
                       mu_b: np.ndarray, var_b: np.ndarray) -> np.ndarray:
    """BC between all component pairs of two GMMs (diagonal covariance, vectorized).

    Closed form under diagonal covariance (ADR-006):
        D_B = (1/8) Σ_d (μᵢ_d - μⱼ_d)² / σ̄²_d
            + (1/2) Σ_d ln(σ̄²_d / √(σ²ᵢ_d · σ²ⱼ_d))
        BC  = exp(-D_B)
    where σ̄²_d = (σ²ᵢ_d + σ²ⱼ_d) / 2.

    Returns (K_A, K_B) float64 matrix of component-pair BC values in [0, 1].
    """
    sigma_avg = (var_a[:, None, :] + var_b[None, :, :]) / 2.0   # (K_A, K_B, D)
    diff = mu_a[:, None, :] - mu_b[None, :, :]                  # (K_A, K_B, D)
    mahal = 0.125 * np.sum(diff**2 / sigma_avg, axis=2)         # (K_A, K_B)
    log_det_avg = np.sum(np.log(sigma_avg), axis=2)             # (K_A, K_B)
    log_det_a = np.sum(np.log(var_a), axis=1)                   # (K_A,)
    log_det_b = np.sum(np.log(var_b), axis=1)                   # (K_B,)
    det_term = 0.5 * (log_det_avg - 0.5 * (log_det_a[:, None] + log_det_b[None, :]))
    return np.exp(-(mahal + det_term))


def bc_mixture_linear(gmm_a: dict, gmm_b: dict) -> float:
    """Mixture-level BC with linear πᵢπⱼ weights (bounded in [0, 1]).

        BC(A, B) = Σᵢ Σⱼ πᵢᴬ · πⱼᴮ · BC(Nᵢᴬ, Nⱼᴮ)

    Do NOT use √(πᵢπⱼ) — that is an upper bound that can exceed 1 for multi-
    component mixtures and caused the original K_max top-tail instability bug.
    """
    bc_grid = bc_component_matrix(
        gmm_a["means"], gmm_a["covariances"],
        gmm_b["means"], gmm_b["covariances"],
    )
    weight_grid = gmm_a["weights"][:, None] * gmm_b["weights"][None, :]
    return float(np.sum(weight_grid * bc_grid))


# -------------------- MAP SCORES TO DEALS & VERIFY ---------------------
# Standardize IDs to match the gvkeys format
deals['acquiror_id'] = deals['acquiror_id'].astype(str).str.replace(r'\.0$', '', regex=True)
deals['target_id'] = deals['target_id'].astype(str).str.replace(r'\.0$', '', regex=True)

# Filter deals: BOTH firms must exist in the BC Matrix AND they cannot be the same firm
valid_deals = deals[
    (deals['acquiror_id'].isin(gvkey_to_idx)) & 
    (deals['target_id'].isin(gvkey_to_idx)) & 
    (deals['acquiror_id'] != deals['target_id'])
].copy()

# Extract BC Scores and Verify
print("Extracting pairwise BC scores and verifying against mathematical formula...")
bc_scores_matrix = []
bc_scores_formula = []

for idx, row in valid_deals.iterrows():
    acq_gvkey = row['acquiror_id']
    tgt_gvkey = row['target_id']
    
    # 1. Fast Matrix Lookup (O(1))
    i_acq = gvkey_to_idx[acq_gvkey]
    i_tgt = gvkey_to_idx[tgt_gvkey]
    matrix_score = float(bc_matrix[i_acq, i_tgt])
    bc_scores_matrix.append(matrix_score)
    
    # 2. Mathematical Recomputation to Verify Correctness
    formula_score = bc_mixture_linear(gmm_lookup[acq_gvkey], gmm_lookup[tgt_gvkey])
    bc_scores_formula.append(formula_score)
    
    # 3. Sanity check: Ensure the pre-computed matrix matches the formula perfectly
    # (Tolerance set to 1e-9 to account for float64 floating-point arithmetic)
    assert abs(matrix_score - formula_score) < 1e-9, f"Mismatch for pair {acq_gvkey}-{tgt_gvkey}"

valid_deals['bc_synergy_score'] = bc_scores_matrix
print("Verification complete! Matrix perfectly matches the mathematical formula.")

# -------------------- MERGE CO-ASSIGNMENT CONTROLS ---------------------
print("\nMerging co-assignment audit controls...")
audit = pd.read_parquet(COASSIGN_PARQUET)

# Audit might list pairs as (A, B) or (B, A), need to merge safely regardless of order.
# Create an order-agnostic composite key for both the audit and the deals.
def make_pair_key(a, b):
    return tuple(sorted([str(a), str(b)]))
    
audit['pair_key'] = audit.apply(lambda r: make_pair_key(r['gvkey_a'], r['gvkey_b']), axis=1)
audit_lookup = audit.set_index('pair_key')[['n_shared', 'jaccard', 'overlap_fraction']].to_dict('index')

n_shared_list = []
jaccard_list = []

for idx, row in valid_deals.iterrows():
    pkey = make_pair_key(row['acquiror_id'], row['target_id'])
    if pkey in audit_lookup:
        n_shared_list.append(audit_lookup[pkey]['n_shared'])
        jaccard_list.append(audit_lookup[pkey]['jaccard'])
    else:
        # If they aren't in the top-100 audit, it means they share statistically negligible/zero patents
        n_shared_list.append(0)
        jaccard_list.append(0.0)
        
valid_deals['shared_patents'] = n_shared_list
valid_deals['jaccard_similarity'] = jaccard_list

print("\nDataset ready")
print(valid_deals[['acquiror_id', 'target_id', 'bc_synergy_score', 'shared_patents']].head())

# Save
valid_deals.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved final regression dataset to: {OUTPUT_PATH}")