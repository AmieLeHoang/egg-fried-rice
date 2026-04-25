# Quantitative M&A Prediction & Target Screening Pipeline

This project develops a quantitative framework to predict mergers and acquisitions by embedding patent text into a geometric space and calculating topological measurements of latent synergy. 

By mapping firm-level patent portfolios using Gaussian Mixture Models (GMM) and PatentSBERT embeddings, this pipeline isolates the drivers of technology-driven M&A, ranks likely acquisition targets, and classifies strategic motives (e.g., Killer Acquisitions, Vertical Integration).

## Pipeline Architecture

The repository is structured sequentially to process raw patent embeddings, compute synergy metrics, train ranking models, and rigorously validate the results. 

* **`01_get_bc_matrix.py` (Topological Synergy Mapping):** Validates and maps the Bhattacharyya Coefficient (BC) matrix derived from firm-level GMMs. This script enforces mathematical integrity checks on the component-pair BC formulas and merges the topological synergy scores with historical M&A deal rosters and co-assignment audit controls.
* **`02_mass.py` (MASS Synergy Computation):** Computes the Mean Average Semantic Similarity (MASS) score. It ingests 50-dimensional PatentSBERT embeddings, calculates the firm-level average vectors, and applies cosine similarity to flag product synergies across acquiring and target firms.
* **`03_ma_target_screening.py` (Target Ranking Engine):** The core deal-sourcing model. It trains an `XGBRanker` or Logistic Regression model on historical deals to predict future targets. It scores candidates based on a blend of pre-computed BC synergy, shared patents, and financial metrics (Tobin's Q, Free Cash Flow, size ratios).
* **`04_ma_motive.py` (Strategic Motive Classification):** Conducts ex-ante and ex-post classification of M&A motives. It maps technology buckets (Substitute, Complementary, Niche Overlap) and infers strategic motives (Horizontal Consolidation, Vertical Integration, Killer Acquisition, Fire Sale) by synthesizing synergy signals, financial distress indicators, and post-merger patent discontinuation drop-offs.
* **`08_robustnes_checks.py` (Econometric Audit):** A comprehensive validation script ensuring the statistical reliability of the GMM pipeline. It executes four critical tests: Gaussianity audits, Dirichlet prior sensitivity analysis, external validity benchmarks against SIC/citation networks, and top-k bootstrap stability.

## Mathematical Underpinnings

The pipeline relies heavily on the closed-form Bhattacharyya Coefficient to measure the overlap between diagonal Gaussian components $N_i^A$ and $N_j^B$. The distance $D_B$ and resulting $BC$ are computed as:

$$D_B = \frac{1}{8} \sum_{d} \frac{(\mu_{i,d} - \mu_{j,d})^2}{\bar{\sigma}_d^2} + \frac{1}{2} \sum_{d} \ln\left(\frac{\bar{\sigma}_d^2}{\sqrt{\sigma_{i,d}^2 \cdot \sigma_{j,d}^2}}\right)$$

$$BC(N_i^A, N_j^B) = \exp(-D_B)$$

Where, 

$$\bar{\sigma}_d^2 = \frac{\sigma_{i,d}^2 + \sigma_{j,d}^2}{2}$$

The mixture-level synergy is then aggregated using linear $\pi_i \pi_j$ weights to maintain stability across the matrix.


## Data Requirements

To execute the scripts, ensure your local `data/` directory is populated with the necessary bundle files from the research team. 

**Raw Data (`data/raw/`)**
* `corrected/bc_matrix_all_k15_dedup_linear.npz`: Pre-computed BC matrix.
* `corrected/firm_gmm_parameters_k15.parquet`: Base GMM parameters.
* `patents/firm_patents_text_metadata_techbio.parquet`: Patent metadata.
* `patents/citation_network_techbio.parquet`: Citation network edges.
* `clean_financials.csv`: Compustat financial metrics.
* `compustat_names.csv`: GVKEY to firm name crosswalk.
* `teammate_deal_roster.csv`: Historical M&A deals list.

**Processed Data (`data/processed/`)**
* `patent_vectors_50d.parquet`: 50D patent vector embeddings.
* `gvkey_map.parquet`: Mapping of patent IDs to GVKEYs.

## Execution Guide

Activate your virtual environment and install dependencies (`requirements.txt` and `requirements-xgboost.txt`) before running the pipeline sequentially:

1.  **Generate base synergy datasets:**
    `python 01_get_bc_matrix.py`
    `python 02_mass.py`
2.  **Run target screening:** Train the model for a specific year and acquirer (e.g., GVKEY 007257).
    `python 03_ma_target_screening.py --acquirer-gvkey 007257 --train-year 2020 --model xgb`
3.  **Classify deal motives:**
    `python 04_ma_motive.py`
4.  **Validate assumptions:** Run the robustness checks to output diagnostic plots and tables.
    `python 08_robustnes_checks.py`

Output CSVs, diagnostic datasets, and matplotlib figures will be written directly to the `output/` directory.

