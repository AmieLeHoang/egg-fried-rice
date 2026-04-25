"""

Script: 02_mass.py
Description: This script builds the dataset of M&A deals with MASS synergy scores and flags. It merges the patent vector 
embeddings with the M&A deal data, computes the cosine similarity between the acquiror and target firm vectors to derive 
the MASS synergy score, and then establishes a binary flag for product synergy based on a threshold. The final output is 
saved as a CSV file for use in later analysis and regression steps.

Inputs:
- data/processed/modeling_dataset_with_synergy.csv: The M&A deal dataset with basic synergy scores and flags.
- data/processed/patent_vectors_50d.parquet: The patent vector embeddings for each firm.
- data/processed/gvkey_map.parquet: A mapping file to link patent IDs to gvkeys. 

Outputs:
- data/processed/patentbert_mass_outcomes.csv: The final dataset with MASS synergy scores and flags for each M&A deal.


"""


import pandas as pd
import numpy as np
from pyarrow import json
from sklearn.metrics.pairwise import cosine_similarity

def build_synergy_dataset():
    """Builds the dataset of M&A deals with MASS synergy scores and flags."""
    deals = pd.read_csv("data/processed/modeling_dataset_with_synergy.csv")
    embeddings_df = pd.read_parquet("data/processed/patent_vectors_50d.parquet")
    gvkey_map = pd.read_parquet("data/processed/gvkey_map.parquet")

    print("Linking vectors to firms...")
    # Merge the map and the vectors (Update 'patent_id' to whatever the shared ID column is)
    mapped_vectors = embeddings_df.merge(gvkey_map, on='patent_id', how='inner')
    mapped_vectors['gvkey'] = mapped_vectors['gvkey'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)

    def parse_vector(v):
            if isinstance(v, bytes):
                # 200 bytes = 50 dimensions of float32
                if len(v) == 200:
                    return np.frombuffer(v, dtype=np.float32)
                # 400 bytes = 50 dimensions of float64
                elif len(v) == 400:
                    return np.frombuffer(v, dtype=np.float64)
                else:
                    # Fallback just in case
                    return np.frombuffer(v, dtype=np.float64)
            
            # If it somehow ended up as a string (from a CSV intermediate step)
            if isinstance(v, str):
                v = v.strip()
                if ',' in v:
                    import json
                    return np.array(json.loads(v), dtype=float)
                else:
                    return np.fromstring(v.strip('[]'), sep=' ', dtype=float)
                    
            return np.array(v, dtype=float)

    # Apply the conversion to the entire column
    mapped_vectors['embedding'] = mapped_vectors['embedding'].apply(parse_vector)

    print("Calculating firm-level average vectors...")
    # Group by firm and average their 50D vectors to get one master vector per firm
    # (Assuming the vector is stored as a list/array in a column named 'embedding')
    firm_vectors = mapped_vectors.groupby('gvkey')['embedding'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).reset_index()


    # 2. Merge Acquirer embeddings
    deals['acquiror_id'] = deals['acquiror_id'].astype(str).str.zfill(6)
    deals['target_id'] = deals['target_id'].astype(str).str.zfill(6)

    df = deals.merge(
            firm_vectors, 
            left_on='acquiror_id', 
            right_on='gvkey', 
            how='inner'
        ).rename(columns={'embedding': 'acq_vector'})

    # 3. Merge Target embeddings
    df = df.merge(
            firm_vectors, 
            left_on='target_id', 
            right_on='gvkey', 
            how='inner'
        ).rename(columns={'embedding': 'tgt_vector'})

    # 4. Compute Cosine Similarity (The MASS Synergy Score)
    synergy_scores = []
    for _, row in df.iterrows():
            acq_vec = np.array(row['acq_vector']).reshape(1, -1)
            tgt_vec = np.array(row['tgt_vector']).reshape(1, -1)
            
            sim = cosine_similarity(acq_vec, tgt_vec)[0][0]
            synergy_scores.append(sim)

    df['mass_synergy_score'] = synergy_scores

    # Establish the binary flag (e.g., top 25% of overlaps are flagged as 1)
    threshold = df['mass_synergy_score'].quantile(0.75) 
    df['has_product_synergy'] = (df['mass_synergy_score'] >= threshold).astype(int)

    # Save the final file
    final_out = df[['acquiror_id', 'target_id', 'year', 'mass_synergy_score', 'has_product_synergy']]
    final_out.to_csv("data/processed/patentbert_mass_outcomes.csv", index=False)
        
    print(f"Saved {len(final_out)} deals to data/processed/patentbert_mass_outcomes.csv")
    print(f"Threshold for product synergy flag: Cosine Similarity >= {threshold:.3f}")

if __name__ == "__main__":
    build_synergy_dataset()