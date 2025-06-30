# find_coint_cluster.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import warnings

# Suppress warnings from the statsmodels, which can be noisy
warnings.filterwarnings("ignore")

def find_top_central_assets(price_df: pd.DataFrame, num_to_check=10):
    """Finds the top N assets that have the highest average correlation."""
    corr_matrix = price_df.corr()
    avg_corr = corr_matrix.mean(axis=1)
    # Return the indices of the top N most central assets
    return avg_corr.nlargest(num_to_check).index.tolist()

def form_cluster_around_asset(price_df: pd.DataFrame, center_asset_index: int, num_assets_in_cluster=5):
    """Forms a cluster of the most correlated assets around a given center asset."""
    corr_matrix = price_df.corr()
    cluster_asset_indices = corr_matrix[center_asset_index].nlargest(num_assets_in_cluster).index.tolist()
    return cluster_asset_indices

def run_johansen_analysis(data_for_test: pd.DataFrame):
    """Runs the Johansen test and returns the results and number of relationships."""
    result = coint_johansen(data_for_test, det_order=0, k_ar_diff=1)
    trace_stat = result.lr1
    crit_vals_95 = result.cvt[:, 1]
    
    num_relationships = 0
    for i in range(len(trace_stat)):
        if trace_stat[i] > crit_vals_95[i]:
            num_relationships = i + 1
            
    return result, num_relationships

if __name__ == '__main__':
    # --- SETUP ---
    # We will use the full 750 days of data to find the most stable relationships
    prices_df = pd.read_csv('prices.txt', header=None, sep=r'\s+')
    
    print("--- Automated Search for Cointegrated Portfolio ---")

    # --- Step 1: Find the most promising assets to build clusters around ---
    print("\n1. Identifying top 10 most central assets...")
    central_assets = find_top_central_assets(prices_df, num_to_check=10)
    print(f"   -> Will test clusters centered around: {central_assets}")
    
    found_portfolio = False

    # --- Step 2: Loop through potential clusters and test for cointegration ---
    for center_asset in central_assets:
        print(f"\n2. Testing cluster centered around Asset {center_asset}...")
        
        # Form the cluster
        asset_indices_to_test = form_cluster_around_asset(prices_df, center_asset, num_assets_in_cluster=5)
        cluster_df = prices_df.iloc[:, asset_indices_to_test]
        
        # Run the Johansen test
        johansen_result, num_vectors = run_johansen_analysis(cluster_df)
        
        # --- Step 3: If a cointegrated cluster is found, analyze and plot it ---
        if num_vectors > 0:
            print(f"\nSUCCESS! Found a cointegrated cluster with {num_vectors} relationship(s): {asset_indices_to_test}")
            
            print("\n3. Constructing and plotting the mean-reverting portfolio(s)...")
            
            for i in range(num_vectors):
                vector = johansen_result.evec[:, i]
                
                print(f"\n--- Portfolio #{i+1} ---")
                print("Weights (Cointegrating Vector):")
                print(pd.Series(vector, index=asset_indices_to_test))
                
                portfolio_series = np.dot(cluster_df.to_numpy(), vector)
                
                plt.figure(figsize=(12, 6))
                plt.plot(portfolio_series)
                plt.title(f'Mean-Reverting Portfolio #{i+1} from Johansen Test (Cluster {asset_indices_to_test})')
                plt.xlabel('Day')
                plt.ylabel('Portfolio Value (Spread)')
                plt.axhline(np.mean(portfolio_series), color='r', linestyle='--', label='Mean')
                plt.grid(True)
                plt.legend()
                plt.show()

            found_portfolio = True
            # We found a working cluster, so we can stop the search
            break
        else:
            print(f"   -> Cluster {asset_indices_to_test} is not cointegrated. Trying next...")

    if not found_portfolio:
        print("\n--- Search Complete: No highly correlated 5-asset clusters were found to be cointegrated. ---")