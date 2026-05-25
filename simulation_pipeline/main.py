import numpy as np
import yaml
import os
from data_generation import  generate_linear_hmm, compute_empirical_overlaps

### Utility function to load configuration

def load_config(config_path):
    """Loads the experiment configuration file safely."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

### Main execution block 

if __name__ == "__main__":
    # Path handling to ensure it runs from any directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "../configs/linear_baseline.yaml")
    
    # Run the generation pipeline
    config = load_config(config_file)
    X, Y, V = generate_linear_hmm(config)
    Q_XY = compute_empirical_overlaps(X, Y)
    
    print(f"--- HMM Sandbox Initialized Successfully ---")
    print(f"Modality 1 Data Shape (X): {X.shape}")
    print(f"Modality 2 Data Shape (Y): {Y.shape}")
    print(f"Empirical Overlap Matrix Shape: {Q_XY.shape}")
    print(f"Top-left 3x3 block of empirical alignment:\n{Q_XY[:3, :3]}")