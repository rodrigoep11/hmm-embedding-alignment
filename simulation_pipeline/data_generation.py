import numpy as np
import yaml
import os

def load_config(config_path):
    """Loads the experiment configuration file safely."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def generate_linear_hmm(config):
    """
    Generates a linear Hidden Manifold Model baseline.
    
    Latent space: V ~ N(0, I_d)
    Modality 1:  X = W1 * V  (W1 is D1 x d random matrix)
    Modality 2:  Y = W2 * V  (W2 is D2 x d random matrix)
    """
    p = config['dimensions']
    np.random.seed(config['simulation']['seed'])
    
    # 1. Generate the underlying shared latent manifold data (N x d)
    V = np.random.normal(0, 1, size=(p['num_samples'], p['latent_dim']))
    
    # 2. Generate random projection matrices (W / sqrt(d) scaling)
    W1 = np.random.normal(0, 1, size=(p['ambient_dim_1'], p['latent_dim'])) / np.sqrt(p['latent_dim'])
    W2 = np.random.normal(0, 1, size=(p['ambient_dim_2'], p['latent_dim'])) / np.sqrt(p['latent_dim'])
    
    # 3. Project to the observed modalities
    X = V @ W1.T  # Shape: N x D1
    Y = V @ W2.T  # Shape: N x D2
    
    # Add minor measurement noise
    X += np.random.normal(0, config['simulation']['noise_level'], size=X.shape)
    Y += np.random.normal(0, config['simulation']['noise_level'], size=Y.shape)
    
    return X, Y, V

def compute_empirical_overlaps(X, Y):
    """Computes the cross-correlation/overlap matrix between modalities."""
    # Standardize data to have zero mean
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Empirical covariance matrix (D1 x D2)
    Q_XY = (X_centered.T @ Y_centered) / X.shape[0]
    return Q_XY

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