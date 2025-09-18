import numpy as np
import os
from toolkit.src.facescape_bm import facescape_bm

# Configuration
MODEL_PATH = "./toolkit/bilinear_model_v1_6/facescape_bm_v1.6_847_50_52_id_front.npz"
OUTPUT_BASE_DIR = "./benchmark/pred/fswild_pred/diff_private/"
NUM_MESHES = 10  # Number of meshes to generate
EPSILONS = [1.0, 5.0, 10.0]  # Different epsilon values to test
DELTA = 1e-7 # Small delta value for (epsilon, delta)-DP

def generate_and_save_dp_mesh(model, epsilon, delta, index):
    """Generates a single differentially private mesh and saves it."""
    # Create a random clean identity vector (can be replaced by a specific one if needed)
    # Using the model's id_mean and id_var for a realistic starting point
    random_id_vec = np.random.normal(model.id_mean, np.sqrt(model.id_var))
    
    # Create a random expression vector (can be replaced by a specific one)
    # Ensure exp_vec has the correct dimension (model.shape_bm_core.shape[1])
    exp_dim = model.shape_bm_core.shape[1] # Corrected to 52
    exp_vec = np.zeros(exp_dim) # Use a neutral/mean expression with correct dimension
    if hasattr(model, 'exp_gmm_means') and model.exp_gmm_means.shape[0] > 0:
        gmm_exp_dim = model.exp_gmm_means.shape[1]
        if gmm_exp_dim == exp_dim:
            exp_vec = model.exp_gmm_means[0]
        elif gmm_exp_dim < exp_dim:
            # Pad with zeros if gmm_exp_dim is smaller than exp_dim
            exp_vec[:gmm_exp_dim] = model.exp_gmm_means[0]
            print(f"Warning: exp_gmm_means dimension ({gmm_exp_dim}) is smaller than expected ({exp_dim}). Padding with zeros.")
        else: # gmm_exp_dim > exp_dim
            # Truncate if gmm_exp_dim is larger than exp_dim
            exp_vec = model.exp_gmm_means[0, :exp_dim]
            print(f"Warning: exp_gmm_means dimension ({gmm_exp_dim}) is larger than expected ({exp_dim}). Truncating.")
    
    # Generate the differentially private mesh
    dp_mesh = model.gen_full_dp(random_id_vec, exp_vec, epsilon, delta)

    # Define output directory and file name
    output_dir_for_export = os.path.join(OUTPUT_BASE_DIR, f"eps_{epsilon}") # Corrected: use output_dir_for_export as directory
    os.makedirs(output_dir_for_export, exist_ok=True) # Ensure directory exists
    file_name_prefix = str(index).zfill(3) # File name without .obj
    
    # Export the mesh
    dp_mesh.export(output_dir_for_export, file_name_prefix) # Corrected: pass directory and file name prefix
    print(f"Generated DP mesh for epsilon={epsilon} at {os.path.join(output_dir_for_export, file_name_prefix + '.obj')}") # Update print statement

def main():
    print("Loading FaceScape Bilinear Model...")
    model = facescape_bm(MODEL_PATH)
    print("Model loaded.")

    for epsilon in EPSILONS:
        print(f"Generating meshes for epsilon = {epsilon}...")
        for i in range(NUM_MESHES):
            generate_and_save_dp_mesh(model, epsilon, DELTA, i)
    
    print("\nGeneration complete. You can now run the benchmark evaluator.")
    print(f"Example command: python benchmark/code/evaluator.py --dataset fswild --method diff_private --num {NUM_MESHES}")

if __name__ == "__main__":
    main()
