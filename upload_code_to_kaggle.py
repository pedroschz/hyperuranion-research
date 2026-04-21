import os
import kagglehub

# Replace these with your actual Kaggle username and a name for your dataset
KAGGLE_USERNAME = "pedrosanchezgil"
DATASET_SLUG = "semantic-autoencoder-code"

handle = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"
local_dataset_dir = "." # Current directory

# We want to ignore everything except the python files, requirements.txt, and README if any.
# Or better, just ignore the bulky/private directories.
ignore_patterns = [
    "terminals/",
    "wandb/",
    "__pycache__/",
    "*.pt",             # Ignore model checkpoints
    ".git/",
    "agent-transcripts/",
    ".cursor/",
    "*.csv",            # e.g., rd_curve.csv
]

print(f"Uploading codebase to Kaggle as dataset: {handle}...")
print(f"Ignoring patterns: {ignore_patterns}")

try:
    kagglehub.dataset_upload(
        handle, 
        local_dataset_dir, 
        version_notes='Latest codebase version',
        ignore_patterns=ignore_patterns
    )
    print("Upload complete! You can now attach this dataset to your Kaggle Notebook.")
    print("In your Kaggle Notebook, you can append the dataset path to sys.path to import your modules:")
    print("import sys")
    print(f"sys.path.append('/kaggle/input/{DATASET_SLUG}')")
except Exception as e:
    print(f"Upload failed: {e}")
    print("\nPlease ensure you have authenticated with Kaggle.")
    print("You can authenticate by running: kagglehub.login()")
