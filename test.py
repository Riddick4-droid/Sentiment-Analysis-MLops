# Diagnostic: list all files in trained_models/
import glob
all_pkl = glob.glob("trained_models/**/*.pkl", recursive=True)
print(f"Found .pkl files: {all_pkl}")