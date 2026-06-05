import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

from pydfc.comparison import SimilarityAssessment   # pip install pydfc

import pickle

# FULL PATH usually looks like:
# "{path_to_datasets}/{dataset_id}/derivatives/dFC_assessed/{subject_id}/{session_id}/*.npy"
# where * has the format "dFC_{identifier}_{method_number}"
# where identifier has the format "{session_id}_{task_id}_{run_id}"
# However, session_id and run_id could be absent, and their keys would be set to None in the output dictionary!

if len(sys.argv) < 2:
    print("Missing a path to the datasets directory")
    print("Usage: sbatch run_dfc.sh <path_to_datasets>")
    sys.exit(1)
    
path_to_datasets = sys.argv[1]

root = Path(path_to_datasets)


# Create a dictionary to store similarity assessment results
# of the form: similarity[dataset_id][subject_id][session_id][run_id][task_id] = {"matrix": matrix, "methods": method_numbers}
# where matrix.shape = (1, num_methods, num_methods) and contains the similarity values between methods

similarity = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(dict)
        )
    )
)

for dataset_dir in root.iterdir():

    if not dataset_dir.is_dir():
        continue

    dataset_id = dataset_dir.name

    dfc_dir = dataset_dir / "derivatives" / "dFC_assessed"
    
    if not dfc_dir.is_dir():
        print(f"Skipping {dataset_id} since /derivatives/dFC_assessed not found")
        continue

    for subject_dir in dfc_dir.iterdir():

        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name

        # If no session folders, treat the subject directory as the session directory
        # to avoid file path issues. If this case, session_id will be set to None later.
        session_dirs = [
            p for p in subject_dir.iterdir()
            if p.is_dir() and p.name.startswith("ses-")
        ]

        if not session_dirs:
            session_dirs = [subject_dir]
            

        for session_dir in session_dirs:
            
            # Group files by identifier
            files_by_identifier = defaultdict(list)

            for npy_file in session_dir.glob("dFC_*.npy"):

                filename = npy_file.stem    # removed .npy

                _, rest = filename.split("_", 1)  # e.g., "dFC", "ses-wave1bas_task-Stroop_run-2_24"
                identifier, method_number = rest.rsplit("_", 1)  # e.g., "ses-wave1bas_task-Stroop_run-2", "24"

                files_by_identifier[identifier].append(
                    (int(method_number), npy_file)
                )


            # Process one identifier at a time (similarity across methods)
            for identifier, file_info in files_by_identifier.items():
                
                # Initialize session_id and run_id as None in case they don't exist
                session_id = None
                run_id = None
                task_id = None  # must exist, see check later to catch error.
                
                # Get session, task, and run from identifier (if they exist)
                for part in identifier.split("_"):
                    if part.startswith("ses-"):     # e.g., "ses-wave1bas"
                        session_id = part

                    elif part.startswith("run-"):   # e.g., "run-2"
                        run_id = part

                    elif part.startswith("task-"):  # e.g., "task-Stroop"
                        task_id = part
                        
                    else:
                        print(f"Warning: Unrecognized part '{part}' in identifier '{identifier}' \
                            of subject '{subject_id}' in dataset '{dataset_id}'. Ignoring this part.")
                
                if task_id is None:
                    print(f"Error: task_id not found in identifier '{identifier}' of subject '{subject_id}' \
                        in dataset '{dataset_id}'. Skipping this file.")
                    continue

                # Sort methods numerically
                file_info.sort(key=lambda x: x[0])

                method_numbers = []
                
                # This is a list of the dFC objects from various methods 
                # that share the same identifier i.e., they came from the same 
                # BOLD time series, but they were computed using different methods
                # Each dFC in the list is recognized as a dFC object by pydfc
                dFC_lst = []

                for method_num, path in file_info:
                    method_numbers.append(method_num)
                    dFC_lst.append(
                        np.load(path, allow_pickle=True).item()
                    )
                
                # Note: type(output) = dict with 
                # dict_keys(['measure_lst', 'TS_info_lst', 'common_TRs', 'time_record_dict', 'all'])
                similarity_assessment = SimilarityAssessment(dFC_lst=dFC_lst)
                output = similarity_assessment.assess_similarity_fast(dFC_lst=dFC_lst)
                
                
                similarity[dataset_id][subject_id][session_id][run_id][task_id] = {
                    "matrix": output,
                    "methods": method_numbers,
                }
                
        print(f"Finished processing subject {subject_id} in dataset {dataset_id}")
                

output_dir = root / "similarity_assessments"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "similarity.pkl"

# Convert to normal dict for pickling. Need to do recursively because of the nested defaultdicts.
def to_dict(d):
    if isinstance(d, defaultdict):
        return {k: to_dict(v) for k, v in d.items()}
    return d

similarity = to_dict(similarity)

with open(output_file, "wb") as f:
    pickle.dump(similarity, f)
    
    
print(f"Saved results to: {output_file}")