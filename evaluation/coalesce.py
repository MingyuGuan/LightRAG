

import os
import json
from collections import defaultdict

INPUT_DIR = "out"
OUTPUT_FILE = "final_result.json"

def is_score_dict(d):
    """Check if the dictionary looks like a score entry."""
    return all(
        isinstance(v, dict) and 
        {"comprehensiveness", "diversity", "empowerment"} <= v.keys()
        for v in d.values()
    )

def process_results_json(file_path):
    """Read results.json and return list of score dictionaries only."""
    with open(file_path, "r") as f:
        data = json.load(f)

    score_dicts = [d for d in data if isinstance(d, dict) and is_score_dict(d)]
    return score_dicts

def main():
    results_summary = {}

    for method in os.listdir(INPUT_DIR):
        method_path = os.path.join(INPUT_DIR, method)
        if not os.path.isdir(method_path):
            continue  # skip non-folder files

        total_scores = defaultdict(float)
        total_count = 0

        # iterate through numbered subfolders
        for subfolder in os.listdir(method_path):
            subfolder_path = os.path.join(method_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            results_file = os.path.join(subfolder_path, "result.json")
            if not os.path.isfile(results_file):
                continue

            score_dicts = process_results_json(results_file)

            for score_dict in score_dicts:
                for response, metrics in score_dict.items():
                    total_scores["comprehensiveness"] += metrics["comprehensiveness"]
                    total_scores["diversity"] += metrics["diversity"]
                    total_scores["empowerment"] += metrics["empowerment"]
                    total_count += 1

        if total_count > 0:
            results_summary[method] = {
                "comprehensiveness": total_scores["comprehensiveness"] / total_count,
                "diversity": total_scores["diversity"] / total_count,
                "empowerment": total_scores["empowerment"] / total_count,
            }
        else:
            results_summary[method] = {
                "comprehensiveness": None,
                "diversity": None,
                "empowerment": None,
            }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results_summary, f, indent=4)

if __name__ == "__main__":
    main()


# for each method average scores across all result.json files contained in its subdirectories

# plot bar graph with method names on x-axis and scores on y-axis