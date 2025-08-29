import json
import csv
from collections import Counter

# Load the JSON file
input_file = "agriculture_evaluation_output@lightrag_base_vs_graphloom_summary_off_rc_dne_eval_cleaned.json"
output_file = input_file.split(".")[0] + ".csv"
summary_file = input_file.split(".")[0] + "_summary.csv"

with open(input_file, "r") as f:
    data = json.load(f)

# Flatten the data into rows for the CSV and count summary stats
rows = []
summary_counts = {
    "Comprehensiveness": Counter(),
    "Diversity": Counter(),
    "Empowerment": Counter(),
    "Overall Winner": Counter()
}

for role, tasks in data.items():
    for task, questions in tasks.items():
        for question, content in questions.items():
            row = {
                "Role": role,
                "Task": task,
                "Question": question,
                "Answer 1(lightrag)": content.get("Answer 1(lightrag)", ""),
                "Answer 2(graphloom)": content.get("Answer 2(graphloom)", "")
            }
            print(f"Question: {question}")
            comparison = content.get("Comparison", {})
            for cat in ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]:
                print(cat)
                winner = comparison.get(cat, {}).get("Winner", "")
                explanation = comparison.get(cat, {}).get("Explanation", "")
                row[f"{cat}_Winner"] = winner
                row[f"{cat}_Explanation"] = explanation

                if winner == "Answer 1":
                    summary_counts[cat]["Lightrag"] += 1
                elif winner == "Answer 2":
                    summary_counts[cat]["Graphloom"] += 1
                else:
                    summary_counts[cat]["Tie"] += 1
            rows.append(row)

# Write detailed results CSV
fieldnames = [
    "Role", "Task", "Question", "Comprehensiveness_Winner", "Diversity_Winner", "Empowerment_Winner", "Overall Winner_Winner",
    "Answer 1(lightrag)", "Answer 2(graphloom)",
    "Comprehensiveness_Explanation", "Diversity_Explanation", "Empowerment_Explanation", "Overall Winner_Explanation"
]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Prepare summary rows
summary_rows = []
categories = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]
scenarios = ["Lightrag", "Graphloom", "Tie", "Total"]

for scenario in scenarios:
    row = {"Scenario": scenario}
    for cat in categories:
        if scenario == "Total":
            row[cat] = sum(summary_counts[cat].values())
        else:
            row[cat] = summary_counts[cat].get(scenario, 0)
    summary_rows.append(row)

# Write summary CSV
with open(summary_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Scenario"] + categories)
    writer.writeheader()
    writer.writerows(summary_rows)