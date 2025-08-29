"""
python models_comparator.py --json1=agriculture_evaluation_output@lightrag_base.json --json2=agriculture_evaluation_output@graphloom_summary_off_rc_exists.json
python models_comparator.py --json1=agriculture_evaluation_output@lightrag_base.json --json2=agriculture_evaluation_output@graphloom_summary_on_rc_exists.json
python models_comparator.py --json1=agriculture_evaluation_output@lightrag_base.json --json2=agriculture_evaluation_output@graphloom_summary_on_rc_dne.json
python models_comparator.py --json1=agriculture_evaluation_output@lightrag_base.json --json2=agriculture_evaluation_output@graphloom_summary_off_rc_dne.json

python models_comparator.py --json1=dickens_evaluation_output@lightrag_base.json --json2=dickens_evaluation_output@graphloom_summary_off_rc_off.json
python models_comparator.py --json1=dickens_evaluation_output@lightrag_base.json --json2=dickens_evaluation_output@graphloom_summary_on_rc_off.json
python models_comparator.py --json1=dickens_evaluation_output@lightrag_base.json --json2=dickens_evaluation_output@graphloom_summary_on_rc_dne.json
python models_comparator.py --json1=dickens_evaluation_output@lightrag_base.json --json2=dickens_evaluation_output@graphloom_summary_off_rc_dne.json
"""

import json
import argparse
from openai import OpenAI
from argparse import Namespace

EVAL_PROMPT_TEMPLATE = """
---Role---
You are an expert tasked with evaluating two answers to the same question based on four criteria: Comprehensiveness, Diversity, and Empowerment.

---Goal---
You will evaluate two answers to the same question based on four criteria:
- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

---Evaluation Input---
Here is the question: {query}
Here are the two answers:
Answer 1: {answer1}
Answer 2: {answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.
Output your evaluation in the following JSON format:
{{
  "Comprehensiveness": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
  "Diversity": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
  "Empowerment": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
  "Overall Winner": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]" }}
}}
"""

def evaluate_pairwise(json1, json2, model="gpt-4o", base_url=None, api_key=None):
    base_name = json1.split("@")[0]
    assert base_name == json2.split("@")[0], "Base names must match"
    mod1 = ((json1.split("@")[1]).split(".")[0])
    mod2 = ((json2.split("@")[1]).split(".")[0])
    output_path = f"{base_name}@{mod1}_vs_{mod2}_eval.json"
    cat1 = mod1.split("_")[0]
    cat2 = mod2.split("_")[0]
    client = OpenAI(api_key=api_key, base_url=base_url)

    with open(json1, "r") as f1, open(json2, "r") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    results = {}

    cnt =1 

    for role in data1:
        results[role] = {}
        for task in data1[role]:
            results[role][task] = {}
            for question in data1[role][task]:
                print(f"cnt={cnt}")
                cnt += 1
                q = question
                a1 = data1[role][task][q]
                a2 = data2[role][task][q]
                prompt = EVAL_PROMPT_TEMPLATE.format(query=q, answer1=a1, answer2=a2)

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                content = response.choices[0].message.content
                results[role][task][q] = {}
                results[role][task][q]["raw"] = content
                results[role][task][q][f"Answer({cat1})"] = a1
                results[role][task][q][f"Answer({cat2})"] = a2
                

    print(f"Saving results to {output_path}")
    with open(output_path, "w") as fout:
        json.dump(results, fout, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json1", type=str, required=True, help="Path to first JSON file")
    parser.add_argument("--json2", type=str, required=True, help="Path to second JSON file")
    
    args = parser.parse_args()

    evaluate_pairwise(
        json1=args.json1,
        json2=args.json2,
        model="meta-llama/Meta-Llama-3.1-8B-Instruct", 
        base_url="http://0.0.0.0:8004/v1",
        api_key="not-needed", 
    )