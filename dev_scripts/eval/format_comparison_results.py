import json
import os

def parse_raw_string(raw_string: str):
    # Remove markdown formatting if present
    if raw_string.startswith("```json"):
        raw_string = raw_string[len("```json"):].strip()
    elif raw_string.startswith("```"):
        raw_string = raw_string[len("```"):].strip()
    if raw_string.endswith("```"):
        raw_string = raw_string[:-3].strip()

    try:
        parsed = json.loads(raw_string)
        if isinstance(parsed, str):  # sometimes double-encoded
            parsed = json.loads(parsed)
        return parsed
    except json.JSONDecodeError as e:
        print(f"[WARNING] Could not parse raw string:\n{raw_string[:200]}...\nError: {e}")
        return raw_string  # leave as is if not parsable


def convert_raw_fields(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "raw" and isinstance(v, str):
                new_obj["Comparison"] = parse_raw_string(v)
            else:
                new_obj[k] = convert_raw_fields(v)
        return new_obj
    elif isinstance(obj, list):
        return [convert_raw_fields(item) for item in obj]
    else:
        return obj


def main(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)

    updated = convert_raw_fields(data)

    with open(output_path, "w") as f:
        json.dump(updated, f, indent=2)

    print(f"[✓] Converted JSON saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    input_file = "agriculture_evaluation_output@lightrag_base_vs_graphloom_summary_off_rc_dne_eval.json"
    output_file = input_file.split(".")[0]+"_cleaned.json"
    main(input_file, output_file)
