import json
import argparse

def analyze_responses(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_queries = len(data)
        no_context_responses = sum(1 for item in data if item['result'].strip() == "Sorry, I'm not able to provide an answer to that question.[no-context]")
        
        percentage = (no_context_responses / total_queries) * 100
        
        print(f"\nAnalysis Results:")
        print(f"Total number of queries: {total_queries}")
        print(f"Number of 'no-context' responses: {no_context_responses}")
        print(f"Percentage of 'no-context' responses: {percentage:.2f}%")
        
        # Print queries with substantive responses
        print("\nQueries with substantive responses:")
        for item in data:
            if item['result'].strip() != "Sorry, I'm not able to provide an answer to that question.[no-context]":
                print(f"- {item['query']}")
                
        # Print queries with no-context responses
        print("\nQueries with no-context responses:")
        for item in data:
            if item['result'].strip() == "Sorry, I'm not able to provide an answer to that question.[no-context]":
                print(f"- {item['query']}")
                
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not valid JSON")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze query responses in a JSON file.')
    parser.add_argument('file_path', type=str, help='Path to the JSON file containing query responses')
    args = parser.parse_args()
    
    analyze_responses(args.file_path) 