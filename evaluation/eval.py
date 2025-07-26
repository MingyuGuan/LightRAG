import sys
import argparse
import os
from evaluation.config import GraphLoomConfig
from evaluation.engine import RAGEngine
from evaluation.test import Test

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluation script for LightRAG')
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        required=True,
        help='Path to the configuration file'
    )
    args = parser.parse_args()
    
    # Validate that the config file exists
    if not os.path.isfile(args.config):
        parser.error(f"Config file not found: {args.config}")
    print(f"Using config file: {args.config}")
    
    return args

def main():
    args = parse_arguments()
    configs = GraphLoomConfig.parse_config_from_file(args.config)
    
    print(configs)
    
    tests = [Test(config, documents=[], queries=[]) for config in configs]
    
    [print(test.engine) for test in tests]
    ### TODO: invoke tests
    

if __name__ == "__main__":
    sys.exit(main())