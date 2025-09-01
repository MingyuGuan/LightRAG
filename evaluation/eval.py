import asyncio
import sys
import argparse
import os
from evaluation.config import GraphLoomConfig
from evaluation.test import create_tests

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
    
    tests = create_tests(configs, limit=1)

    [test.parse() for test in tests]

    asyncio.run(asyncio.gather(*[test.evaluate() for test in tests]))

if __name__ == "__main__":
    sys.exit(main())