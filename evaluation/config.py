from dataclasses import dataclass
from typing import Dict, Any, List
import json


@dataclass
class EvaluationConfig:
    input_file_path: str

@dataclass
class GraphLoomConfig(EvaluationConfig):
    gl_enabled: bool
    gl_summ_enabled: bool
    
    @staticmethod
    def parse_config(config: Dict[str, Any]) -> 'GraphLoomConfig':
        try:
            gl_config = config["graphloom_config"]
            test_config = config["test_config"]
            return GraphLoomConfig(
                gl_enabled=gl_config.get("gl_enabled", False),
                gl_summ_enabled=gl_config.get("gl_summ_enabled", False),
                input_file_path=test_config.get("input_file_path", None)
            )
        except Exception as e:
            raise ValueError(f"Error parsing GraphLoomConfig: {e}")
        
    @staticmethod
    def parse_configs(configs: List[Dict[str, Any]]) -> List[EvaluationConfig]:
        return [GraphLoomConfig.parse_config(config) for config in configs]

    @staticmethod
    def parse_config_from_file(file_path: str) -> EvaluationConfig:
        with open(file_path, 'r') as f:
            config_list = json.load(f)
        return GraphLoomConfig.parse_configs(config_list)