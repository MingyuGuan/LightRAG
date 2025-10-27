from dataclasses_json import dataclass_json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import json
import os

def create_dir(config_name: str) -> Path:
    dir_name = config_name.replace(" ", "_").lower()
    path = Path(f"./out/{dir_name}")
    path.mkdir(parents=True, exist_ok=True)
    return path

@dataclass
class ModelProxy:
    api_key: str
    base_url: str

@dataclass_json
@dataclass
class ModelConfig:
    embedding_model: str = "text-embedding-3-small"
    inference_model: str = "gpt-4o-mini"
    evaluation_model: str = "gpt-4o-mini"

@dataclass
class EvaluationConfig:
    name: str
    input_file_path: str
    output_path: Path
    model_config: ModelConfig
    proxy: ModelProxy

@dataclass
class GraphLoomConfig(EvaluationConfig):
    gl_enabled: bool
    gl_summ_enabled: bool
    benchmark: str
    
    @staticmethod
    def parse_config(config: Dict[str, Any]) -> 'GraphLoomConfig':
        
        proxy = ModelProxy(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("LITELLM_PROXY_URL")
        )
        try:
            gl_config = config["graphloom_config"]
            test_config = config["test_config"]
            model_config = config["model_config"]
            return GraphLoomConfig(
                name=config.get("name"),
                gl_enabled=gl_config.get("gl_enabled", False),
                gl_summ_enabled=gl_config.get("gl_summ_enabled", False),
                input_file_path=test_config.get("input_file_path", None),
                benchmark=test_config.get("benchmark", None),
                output_path=create_dir(config.get("name")),
                model_config=ModelConfig.from_dict(model_config),
                proxy=proxy
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

    def add_path(self, identifier: str) -> Path:
        return self.output_path / identifier.replace(" ", "_").replace("-", "_").lower()