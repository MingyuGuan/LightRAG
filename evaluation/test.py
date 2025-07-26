
from evaluation.config import GraphLoomConfig
from evaluation.engine import RAGEngine
from typing import List
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path

@dataclass
class Query:
    question: str
    responses: List[str]

@dataclass
class TestCase:
    name: str
    document: str
    queries: List[Query]
    
@dataclass    
class Result:
    test_case: TestCase
    llm_output: str
    metrics: Dict[str, Any]

class Test:
    
    def __init__(self, config: GraphLoomConfig, documents, queries):
        self.config = config
        self.documents = documents
        self.queries = queries
        self.engine = RAGEngine(config)
        
    def name(self) -> str:
        pass
    
    def parse() -> List[TestCase]:
        pass
    
    def evaluate() -> List[Result]:
        pass
    
    def write_results(self) -> None:
        result_file = Path(self.config.working_dir) / f"results-{self.name()}.json"
        print(f"Writing results to {result_file}")
        ### TODO: 
        
class SQualityTest(Test):
    
    def name(self) -> str:
        return "SQuality"
    
    def parse(self) -> List[TestCase]:
        pass
    
    def evaluate(self) -> List[Result]:
        pass
    
    def write_results(self) -> None:
        pass