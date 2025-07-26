from evaluation.config import GraphLoomConfig
from evaluation.engine import RAGEngine
from typing import List
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
import json

from lightrag.utils import logger


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

    def parse(self) -> List[TestCase]:
        pass

    def evaluate(self) -> List[Result]:
        pass

    def write_results(self) -> None:
        result_file = Path(self.config.working_dir) / f"results-{self.name()}.json"
        print(f"Writing results to {result_file}")
        ### TODO:


class SQualityTest(Test):
    def name(self) -> str:
        return "SQuality"

    def parse(self) -> List[TestCase]:
        test_file = Path(self.config.input_file_path)
        logger.info(f"Test file: {test_file}")

        with open(test_file, "r") as f:
            lines = f.readlines()
            test_cases = [
                json.JSONDecoder().decode(line) for line in lines if line.strip()
            ]

        self.tests = [
            TestCase(
                name=test["metadata"]["passage_id"],
                document=test["document"],
                queries=[
                    Query(
                        question=question["question_text"],
                        responses=[
                            response["response_text"]
                            for response in question["responses"]
                        ],
                    )
                    for question in test["questions"]
                ],
            )
            for test in test_cases
        ]

        return self.tests

    def evaluate(self) -> List[Result]:
        results = []
        for test_case in self.tests:
            logger.info(f"Evaluating test case: {test_case.name}")
            self.engine.engine.insert(test_case.document)
            ### TODO:
            # 1. Query RAG
            # 2. Compute averaged metric against 4 sample answers
            # 3. Add to results
        return results
