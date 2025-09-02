import asyncio

from pluggy import Result
from tqdm.asyncio import tqdm as a_tqdm
from tqdm import tqdm
from evaluation.config import GraphLoomConfig
from evaluation.engine import RAGEngine
from typing import List, Type, override
from dataclasses import dataclass, asdict
from typing import Dict
from pathlib import Path
import json

from evaluation.llm import InstructorClient
from evaluation.metric import Metric, MetricRegistry
from evaluation.types import LLMOutput, Query
from lightrag.utils import logger

def serialize(metrics: List[Dict[str, Metric]]) -> List[str]:
    def serialize_inner(metric: Dict[str, Metric]) -> str:
        return {
            k: v.model_dump()
            for k, v in metric.items()
        }
    return [
        serialize_inner(metric) for metric in metrics
    ]
    
@dataclass
class TestCase:
    name: str
    document: str
    queries: List[Query]
    engine: RAGEngine

    def reset_engine(self, config: GraphLoomConfig):
        del self.engine
        self.engine = RAGEngine(config)
        logger.info("Reset RAG engine")

    def write_result(self, config: GraphLoomConfig, result: Result):
        file = config.add_path(self.name) / "result.json"
        with open(file, "w") as f:
            outputs = [asdict(output) for output in result.llm_output.values()]
            metrics = serialize(result.metrics)
            json.dump(outputs + metrics, f)

@dataclass
class Result:
    test_case: TestCase
    llm_output: Dict[str, LLMOutput]
    metrics: List[Dict[str, Metric]]

class Test:
    def __init__(
        self,
        config: GraphLoomConfig,
        limit=None,
        metrics: MetricRegistry = MetricRegistry(),
    ):
        self.config = config
        self.limit = limit
        self.results: List[Result] = []
        self.instructor_client = InstructorClient()
        self.metrics = metrics

    def name(self) -> str:
        pass

    def parse(self) -> List[TestCase]:
        pass

    async def evaluate(self) -> List[Result]:
        pass


class SQualityTest(Test):
    @staticmethod
    def name() -> str:
        return "squality"

    @override
    def parse(self) -> List[TestCase]:
        test_file = Path(self.config.input_file_path)
        logger.info(f"Test file: {test_file}")

        with open(test_file, "r") as f:
            lines = f.readlines()
            test_cases = [
                json.JSONDecoder().decode(line) for line in lines if line.strip()
            ]

        _limit = self.limit if self.limit is not None else len(test_cases)
        self.tests = [
            TestCase(
                name=test["metadata"]["passage_id"],
                document=test["document"],
                engine=RAGEngine(
                    self.config, identifier=test["metadata"]["passage_id"]
                ),
                queries=[
                    Query(
                        question=question["question_text"],
                        target_responses=[
                            response["response_text"]
                            for response in question["responses"]
                        ],
                    )
                    for question in test["questions"]
                ],
            )
            for test in tqdm(test_cases[:_limit])
        ]

        return self.tests

    @override
    async def evaluate(self) -> List[Result]:
        logger.info(f"Evaluating {len(self.tests)} test cases")
        evaluate_tasks = [
            asyncio.create_task(self.a_evaluate(test_case)) for test_case in self.tests
        ]
        results = await a_tqdm.gather(*evaluate_tasks)

        logger.info(f"Evaluated test cases: {len(self.tests)}")
        self.results = results
        return results

    async def a_evaluate(self, test_case: TestCase) -> Result:
        logger.info(f"Evaluating test case: {test_case.name}")
        await test_case.engine.insert(
            test_case.document
        )  # <-- insert all documents for this test case into the index
        logger.info("Inserted document for test case {test_case.name} into index")

        responses = []
        # synchronously query engine to emulate conversation
        for query in test_case.queries:
            response = await test_case.engine.query(query.question)
            responses.append(response)

        # test_case.reset_engine(self.config)

        query_response_pairs = zip(test_case.queries, responses)
        result_dict = {
            pair[0].question: LLMOutput(
                question=pair[0].question,
                target_responses=pair[0].target_responses,
                llm=pair[1],
            )
            for pair in query_response_pairs
        }

        metrics = await asyncio.gather(
            *[
                metric.evaluate_metric(llm_output, self.instructor_client)
                for metric in self.metrics.get_metrics().values()
                for llm_output in result_dict.values()
            ]
        )

        result = Result(test_case=test_case, llm_output=result_dict, metrics=metrics)
        test_case.write_result(self.config, result)
        return result


REGISTRY = {
    test.name(): test
    for test in [SQualityTest]  # add more tests here
}


def get_test(name: str) -> Type[Test]:
    if name in REGISTRY:
        return REGISTRY[name]
    raise ValueError(f"Test not found: {name}")


def create_tests(
    configs: List[GraphLoomConfig],
    metrics: MetricRegistry = MetricRegistry(),
    limit: int = None,
) -> List[Test]:
    return [
        get_test(config.benchmark)(config, limit=limit, metrics=metrics)
        for config in configs
    ]
