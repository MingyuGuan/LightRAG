
from evaluation.config import GraphLoomConfig
from evaluation.engine import RAGEngine

class Test:
    def __init__(self, config: GraphLoomConfig, documents, queries):
        self.config = config
        self.documents = documents
        self.queries = queries
        self.engine = RAGEngine(config)
        
    def run(self):
        pass