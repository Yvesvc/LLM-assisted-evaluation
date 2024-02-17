import utils
from deepeval.models.base import DeepEvalBaseModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

class AzureOpenAI(DeepEvalBaseModel):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def _call(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"


llm_evaluator = AzureOpenAI(model=utils.llm)

metric = AnswerRelevancyMetric(model=llm_evaluator)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
)

metric.measure(test_case)
print(metric.score)
