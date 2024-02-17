import utils
import numpy as np
from trulens_eval.app import App
from trulens_eval.feedback import Groundedness
from trulens_eval import TruChain, Feedback, Tru
from trulens_eval.feedback.provider.openai import AzureOpenAI
import os

tru = Tru()
tru.reset_database()

provider = AzureOpenAI(deployment_name=os.getenv('AZURE_GPT_DEPLOYMENT_NAME'))

context = App.select_context(utils.rag_chain)

grounded = Groundedness(groundedness_provider=provider)
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(context.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

f_qa_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = (
    Feedback(provider.qs_relevance)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

tru_recorder = TruChain(utils.rag_chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_qa_relevance, f_context_relevance, f_groundedness])


with tru_recorder as recording:
    llm_response = utils.rag_chain.invoke("Where did harrison work?")


tru.get_leaderboard(app_ids=["RAG v1"])
tru.run_dashboard()