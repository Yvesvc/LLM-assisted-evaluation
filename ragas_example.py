import utils
import pandas as pd
import datasets
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate

data = {
    "question": [ "What are the global implications of the USA Supreme Court ruling on abortion?"],
    "contexts": [
        ["- In 2022, the USA Supreme Court handed down a decision ruling that overturned 50 years of jurisprudence recognizing a constitutional right to abortion.\n- This decision has had a massive impact: one in three women and girls of reproductive age now live in states where abortion access is either totally or near-totally inaccessible.\n- The states with the most restrictive abortion laws have the weakest maternal health support, higher maternal death rates, and higher child poverty rates.\n- The USA Supreme Court ruling has also had impacts beyond national borders due to the geopolitical and cultural influence wielded by the USA globally and the aid it funds.\n- SRR organizations and activists across the world have expressed fear about the ruling laying the groundwork for anti-abortion legislative and policy attacks in other countries.\n- Advocates have also observed the ruling's impact on progressive law reform and the stalling of the adoption and enforcement of abortion guidelines in certain African countries.\n- The ruling has created a chilling effect in international policy spaces, emboldening anti-abortion state and non-state actors to undermine human rights protections."]
    ],
    "ground_truths": [
        ["The global implications of the USA Supreme Court ruling on abortion are significant. The ruling has led to limited or no access to abortion for one in three women and girls of reproductive age in states where abortion access is restricted. These states also have weaker maternal health support, higher maternal death rates, and higher child poverty rates. Additionally, the ruling has had an impact beyond national borders due to the USA's geopolitical and cultural influence globally. Organizations and activists worldwide are concerned that the ruling may inspire anti-abortion legislative and policy attacks in other countries. The ruling has also hindered progressive law reform and the implementation of abortion guidelines in certain African countries. Furthermore, the ruling has created a chilling effect in international policy spaces, empowering anti-abortion actors to undermine human rights protections."],
    ],
    "answer": ["The global implications of the USA Supreme Court ruling on abortion can be significant, as it sets a precedent for other countries and influences the global discourse on reproductive rights. Here are some potential implications:\n\n1. Influence on other countries: The Supreme Court's ruling can serve as a reference point for other countries grappling with their own abortion laws. It can provide legal arguments and reasoning that advocates for reproductive rights can use to challenge restrictive abortion laws in their respective jurisdictions.\n\n2. Strengthening of global reproductive rights movements: A favorable ruling by the Supreme Court can energize and empower reproductive rights movements worldwide. It can serve as a rallying point for activists and organizations advocating for women's rights, leading to increased mobilization and advocacy efforts globally.\n\n3. Counteracting anti-abortion movements: Conversely, a ruling that restricts abortion rights can embolden anti-abortion movements globally. It can provide legitimacy to their arguments and encourage similar restrictive measures in other countries, potentially leading to a rollback of existing reproductive rights.\n\n4. Impact on international aid and policies: The Supreme Court's ruling can influence international aid and policies related to reproductive health. It can shape the priorities and funding decisions of donor countries and organizations, potentially leading to increased support for reproductive rights initiatives or conversely, restrictions on funding for abortion-related services.\n\n5. Shaping international human rights standards: The ruling can contribute to the development of international human rights standards regarding reproductive rights. It can influence the interpretation and application of existing human rights treaties and conventions, potentially strengthening the recognition of reproductive rights as fundamental human rights globally.\n\n6. Global health implications: The Supreme Court's ruling can have implications for global health outcomes, particularly in countries with restrictive abortion laws. It can impact the availability and accessibility of safe and legal abortion services, potentially leading to an increase in unsafe abortions and related health complications.\n\nIt is important to note that the specific implications will depend on the nature of the Supreme Court ruling and the subsequent actions taken by governments, activists, and organizations both within and outside the United States.",]
}

dataset = datasets.Dataset.from_dict(pd.DataFrame(data=data))

result = evaluate(
    dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ], 
    llm=utils.llm, 
    embeddings=utils.embeddings
)

df = result.to_pandas()
print(df)
