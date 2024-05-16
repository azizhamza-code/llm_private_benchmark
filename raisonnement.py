from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, chain_of_thought, self_critique

dataset = json_dataset(
    "./data.json"
)

@task
def test_de_raisonnement():
    return Task(
        dataset=dataset,
        plan=[generate(), chain_of_thought(), self_critique()],
        scorer=model_graded_fact(),
    )