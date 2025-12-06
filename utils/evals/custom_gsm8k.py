# Copied from https://github.com/huggingface/lighteval/blob/99ef5b98d422cf3620eebec9db13285493d35542/src/lighteval/tasks/tasks/gsm8k.py
# Increases generation size to 768 from 256 to better accommodate longer solutions by dsr1.
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks.gsm8k import gsm8k_prompt

gsm8k_long = LightevalTaskConfig(
    name="gsm8k_long",
    prompt_function=gsm8k_prompt,
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1024,         # raised this from 256
    metrics=[Metrics.expr_gold_metric],
    stop_sequence=None,           # avoid early stop on "Question:"
    version=0,
)

TASKS_TABLE = [gsm8k_long]
