"""Optimize the cap table formatting program using DSPy's MIPROv2 optimizer."""

import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate

from .program import CapTableProgram, CapTableSignature
from .data import trainset, devset


def exact_match_metric(example: dspy.Example, pred: CapTableSignature, trace=None) -> int:
    """Simple metric that checks whether the predicted cap_table matches the gold one."""
    return int(pred.cap_table == example.cap_table)


if __name__ == "__main__":
    # Configure your language model before running (e.g., OpenAI key via env vars).
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    teleprompter = MIPROv2(metric=exact_match_metric, auto="light")
    optimized_program = teleprompter.compile(
        CapTableProgram(),
        trainset=trainset,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
        requires_permission_to_run=False,
    )

    optimized_program.save("cap_table_program_optimized.json")

    evaluator = Evaluate(devset=devset, metric=exact_match_metric)
    evaluator(optimized_program)
