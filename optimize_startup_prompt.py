import json
import dspy
from dspy.teleprompt import MIPROv2


def load_jsonl(path):
    examples = []
    system = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if system is None:
                system = next(m["content"] for m in obj["messages"] if m["role"] == "system")
            desc = next(m["content"] for m in obj["messages"] if m["role"] == "user")
            label_msg = next(m["content"] for m in obj["messages"] if m["role"] == "assistant")
            label = label_msg.replace("Label:", "").strip()
            examples.append(dspy.Example(description=desc, label=label).with_inputs("description"))
    return examples, system


trainset, system_prompt = load_jsonl("startup_classifier/finetune_data_train.jsonl")
valset, _ = load_jsonl("startup_classifier/finetune_data_validation.jsonl")


class StartupSignature(dspy.Signature):
    """{}""".format(system_prompt)

    description = dspy.InputField(desc="Company description")
    label = dspy.OutputField(desc="Yes or No")


# Configure the LM (replace the API key with your own)
lm = dspy.LM("openai/gpt-4o-mini", api_key="YOUR_OPENAI_API_KEY")
dspy.configure(lm=lm)

# Create the program
dspy_program = dspy.ChainOfThought(StartupSignature)


def accuracy_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    return example.label.lower() == str(pred.label).lower()


teleprompter = MIPROv2(metric=accuracy_metric, auto="medium")
optimized_program = teleprompter.compile(
    dspy_program,
    trainset=trainset,
    valset=valset,
    requires_permission_to_run=False,
)

optimized_program.save("optimized_startup_classifier.json")
