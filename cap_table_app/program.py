from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import dspy

# ---------------------------------------------------------------------------
# Typed data structures used in the cap table output
# ---------------------------------------------------------------------------

@dataclass
class Founder:
    full_name: str
    ownership_percentage: Optional[float]

@dataclass
class OwnershipGroup:
    group_name: str
    ownership_percentage: float

@dataclass
class CapTable:
    founders: List[Founder]
    ownership_groups: List[OwnershipGroup]


# ---------------------------------------------------------------------------
# DSPy signature describing the task
# ---------------------------------------------------------------------------

class CapTableSignature(dspy.Signature):
    """Convert a detailed text about a startup's ownership into a structured cap table."""

    cap_table_answer = dspy.InputField(desc="Detailed description of the startup's ownership structure")
    founders_list = dspy.InputField(desc="Comma-separated list of founder names")
    cap_table = dspy.OutputField(type_=CapTable, desc="Formatted cap table information")


# ---------------------------------------------------------------------------
# Instruction text for the LM
# ---------------------------------------------------------------------------

INSTRUCTIONS = """
You are a specialized formatter for startup information. Given a founders list and a detailed text about the ownership structure, extract the founders and other ownership groups with their ownership percentages. Always return valid JSON matching the CapTable dataclass.

Follow these rules:
- Include an entry for each founder in the provided list. If no percentage is mentioned, set it to null. A founder can have 0 if explicitly stated.
- Group non-founder entities under standard names, e.g. use "ESOP" for employee option pools or "ESOP & Advisors" when advisors share the same pool.
- Only include other ownership groups when a percentage is mentioned.
- Ignore unconverted or pending investments.
- Adjust founder percentages proportionally if the text makes it clear that percentages should sum to accommodate an option pool or other group.

Output only JSON formatted according to the CapTable dataclass.
"""


# ---------------------------------------------------------------------------
# DSPy module implementing the formatting logic
# ---------------------------------------------------------------------------

class CapTableProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.formatter = (
            dspy.ChainOfThought(CapTableSignature)
            .with_instructions(INSTRUCTIONS)
            .with_adapters(dspy.JSONAdapter())
        )

    def forward(self, cap_table_answer: str, founders_list: str) -> CapTableSignature:
        return self.formatter(cap_table_answer=cap_table_answer, founders_list=founders_list)


def format_cap_table_info(cap_table_answer: str, founders_list: str) -> CapTable:
    """Convenience wrapper that runs the CapTableProgram."""
    program = CapTableProgram()
    prediction = program(cap_table_answer=cap_table_answer, founders_list=founders_list)
    return prediction.cap_table
