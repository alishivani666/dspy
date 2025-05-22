# Cap Table Formatter with DSPy

This folder contains a small example of using [DSPy](https://dspy.ai) to convert a descriptive cap table into structured data.  
The program defines typed dataclasses for founders and ownership groups and optimizes the underlying prompt using DSPy's `MIPROv2` optimizer.

## Files

- `program.py` – DSPy module that implements the formatting logic.
- `data.py` – Example training/dev data derived from the instructions.
- `optimize.py` – Script that runs the optimizer and evaluates the resulting program.

Run `python optimize.py` after configuring your language model (e.g., setting `OPENAI_API_KEY`).  The optimized program is saved as `cap_table_program_optimized.json`.
