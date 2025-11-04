RL Env Engineer Take Home - Dataset Processing
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/JNK234/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable or add it to .env file:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.

## Task Overview

This harness evaluates an RL-style dataset cleaning task aimed at ML engineers. The model is instructed to:

- inspect `messy_data.csv` and resolve all data quality problems
- remove placeholder tokens (`?`, `unknown`, `999`, `999999`, `-999`) and fill/drop missing values
- keep `age` within `[0, 120]`, ensure incomes are positive and not sentinel values, and keep contact counts between `0` and `100`
- deduplicate rows while preserving a binary `target` column with labels `0` and `1`
- create stratified folds labeled `0`–`4` whose positive-rate stays within ±15% of the overall rate

The prompt presented to the model calls out these concrete requirements, and the grader checks for the same conditions plus the absence of NaNs, duplicates, and invalid sentinel values. Any cleaning strategy that produces a CSV meeting these constraints will pass, keeping the task open-ended while still objectively verifiable.
