import ast
import asyncio
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlretrieve
import os
import numpy as np
import pandas as pd
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

api_key = ""

BASE_DIR = Path(__file__).parent
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional/bank-additional-full.csv"
RAW_DATA_PATH = BASE_DIR / "messy_data.csv"
TOOL_STATE: dict[str, Any] | None = None

def ensure_dataset() -> Path:
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    try:
        urlretrieve(DATASET_URL, RAW_DATA_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to download dataset: {exc}")
    return RAW_DATA_PATH


def create_python_tool() -> Callable[[str], dict[str, Any]]:
    global TOOL_STATE
    
    # Import sklearn for the model
    try:
        from sklearn.model_selection import StratifiedKFold
        sklearn_available = StratifiedKFold
    except ImportError:
        sklearn_available = None
    
    state: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "Path": Path,
        "RAW_DATA_PATH": str(RAW_DATA_PATH),
        "DATASET_URL": DATASET_URL,
        "CSV_SEP": ",",
        "BASE_DIR": str(BASE_DIR),
        "StratifiedKFold": sklearn_available,
    }
    TOOL_STATE = state

    def python_expression_tool(expression: str) -> dict[str, Any]:
        stdout = StringIO()
        try:
            with redirect_stdout(stdout):
                exec(expression, state, state)
            return {"stdout": stdout.getvalue(), "error": None}
        except Exception as exc:
            return {"stdout": stdout.getvalue(), "error": repr(exc)}

    return python_expression_tool


def submit_answer_tool(answer: Any) -> dict[str, Any]:
    coerced = coerce_submission(answer)
    return {"answer": coerced, "submitted": True}


def download_dataset_tool() -> dict[str, Any]:
    try:
        path = ensure_dataset()
        size = path.stat().st_size if path.exists() else 0
        return {"path": str(path), "size_bytes": int(size)}
    except Exception as exc:
        return {"error": repr(exc)}


def coerce_submission(payload: Any) -> Any:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                candidate = parser(payload)
            except Exception:
                continue
            if isinstance(candidate, dict):
                return candidate
        if TOOL_STATE is not None:
            try:
                candidate = eval(payload, {"__builtins__": {}}, TOOL_STATE)
                if isinstance(candidate, dict):
                    return candidate
            except Exception:
                pass
    return payload


def grade_submission(result: Any) -> bool:
    result = coerce_submission(result)

    if not isinstance(result, dict):
        print("FAIL: submission must be a dict")
        return False

    cleaned_path = result.get("cleaned_csv")
    if not isinstance(cleaned_path, str):
        print("FAIL: cleaned_csv must be a string path")
        return False

    cleaned_file = (BASE_DIR / cleaned_path).resolve()
    if not cleaned_file.exists():
        print("FAIL: cleaned CSV file does not exist")
        return False

    # Load cleaned data
    try:
        df = pd.read_csv(cleaned_file)
    except Exception as e:
        print(f"FAIL: could not load cleaned CSV: {e}")
        return False

    # Check has reasonable amount of data (at least some data for ML)
    if len(df) < 30:
        print(f"FAIL: cleaned data has only {len(df)} rows, too few for meaningful cross-validation")
        return False

    # Check 'fold' column exists
    if 'fold' not in df.columns:
        print("FAIL: cleaned data must have a 'fold' column")
        return False

    # Check 'target' column exists
    if 'target' not in df.columns:
        print("FAIL: cleaned data must have a 'target' column")
        return False

    # Validate no missing values
    if df.isnull().any().any():
        print("FAIL: cleaned data contains missing values (NaN)")
        return False

    # Validate target is binary
    unique_targets = sorted(df['target'].unique())
    if unique_targets != [0, 1]:
        print(f"FAIL: target must be binary 0 and 1, found: {unique_targets}")
        return False

    # Validate fold column has exactly 5 folds (0-4)
    unique_folds = sorted(df['fold'].unique())
    if unique_folds != [0, 1, 2, 3, 4]:
        print(f"FAIL: fold column must have values 0,1,2,3,4, found: {unique_folds}")
        return False

    # Check for common sentinel/placeholder values in numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ['target', 'fold']:
            continue
        # Check for obvious sentinel values
        if (df[col] == 999).any() or (df[col] == 999999).any() or (df[col] == -999).any():
            print(f"FAIL: column '{col}' contains sentinel values (999, 999999, or -999)")
            return False

    # Check for placeholder values in string columns
    for col in df.select_dtypes(include=['object']).columns:
        if (df[col] == '?').any():
            print(f"FAIL: column '{col}' contains '?' placeholder values")
            return False
        if (df[col].str.lower() == 'unknown').any():
            print(f"FAIL: column '{col}' contains 'unknown' placeholder values")
            return False

    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ['target', 'fold']:
            continue
            
        if 'age' in col.lower():
            if (df[col] < 0).any() or (df[col] > 120).any():
                print(f"FAIL: {col} contains unrealistic values (found values < 0 or > 120)")
                return False
        
        elif 'income' in col.lower() or 'salary' in col.lower():
            if (df[col] <= 0).any():
                print(f"FAIL: {col} contains invalid values (must be positive)")
                return False
        
        elif 'day' in col.lower() or 'time' in col.lower():
            if (df[col] < 0).any():
                print(f"FAIL: {col} contains negative values (time cannot be negative)")
                return False
            if (df[col] > 1000).any():
                print(f"FAIL: {col} contains unrealistic values (> 1000 days)")
                return False
        
        elif 'contact' in col.lower() or 'previous' in col.lower() or 'campaign' in col.lower():
            if (df[col] < 0).any():
                print(f"FAIL: {col} contains negative values (counts cannot be negative)")
                return False
            if (df[col] > 100).any():
                print(f"FAIL: {col} contains unrealistic values (> 100 contacts)")
                return False
        
        else:
            # Check for obviously invalid negative values in columns that should be positive
            if (df[col] < -100).any():
                print(f"FAIL: {col} contains extreme negative values")
                return False
            # Check for suspiciously large values (likely data errors)
            if (df[col] > 1000000).any():
                print(f"FAIL: {col} contains extreme outlier values (> 1,000,000)")
                return False

    # Check stratification: each fold should have similar target distribution
    overall_pos_ratio = (df['target'] == 1).mean()
    
    for fold_num in range(5):
        fold_data = df[df['fold'] == fold_num]
        
        if len(fold_data) == 0:
            print(f"FAIL: fold {fold_num} is empty")
            return False
        
        fold_pos_ratio = (fold_data['target'] == 1).mean()
        
        # Check stratification (allow 15% deviation)
        if abs(fold_pos_ratio - overall_pos_ratio) > 0.15:
            print(f"FAIL: fold {fold_num} is not stratified (ratio {fold_pos_ratio:.3f} vs overall {overall_pos_ratio:.3f})")
            return False

    # Check no duplicate rows (excluding fold column)
    df_without_fold = df.drop(columns=['fold'])
    if df_without_fold.duplicated().any():
        print("FAIL: cleaned data contains duplicate rows")
        return False

    return True


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    handlers: dict[str, Callable[[Any], dict[str, Any]]],
    max_steps: int = 20,
    verbose: bool = False,
) -> Any | None:
    # api_key = os.getenv("ANTHROPIC_API_KEY")
    # if not api_key:
    #     raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = AsyncAnthropic(api_key=api_key)
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=2000,
            tools=tools,
            messages=messages,
        )

        tool_results = []
        submitted = None

        for content in response.content:
            if content.type == "text" and verbose:
                print(content.text)
            elif content.type == "tool_use":
                tool_name = content.name
                tool_input = content.input if isinstance(content.input, dict) else {}
                handler = handlers.get(tool_name)
                if not handler:
                    continue

                if tool_name == "python_expression":
                    expr = tool_input.get("expression", "")
                    if verbose:
                        print(f"\n[{tool_name} input]\n{expr}")
                    result = handler(expr)
                elif tool_name == "submit_answer":
                    answer = tool_input.get("answer")
                    result = handler(answer)
                    submitted = result["answer"]
                else:
                    result = handler(**tool_input)

                if verbose:
                    print(f"[{tool_name} output]\n{result}")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": json.dumps(result, default=lambda x: int(x) if hasattr(x, "__int__") else str(x)),
                    }
                )

        if tool_results:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        if submitted is not None:
            return submitted

    return None


async def run_single_test(run_id: int, prompt: str, tools: list, handlers: dict, verbose: bool = False):
    # Create unique prompt per run to avoid file conflicts
    unique_prompt = prompt.replace(
        f"{RAW_DATA_PATH.stem}_cleaned.csv",
        f"{RAW_DATA_PATH.stem}_cleaned_run{run_id}.csv"
    )
    
    result = await run_agent_loop(unique_prompt, tools, handlers, verbose=verbose)

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            pass

    success = grade_submission(result)
    status = "✓ SUCCESS" if success else "✗ FAILURE"
    print(f"Run {run_id:2d}: {status}")
    
    # Cleanup: remove the unique file after grading
    if result and isinstance(result, dict):
        cleaned_path = result.get("cleaned_csv")
        if cleaned_path:
            try:
                (BASE_DIR / cleaned_path).unlink(missing_ok=True)
            except Exception:
                pass
    
    return run_id, success


async def main(concurrent: bool = True):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY in your environment before running the harness.")
        return

    ensure_dataset()

    tools: list[ToolUnionParam] = [
        {
            "name": "download_dataset",
            "description": "Check if dataset exists at RAW_DATA_PATH and return path info",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "python_expression",
            "description": "Execute Python code. State persists. RAW_DATA_PATH points to the raw CSV.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python code to exec(). Use prints for feedback; outputs persist in the session namespace.",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit your final cleaned CSV filename",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "description": "Dict with 'cleaned_csv' key containing the filename",
                        "type": "object"
                    }
                },
                "required": ["answer"],
            },
        },
    ]

    handlers = {
        "download_dataset": lambda: download_dataset_tool(),
        "python_expression": create_python_tool(),
        "submit_answer": submit_answer_tool,
    }

    prompt = f"""
You are a machine learning engineer preparing a dataset for production model training. Your goal is to perform thorough data cleaning, outlier detection, and set up proper stratified cross-validation to ensure reliable model evaluation.

DATASET: {RAW_DATA_PATH}
A dataset with demographic and behavioral features, plus a binary target variable for classification.

1. EXPLORATORY DATA ANALYSIS (EDA)
Start by thoroughly understanding the dataset structure and identifying all data quality issues:

a) Load and understand structure

b) Systematically check for issues in each column:
   - Missing values
   - Sentinel/placeholder values
   - Duplicates
   - Outliers: Examine numeric columns for unrealistic values (e.g., impossible ages, negative values where they shouldn't be)
   - Target column


2. DATA CLEANING
Fix the issues you identified, one by one, in a systematic way:

3. OUTLIER DETECTION AND REMOVAL
Systematically identify and remove outliers from numeric columns:

a) Examine each numeric column (except target and fold):
b) Apply outlier detection methods
c) Remove outlier rows


4. STRATIFIED 5-FOLD CROSS-VALIDATION
After all cleaning is complete, create validation folds:

a) Prepare the cleaned dataset
b) Create stratified folds
c) Add fold column
d) Verify stratification


5. Save and submit:
   - Save the cleaned dataset with fold column as '{RAW_DATA_PATH.stem}_cleaned.csv'
   - Call submit_answer with: {{"cleaned_csv": "{RAW_DATA_PATH.stem}_cleaned.csv"}}

TOOLS:
Use python_expression tool to execute Python code. pandas, numpy, and sklearn are available. State persists across tool calls. RAW_DATA_PATH variable contains the dataset path.
""".strip()

    num_runs = 10
    tasks = [
        run_single_test(i + 1, prompt, tools, handlers, verbose=(i == 0))
        for i in range(num_runs)
    ]

    if concurrent:
        results = [await coro for coro in asyncio.as_completed(tasks)]
    else:
        results = [await task for task in tasks]

    successes = sum(1 for _, ok in results if ok)
    print("=" * 60)
    print(f"Passed: {successes}/{num_runs}")
    print(f"Failed: {num_runs - successes}/{num_runs}")
    print(f"Pass Rate: {successes / num_runs * 100:.1f}% (target 10-40%)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main(concurrent=True))
