#!/usr/bin/env python3
"""
OpenCode Agent CRUXEval Evaluation Script

Completely self-contained evaluation runner for CRUXEval benchmark using OpenCode as an agent.
Zero dependency on existing infrastructure/prompts in this repo.

Workflow:
1. For each task in cruxeval.jsonl, create an isolated workspace for OpenCode.
2. Prompt OpenCode agent to solve the task:
   - For output prediction: OpenCode investigates/executes the code and writes the predicted output to 'output.txt'.
   - For input prediction: OpenCode investigates/executes/searches and writes the predicted input to 'input.txt'.
3. Read 'output.txt' or 'input.txt' from the workspace.
4. Compare against the expected value via isolated Python execution verification.
5. Report individual results and the final pass rate.
"""

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
from pathlib import Path

# Paths resolved relative to this script
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "data" / "cruxeval.jsonl"


def load_dataset(dataset_path: Path = DATASET_PATH) -> list[dict[str, any]]:
  """Loads CRUXEval dataset from JSONL file."""
  if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

  records = []
  with open(dataset_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f):
      line = line.strip()
      if not line:
        continue
      data = json.loads(line)
      if "id" not in data:
        data["id"] = f"sample_{line_num}"
      records.append(data)
  return records


def make_agent_prompt(code: str, target_val: str, mode: str) -> str:
  """
  Constructs instructions for the OpenCode agent to write the answer
  into output.txt or input.txt.
  """
  if mode == "output":
    sample_input = target_val
    return f"""Here is a Python code:
```python
{code}
```

The function `f` is invoked with the input:
```python
{sample_input}
```

Your goal: Determine the exact return value of `f({sample_input})`.

Instructions:
1. You have full access to bash, python, tools, and the environment. You may run python scripts to inspect, test, or execute the code.
2. Once you have determined the exact return value / output literal, write ONLY the output value to a file named `output.txt` in the current working directory.
3. Do not include any extra text, markdown tags, or explanation in `output.txt`. Only the exact Python literal/value (e.g. `42`, `'result'`, `[1, 2]`, `{{'a': 1}}`).
"""
  elif mode == "input":
    sample_output = target_val
    return f"""Here is a Python code:
```python
{code}
```

The expected return value is:
```python
{sample_output}
```

Your goal:
Find an input argument or arguments such that executing `f(...)` returns the expected output.

Instructions:
1. You have full access to bash, python, tools, and the environment. You may run python scripts to test candidate inputs, search, or inspect execution.
2. Once you have found a valid input, write the input argument(s) or the full function call `f(...)` to a file named `input.txt` in the current working directory.
3. Do not include any extra text, markdown tags, or explanation in `input.txt`. Only the input (e.g. `'my_input'`, `1, 2`, `[1, 2, 3]`, or `f(1, 2)`).
"""
  else:
    raise ValueError(f"Unknown mode: {mode}")


def run_opencode_agent(
  prompt: str,
  workspace: Path,
  model: str,
  opencode: str = "opencode",
  timeout: int = 300,
) -> tuple[str, str, int]:
  """
  Executes OpenCode agent in the given workspace directory.
  Returns (stdout, stderr, exit_code).
  """
  cmd = [
    opencode,
    "run",
    prompt,
    "--model",
    model,
    "--auto",
    "--format",
    "json",
  ]

  try:
    proc = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      cwd=str(workspace),
    )
    stdout, stderr = proc.communicate(timeout=timeout)
    return stdout, stderr, proc.returncode
  except subprocess.TimeoutExpired:
    proc.kill()
    stdout, stderr = proc.communicate()
    # Save captured partial logs on timeout
    (workspace / "trajectory.jsonl").write_text(stdout or "", encoding="utf-8")
    (workspace / "error.txt").write_text(
      (stderr or "") + f"\nOpenCode timed out after {timeout}s\n",
      encoding="utf-8",
    )
    raise TimeoutError(f"OpenCode timed out after {timeout}s in {workspace}")
  except FileNotFoundError:
    raise RuntimeError(
      f"OpenCode executable '{opencode}' not found. "
      "Please ensure opencode is installed or specify --opencode-binary."
    )


def clean_target_file_content(content: str, mode: str) -> str:
  """Cleans up raw text read from input.txt or output.txt."""
  text = content.strip()

  # Remove markdown code block fences if present
  if "```" in text:
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)\s*```", text)
    if blocks:
      text = blocks[-1].strip()
    else:
      text = text.replace("```python", "").replace("```", "").strip()

  # Strip answer tags if present
  if "[ANSWER]" in text:
    text = text.split("[ANSWER]")[-1]
    if "[/ANSWER]" in text:
      text = text.split("[/ANSWER]")[0]
    text = text.strip()

  if mode == "output":
    if "==" in text:
      text = text.split("==")[-1].strip()
    if text.startswith("assert "):
      text = text.split("assert ", 1)[-1].strip()
    # Remove trailing comments
    text = re.sub(r"#.*$", "", text).strip()
    return text.strip()

  elif mode == "input":
    if "==" in text:
      text = text.split("==")[0].strip()
    if text.startswith("assert "):
      text = text.split("assert ", 1)[-1].strip()
    text = re.sub(r"#.*$", "", text).strip()
    return text.strip()

  return text.strip()


def verify_functional_correctness(
  code: str,
  answer: str,
  expected: str,
  mode: str,
  timeout: float = 4.0,
) -> tuple[bool, str]:
  """
  Self-contained sandboxed execution to verify if the model's answer is correct.
  Does not depend on any external modules.
  """
  if not answer:
    return False, "Answer is empty"

  if mode == "output":
    # Check if f(input) == predicted_output or expected_output == predicted_output
    test_script = f"""
{code}

# Test equivalence
try:
    _expected = {expected}
except Exception as _e:
    _expected = None

_predicted_raw = {repr(answer)}

try:
    _predicted_eval = eval({repr(answer)})
except Exception:
    _predicted_eval = _predicted_raw

if _expected is not None and _predicted_eval == _expected:
    sys.exit(0)

# Check assertion
try:
    assert _expected == _predicted_eval
    sys.exit(0)
except Exception:
    pass

try:
    assert {expected} == {answer}
    sys.exit(0)
except Exception as e:
    sys.exit(1)
"""
  elif mode == "input":
    # Format the call: either f(...) or f(arg)
    if answer.startswith("f(") and answer.endswith(")"):
      call_expr = answer
    else:
      call_expr = f"f({answer})"

    test_script = f"""
{code}

try:
    _expected = {expected}
    _actual = {call_expr}
    assert _actual == _expected
    sys.exit(0)
except Exception as e:
    sys.exit(1)
"""
  else:
    return False, f"Invalid mode {mode}"

  try:
    proc = subprocess.run(
      [sys.executable, "-c", test_script],
      capture_output=True,
      text=True,
      timeout=timeout,
    )
    if proc.returncode == 0:
      return True, "Passed"
    else:
      err = proc.stderr.strip() or proc.stdout.strip()
      return False, f"Assertion failed: {err}" if err else "Assertion failed"
  except subprocess.TimeoutExpired:
    return False, f"Verification execution timed out ({timeout}s)"
  except Exception as e:
    return False, f"Verification error: {e}"


def evaluate_task(
  sample: dict[str, any],
  mode: str,
  model: str,
  workspace: Path,
  opencode: str,
  timeout: int,
  verbose: bool = False,
) -> dict[str, any]:
  """Runs a single task with OpenCode agent in its workspace and evaluates result."""
  sample_id = sample["id"]
  code = sample["code"]
  input_val = sample.get("input", "")
  output_val = sample.get("output", "")

  target_val = input_val if mode == "output" else output_val
  target_filename = "output.txt" if mode == "output" else "input.txt"

  # Setup isolated task workspace
  workspace.mkdir(parents=True, exist_ok=True)

  # Save code to workspace for agent convenience
  with open(workspace / "code.py", "w", encoding="utf-8") as f:
    f.write(code + "\n")

  prompt = make_agent_prompt(code=code, target_val=target_val, mode=mode)

  result = {
    "id": sample_id,
    "input": input_val,
    "output": output_val,
    "raw_answer": None,
    "answer": None,
    "correct": False,
    "error": None,
    "workspace": str(workspace),
  }

  try:
    stdout, stderr, _ = run_opencode_agent(
      prompt=prompt,
      workspace=workspace,
      model=model,
      opencode=opencode,
      timeout=timeout,
    )

    # Save agent stdout to trajectory.jsonl and stderr to error.txt
    (workspace / "trajectory.jsonl").write_text(stdout or "", encoding="utf-8")
    (workspace / "error.txt").write_text(stderr or "", encoding="utf-8")

    target_file_path = workspace / target_filename

    if target_file_path.exists():
      raw_answer = target_file_path.read_text(encoding="utf-8").strip()
      result["raw_answer"] = raw_answer
      result["answer"] = clean_target_file_content(raw_answer, mode)

      correct, msg = verify_functional_correctness(
        code=code,
        answer=result["answer"],
        expected=output_val,
        mode=mode,
      )
      result["correct"] = correct
      result["error"] = None if correct else msg
    else:
      result["error"] = f"File {target_filename} was not created by OpenCode"

    if verbose:
      status = "✅ PASS" if result["correct"] else "❌ FAIL"
      content_preview = (result["raw_answer"] or result["error"] or "")[:60]
      print(f"[{sample_id}] {status} | {target_filename}: {content_preview}")

  except Exception as exc:
    result["error"] = str(exc)
    err_path = workspace / "error.txt"
    existing_err = err_path.read_text(encoding="utf-8") if err_path.exists() else ""
    err_path.write_text(f"{existing_err}\nException: {exc}\n".strip(), encoding="utf-8")
    if verbose:
      print(f"[{sample_id}] ⚠️ ERROR: {exc}")

  return result


def main():
  parser = argparse.ArgumentParser(
    description="Self-contained OpenCode agent evaluation on CRUXEval (input/output prediction via file)"
  )
  parser.add_argument(
    "--mode",
    type=str,
    choices=["output", "input"],
    default="output",
    help="Evaluation mode: 'output' (write to output.txt) or 'input' (write to input.txt)",
  )
  parser.add_argument(
    "--model",
    type=str,
    default="opencode/deepseek-v4-flash-free",
    help="Model passed to OpenCode (default: opencode/deepseek-v4-flash-free)",
  )
  parser.add_argument(
    "--num-workers",
    "-j",
    type=int,
    default=4,
    help="Concurrency / parallel worker processes (default: 4)",
  )
  parser.add_argument(
    "--start",
    "-s",
    type=int,
    default=0,
    help="Start index in dataset (default: 0)",
  )
  parser.add_argument(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Limit evaluation to N tasks (default: all remaining from start)",
  )
  parser.add_argument(
    "--timeout",
    type=int,
    default=300,
    help="Timeout in seconds for OpenCode agent per task (default: 300s / 5min)",
  )
  parser.add_argument(
    "--opencode",
    type=str,
    default="opencode",
    help="Path or name of opencode executable (default: opencode)",
  )
  parser.add_argument(
    "--outdir",
    "-o",
    type=str,
    default="codokus/output",
    help="Output directory to store workspaces and result.json (default: codokus/output)",
  )
  parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Print detailed logs per task",
  )

  args = parser.parse_args()

  # Load dataset
  dataset = load_dataset()
  start = max(0, args.start)
  if args.limit is not None:
    samples = dataset[start : start + args.limit]
  else:
    samples = dataset[start:]

  total_tasks = len(samples)
  outdir = Path(args.outdir).resolve()
  workspace_base = outdir / "workspace"
  workspace_base.mkdir(parents=True, exist_ok=True)

  print("=" * 70)
  print("🤖 OpenCode Agent CRUXEval Evaluation")
  print(
    f"   Mode         : CRUXEval-{'O (Output -> output.txt)' if args.mode == 'output' else 'I (Input -> input.txt)'}"
  )
  print(f"   Model        : {args.model}")
  print(f"   Tasks        : {total_tasks} samples")
  print(f"   Workers      : {args.num_workers} parallel workers")
  print(f"   Timeout      : {args.timeout}s per task")
  print(f"   Output Dir   : {outdir}")
  print(f"   Workspaces   : {workspace_base}")
  print("=" * 70)

  results = []
  passed = 0
  missing_answer = 0
  answer_incorrect = 0

  if args.num_workers > 1 and total_tasks > 1:
    with concurrent.futures.ThreadPoolExecutor(
      max_workers=args.num_workers
    ) as executor:
      future_to_sample = {
        executor.submit(
          evaluate_task,
          sample=sample,
          mode=args.mode,
          model=args.model,
          workspace=workspace_base / sample["id"],
          opencode=args.opencode,
          timeout=args.timeout,
          verbose=args.verbose,
        ): sample
        for sample in samples
      }

      for i, future in enumerate(
        concurrent.futures.as_completed(future_to_sample), start=1
      ):
        res = future.result()
        results.append(res)

        if res["correct"]:
          passed += 1
        elif res["raw_answer"] is None:
          missing_answer += 1
        else:
          answer_incorrect += 1

        rate = (passed / i) * 100
        if not args.verbose:
          print(
            f"[{i}/{total_tasks}] Passed: {passed} | Incorrect: {answer_incorrect} | "
            f"Missing Answer: {missing_answer} | Current Pass Rate: {rate:.2f}%",
            end="\r",
            flush=True,
          )
  else:
    for i, sample in enumerate(samples, start=1):
      res = evaluate_task(
        sample=sample,
        mode=args.mode,
        model=args.model,
        workspace=workspace_base / sample["id"],
        opencode=args.opencode,
        timeout=args.timeout,
        verbose=args.verbose,
      )
      results.append(res)

      if res["correct"]:
        passed += 1
      elif res["raw_answer"] is None:
        missing_answer += 1
      else:
        answer_incorrect += 1

      rate = (passed / i) * 100
      if not args.verbose:
        print(
          f"[{i}/{total_tasks}] Passed: {passed} | Incorrect: {answer_incorrect} | "
          f"Missing Answer: {missing_answer} | Current Pass Rate: {rate:.2f}%",
          end="\r",
          flush=True,
        )

  print("\n" + "=" * 70)
  final_pass_rate = (passed / total_tasks * 100) if total_tasks > 0 else 0.0
  total_failed = missing_answer + answer_incorrect

  print("📊 EVALUATION RESULTS")
  print(f"   Task Mode        : CRUXEval-{args.mode.upper()}")
  print(f"   Model            : {args.model}")
  print(f"   Total Tasks      : {total_tasks}")
  print(f"   Passed           : {passed}")
  print(
    f"   Failed           : {total_failed} (missing_answer: {missing_answer}, answer_incorrect: {answer_incorrect})"
  )
  print(f"   🏆 Pass Rate     : {final_pass_rate:.2f}%")
  print("=" * 70)

  # Save summary and results JSON
  output_file = outdir / "result.json"

  summary = {
    "benchmark": "CRUXEval",
    "mode": args.mode,
    "model": args.model,
    "total_tasks": total_tasks,
    "passed": passed,
    "failed": {
      "missing_answer": missing_answer,
      "answer_incorrect": answer_incorrect,
    },
    "pass_rate": final_pass_rate,
    "results": sorted(results, key=lambda x: x["id"]),
  }

  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

  print(f"💾 Results saved to: {output_file}")


if __name__ == "__main__":
  main()
