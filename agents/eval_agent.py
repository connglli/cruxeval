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
import os
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


def make_code_file(code: str, input_val: str, output_val: str, mode: str) -> str:
  """
  Generates code.py with problem definition and a main test block that calls
  get_output() or get_input() from answer.py.
  """
  if mode == "output":
    return f"""{code}


if __name__ == "__main__":
    from answer import get_output
    predicted = get_output()
    print(f"__ANSWER__={{repr(predicted)}}")
    expected = f({input_val})
    assert predicted == expected, f"Mismatch: expected {{expected!r}}, got {{predicted!r}}"
    print("✅ Correct!")
"""
  elif mode == "input":
    return f"""{code}


if __name__ == "__main__":
    from answer import get_input
    inp = get_input()
    print(f"__ANSWER__={{repr(inp)}}")
    if isinstance(inp, tuple):
        try:
            actual = f(*inp)
        except TypeError:
            actual = f(inp)
    else:
        actual = f(inp)
    expected = {output_val}
    assert actual == expected, f"Mismatch: expected {{expected!r}}, got {{actual!r}}"
    print("✅ Correct!")
"""
  else:
    raise ValueError(f"Unknown mode: {mode}")


def make_agent_prompt(input_val: str, output_val: str, mode: str) -> str:
  """
  Constructs prompt instructions for the agent to implement get_output() or get_input()
  in answer.py.
  """
  if mode == "output":
    return f"""You are solving a Python code execution task.

In the current working directory, you will find `code.py` which defines a function `f` and a self-test script.

Your goal:
Determine the exact return value of calling `f({input_val})`.

Instructions:
1. Create a file named `answer.py` in the current working directory containing a function `get_output()` that returns your predicted output.
   Example `answer.py`:
   ```python
   def get_output():
       return 42
   ```
2. You have full access to bash, python, tools, and the environment.
3. You can test your solution at any time by running `python3 code.py`.
4. Save answer.py in the current working directory and exit when the above test passes. Otherwise, continue iterating until you find the correct output.
"""
  elif mode == "input":
    return f"""You are solving a Python input prediction task.

In the current working directory, you will find `code.py` which defines a function `f` and a self-test script testing against the expected output:
{output_val}

Your goal:
Find an input argument or arguments such that executing `f(...)` returns the expected output.

Instructions:
1. Create a file named `answer.py` in the current working directory containing a function `get_input()` that returns the input argument(s).
   Example `answer.py`:
   ```python
   def get_input():
       return [1, 2, 3]
   ```
2. You have full access to bash, python, tools, and the environment.
3. You can test your solution at any time by running `python3 code.py`.
4. Save answer.py in the current working directory and exit when the above test passes. Otherwise, continue iterating until you find the correct input.
"""
  else:
    raise ValueError(f"Unknown mode: {mode}")


def run_agent(
  prompt: str,
  workspace: Path,
  agent: str,
  model: str,
  timeout: int = 300,
) -> int:
  """
  Executes an AI coding agent (OpenCode or Claude Code) in the given workspace,
  piping stdout directly to traj.jsonl and stderr to error.txt.
  Returns exit_code.
  """
  env = os.environ.copy()

  if agent == "opencode":
    cmd = [
      "opencode",
      "run",
      prompt,
      "--model",
      model,
      "--auto",
      "--format",
      "json",
    ]
  elif agent == "claude":
    cmd = [
      "claude",
      "--print",
      "--verbose",
      "--model",
      model,
      "--output-format",
      "stream-json",
      "--dangerously-skip-permissions",
      prompt,
    ]
    # Set default Claude model environment variables
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    env["CLAUDE_CODE_SUBAGENT_MODEL"] = model
  else:
    raise ValueError(f"Unsupported agent '{agent}'. Choose 'opencode' or 'claude'.")

  traj_path = workspace / "traj.jsonl"
  error_path = workspace / "error.txt"

  with (
    open(traj_path, "w", encoding="utf-8") as fout,
    open(error_path, "w", encoding="utf-8") as ferr,
  ):
    try:
      proc = subprocess.Popen(
        cmd,
        stdout=fout,
        stderr=ferr,
        cwd=str(workspace),
        env=env,
      )
      return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
      proc.kill()
      proc.wait()
      with open(error_path, "a", encoding="utf-8") as append_err:
        append_err.write(f"\n{agent} timed out after {timeout}s\n")
      raise TimeoutError(f"{agent} timed out after {timeout}s in {workspace}")
    except FileNotFoundError:
      raise RuntimeError(
        f"Agent executable '{agent}' not found. Please ensure {agent} is installed."
      )


def verify_functional_correctness(
  workspace: Path,
  clean_code_content: str,
  timeout: float = 4.0,
) -> tuple[bool, str | None, str]:
  """
  Independently executes code.py in workspace to verify answer.py.
  Restores trusted code.py before execution to prevent tampered assertions.
  Returns (correct: bool, evaluated_answer_repr: str | None, error_or_status: str).
  """
  answer_file = workspace / "answer.py"
  if not answer_file.exists():
    return False, None, "File answer.py was not created by agent"

  # Restore trusted code.py to ensure the assertion logic is authentic
  (workspace / "code.py").write_text(clean_code_content, encoding="utf-8")

  try:
    proc = subprocess.run(
      [sys.executable, "code.py"],
      cwd=str(workspace),
      capture_output=True,
      text=True,
      timeout=timeout,
    )

    # Extract evaluated answer representation from stdout
    answer_repr = None
    for line in proc.stdout.splitlines():
      if line.startswith("__ANSWER__="):
        answer_repr = line[len("__ANSWER__=") :]
        break

    if proc.returncode == 0:
      return True, answer_repr, "Passed"
    else:
      err = proc.stderr.strip() or proc.stdout.strip()
      return (
        False,
        answer_repr,
        f"Assertion failed: {err}" if err else "Assertion failed",
      )

  except subprocess.TimeoutExpired:
    return False, None, f"Verification execution timed out ({timeout}s)"
  except Exception as e:
    return False, None, f"Verification error: {e}"


def evaluate_task(
  sample: dict[str, any],
  mode: str,
  agent: str,
  model: str,
  workspace: Path,
  timeout: int,
  verbose: bool = False,
) -> dict[str, any]:
  """Runs a single task with an agent (OpenCode or Claude) in its workspace and evaluates result."""
  sample_id = sample["id"]
  code = sample["code"]
  input_val = sample.get("input", "")
  output_val = sample.get("output", "")

  # Setup isolated task workspace
  workspace.mkdir(parents=True, exist_ok=True)

  # Generate code.py with main test block
  code_content = make_code_file(
    code=code, input_val=input_val, output_val=output_val, mode=mode
  )
  with open(workspace / "code.py", "w", encoding="utf-8") as f:
    f.write(code_content)

  prompt = make_agent_prompt(input_val=input_val, output_val=output_val, mode=mode)

  result = {
    "id": sample_id,
    "input": input_val,
    "output": output_val,
    "answer": None,
    "correct": False,
    "error": None,
    "workspace": str(workspace),
  }

  try:
    run_agent(
      prompt=prompt,
      workspace=workspace,
      agent=agent,
      model=model,
      timeout=timeout,
    )

    answer_file_path = workspace / "answer.py"

    if answer_file_path.exists():
      correct, answer_val, msg = verify_functional_correctness(
        workspace=workspace,
        clean_code_content=code_content,
      )
      result["answer"] = answer_val
      result["correct"] = correct
      result["error"] = None if correct else msg
    else:
      result["error"] = f"File answer.py was not created by {agent}"

    if verbose:
      status = "✅ PASS" if result["correct"] else "❌ FAIL"
      content_preview = (result["answer"] or result["error"] or "")[:60]
      print(f"[{sample_id}] {status} | answer.py: {content_preview}", flush=True)

  except Exception as exc:
    result["error"] = str(exc)
    err_path = workspace / "error.txt"
    existing_err = err_path.read_text(encoding="utf-8") if err_path.exists() else ""
    err_path.write_text(f"{existing_err}\nException: {exc}\n".strip(), encoding="utf-8")
    if verbose:
      print(f"[{sample_id}] ⚠️ ERROR: {exc}", flush=True)

  return result


def main():
  parser = argparse.ArgumentParser(
    description="Self-contained AI agent evaluation on CRUXEval (OpenCode & Claude Code)"
  )
  parser.add_argument(
    "--agent",
    type=str,
    choices=["opencode", "claude"],
    default="opencode",
    help="Agent to evaluate: 'opencode' or 'claude' (default: opencode)",
  )
  parser.add_argument(
    "--mode",
    type=str,
    choices=["output", "input"],
    default="output",
    help="Evaluation mode: 'output' (answer.py with get_output()) or 'input' (answer.py with get_input())",
  )
  parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Model passed to agent (default: 'opencode/deepseek-v4-flash-free' for opencode, 'claude-3-5-sonnet-20241022' for claude)",
  )
  parser.add_argument(
    "--num-workers",
    "-j",
    type=int,
    default=1,
    help="Concurrency / parallel worker processes (default: 1)",
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
    help="Timeout in seconds for agent per task (default: 300s / 5min)",
  )
  parser.add_argument(
    "--outdir",
    "-o",
    type=str,
    default="agents/output",
    help="Output directory to store sample directories and result.json (default: agents/output)",
  )
  parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Print detailed logs per task",
  )

  args = parser.parse_args()

  # Set default model based on agent if not specified
  if args.model is None:
    model = (
      "claude-3-5-sonnet-20241022"
      if args.agent == "claude"
      else "opencode/deepseek-v4-flash-free"
    )
  else:
    model = args.model

  # Load dataset
  dataset = load_dataset()
  start = max(0, args.start)
  if args.limit is not None:
    samples = dataset[start : start + args.limit]
  else:
    samples = dataset[start:]

  total_tasks = len(samples)
  outdir = Path(args.outdir).resolve()
  outdir.mkdir(parents=True, exist_ok=True)

  print("=" * 70)
  print(f"🤖 {args.agent.upper()} Agent CRUXEval Evaluation")
  print(
    f"   Mode         : CRUXEval-{
      'O (Output -> answer.py:get_output())'
      if args.mode == 'output'
      else 'I (Input -> answer.py:get_input())'
    }"
  )
  print(f"   Agent        : {args.agent}")
  print(f"   Model        : {model}")
  print(f"   Tasks        : {total_tasks} samples")
  print(f"   Workers      : {args.num_workers} parallel workers")
  print(f"   Timeout      : {args.timeout}s per task")
  print(f"   Outdir       : {outdir}")
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
          agent=args.agent,
          model=model,
          workspace=outdir / sample["id"],
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
        elif res["answer"] is None:
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
        agent=args.agent,
        model=model,
        workspace=outdir / sample["id"],
        timeout=args.timeout,
        verbose=args.verbose,
      )
      results.append(res)

      if res["correct"]:
        passed += 1
      elif res["answer"] is None:
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
  print(f"   Agent            : {args.agent}")
  print(f"   Task Mode        : CRUXEval-{args.mode.upper()}")
  print(f"   Model            : {model}")
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
    "agent": args.agent,
    "mode": args.mode,
    "model": model,
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
