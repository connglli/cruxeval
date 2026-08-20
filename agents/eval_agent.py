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

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Paths resolved relative to this script
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "data" / "cruxeval.jsonl"


def load_dataset(dataset_path: Path = DATASET_PATH) -> list[dict[str, Any]]:
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
  docker_image: str = "cruxeval-agent:latest",
) -> int:
  """
  Executes an AI coding agent (OpenCode or Claude Code) in an isolated Docker container
  mounting ONLY the workspace directory. Pipes stdout directly to traj.jsonl and stderr to error.txt.
  Returns exit_code.
  """
  env = os.environ.copy()

  if agent == "opencode":
    agent_cmd = [
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
    agent_cmd = [
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

  cmd = [
    "docker",
    "run",
    "--rm",
    "-e",
    "PYTHONUNBUFFERED=1",
    "-v",
    f"{workspace.resolve()}:/workspace",
    "-w",
    "/workspace",
  ]

  # Forward all environment variables present in eval_agent's environment
  for key, val in env.items():
    cmd.extend(["-e", f"{key}={val}"])

  cmd.append(docker_image)
  cmd.extend(agent_cmd)

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
      executable = cmd[0]
      raise RuntimeError(
        f"Executable '{executable}' not found. Please ensure Docker is installed and in PATH."
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
  sample: dict[str, Any],
  mode: str,
  agent: str,
  model: str,
  workspace: Path,
  timeout: int,
  docker_image: str = "cruxeval-agent:latest",
  verbose: bool = False,
) -> dict[str, Any]:
  """Runs a single task with an agent (OpenCode or Claude) in its workspace and evaluates result."""
  sample_id = sample["id"]
  sample_result_file = workspace / "result.json"

  # Resume from previous execution if sample result.json already exists
  if sample_result_file.exists():
    try:
      with open(sample_result_file, "r", encoding="utf-8") as f:
        cached_result = json.load(f)
      if verbose:
        status = "✅ PASS" if cached_result.get("correct") else "❌ FAIL"
        content_preview = (
          cached_result.get("answer") or cached_result.get("error") or ""
        )[:60]
        print(
          f"[{sample_id}] ⏩ RESUMED {status} ({cached_result.get('elapsed', 0.0)}s) | answer.py: {content_preview}",
          flush=True,
        )
      return cached_result
    except Exception:
      pass  # Corrupt or incomplete, re-run

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
    "elapsed": 0.0,
    "workspace": str(workspace),
  }

  start_time = time.perf_counter()
  try:
    run_agent(
      prompt=prompt,
      workspace=workspace,
      agent=agent,
      model=model,
      timeout=timeout,
      docker_image=docker_image,
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

  except Exception as exc:
    result["error"] = str(exc)
    err_path = workspace / "error.txt"
    existing_err = err_path.read_text(encoding="utf-8") if err_path.exists() else ""
    err_path.write_text(f"{existing_err}\nException: {exc}\n".strip(), encoding="utf-8")

  finally:
    result["elapsed"] = round(time.perf_counter() - start_time, 2)
    # Save per-sample result.json
    with open(sample_result_file, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2)

    if verbose:
      status = "✅ PASS" if result["correct"] else "❌ FAIL"
      content_preview = (result["answer"] or result["error"] or "")[:60]
      print(
        f"[{sample_id}] {status} ({result['elapsed']}s) | answer.py: {content_preview}",
        flush=True,
      )

  return result


def parse_throttle(throttle_str: str | None) -> tuple[int, float] | None:
  """
  Parses throttle specification 'num:seconds' (e.g. '5:10' for 10s after every 5 tasks).
  Also supports single number 'seconds' as '1:seconds'.
  Returns (interval: int, delay: float) or None if disabled.
  """
  if not throttle_str:
    return None

  throttle_str = str(throttle_str).strip()
  if ":" in throttle_str:
    parts = throttle_str.split(":", 1)
    try:
      interval = int(parts[0].strip())
      delay = float(parts[1].strip())
      if interval <= 0 or delay <= 0:
        return None
      return interval, delay
    except ValueError:
      raise argparse.ArgumentTypeError(
        f"Invalid throttle format '{throttle_str}'. Expected 'num:seconds' (e.g. '5:10' or '1:2.5')."
      )
  else:
    try:
      delay = float(throttle_str)
      if delay <= 0:
        return None
      return 1, delay
    except ValueError:
      raise argparse.ArgumentTypeError(
        f"Invalid throttle format '{throttle_str}'. Expected 'num:seconds' (e.g. '5:10' or '1:2.5')."
      )


def main():
  try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
  except Exception:
    pass

  parser = argparse.ArgumentParser(
    description="Self-contained AI agent evaluation on CRUXEval (OpenCode & Claude Code)"
  )
  parser.add_argument(
    "model",
    type=str,
    help="Model passed to agent (e.g., 'claude-opus-5', 'opencode/deepseek-v4-pro')",
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
    "--docker-image",
    type=str,
    default="cruxeval-agent:latest",
    help="Docker image for isolated task execution (default: cruxeval-agent:latest)",
  )
  parser.add_argument(
    "--throttle",
    type=parse_throttle,
    default=None,
    help="Throttling as 'num:seconds' to sleep for 'seconds' after every 'num' tasks (e.g. '5:10' or '1:2.5')",
  )
  parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Print detailed logs per task",
  )

  args = parser.parse_args()

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

  # Save configuration to command.json using vars(args)
  cmd_vars = vars(args).copy()
  if cmd_vars.get("throttle") is not None:
    cmd_vars["throttle"] = f"{cmd_vars['throttle'][0]}:{cmd_vars['throttle'][1]}"
  with open(outdir / "command.json", "w", encoding="utf-8") as f:
    json.dump(cmd_vars, f, indent=2)

  mode_desc = (
    "CRUXEval-O (Output -> answer.py:get_output())"
    if args.mode == "output"
    else "CRUXEval-I (Input -> answer.py:get_input())"
  )

  print("=" * 70)
  print(f"🤖 {args.agent.upper()} Agent CRUXEval Evaluation")
  print(f"   Mode         : {mode_desc}")
  print(f"   Agent        : {args.agent}")
  print(f"   Model        : {model}")
  print(f"   Docker Image : {args.docker_image}")
  print(f"   Tasks        : {total_tasks} samples")
  print(f"   Workers      : {args.num_workers} parallel workers")
  print(f"   Timeout      : {args.timeout}s per task")
  if args.throttle is not None:
    t_interval, t_delay = args.throttle
    print(f"   Throttle     : Sleep {t_delay}s after every {t_interval} task(s)")
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
      sample_iter = iter(enumerate(samples))
      future_to_sample = {}
      executed_count = 0

      def submit_next() -> bool:
        nonlocal executed_count
        try:
          idx, sample = next(sample_iter)
        except StopIteration:
          return False

        sample_res_file = outdir / sample["id"] / "result.json"
        is_cached = sample_res_file.exists()

        if not is_cached:
          executed_count += 1
          if (
            args.throttle is not None
            and executed_count > 1
            and (executed_count - 1) % args.throttle[0] == 0
          ):
            time.sleep(args.throttle[1])

        future = executor.submit(
          evaluate_task,
          sample=sample,
          mode=args.mode,
          agent=args.agent,
          model=model,
          workspace=outdir / sample["id"],
          timeout=args.timeout,
          docker_image=args.docker_image,
          verbose=args.verbose,
        )
        future_to_sample[future] = sample
        return True

      # Pre-fill worker pool
      for _ in range(min(args.num_workers, total_tasks)):
        if not submit_next():
          break

      completed_count = 0
      while future_to_sample:
        done, _ = concurrent.futures.wait(
          future_to_sample.keys(),
          return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for future in done:
          sample = future_to_sample.pop(future)
          res = future.result()
          results.append(res)
          completed_count += 1

          if res["correct"]:
            passed += 1
          elif res["answer"] is None:
            missing_answer += 1
          else:
            answer_incorrect += 1

          rate = (passed / completed_count) * 100
          if not args.verbose:
            print(
              f"\r\033[K[{completed_count}/{total_tasks}] Passed: {passed} | Incorrect: {answer_incorrect} | "
              f"Missing Answer: {missing_answer} | Current Pass Rate: {rate:.2f}%",
              end="",
              flush=True,
            )

          # Submit next task to keep pool full
          submit_next()
  else:
    executed_count = 0
    for i, sample in enumerate(samples, start=1):
      sample_res_file = outdir / sample["id"] / "result.json"
      is_cached = sample_res_file.exists()

      if not is_cached:
        executed_count += 1
        if (
          args.throttle is not None
          and executed_count > 1
          and (executed_count - 1) % args.throttle[0] == 0
        ):
          time.sleep(args.throttle[1])

      res = evaluate_task(
        sample=sample,
        mode=args.mode,
        agent=args.agent,
        model=model,
        workspace=outdir / sample["id"],
        timeout=args.timeout,
        docker_image=args.docker_image,
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
          f"\r\033[K[{i}/{total_tasks}] Passed: {passed} | Incorrect: {answer_incorrect} | "
          f"Missing Answer: {missing_answer} | Current Pass Rate: {rate:.2f}%",
          end="",
          flush=True,
        )

  if not args.verbose and total_tasks > 0:
    print()  # Ensure newline after progress line carriage return

  print("\n" + "=" * 70)
  final_pass_rate = (passed / total_tasks * 100) if total_tasks > 0 else 0.0
  total_failed = missing_answer + answer_incorrect
  total_elapsed = sum(r.get("elapsed", 0.0) for r in results)
  avg_elapsed = round(total_elapsed / len(results), 2) if results else 0.0

  print("📊 EVALUATION RESULTS")
  print(f"   Agent            : {args.agent}")
  print(f"   Task Mode        : CRUXEval-{args.mode.upper()}")
  print(f"   Model            : {model}")
  print(f"   Total Tasks      : {total_tasks}")
  print(f"   Passed           : {passed}")
  print(
    f"   Failed           : {total_failed} (missing_answer: {missing_answer}, answer_incorrect: {answer_incorrect})"
  )
  print(f"   🏆 Pass Rate      : {final_pass_rate:.2f}%")
  print(f"   ⏱️ Avg Elapsed    : {avg_elapsed:.2f}s")
  print("=" * 70)

  # Save summary JSON (individual task results are stored in each sample's result.json)
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
    "pass_rate": round(final_pass_rate, 2),
    "avg_elapsed": avg_elapsed,
  }

  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

  print(f"💾 Results saved to: {output_file}")


if __name__ == "__main__":
  main()
