"""Shared-warmup process branching without forking an initialized CUDA runtime.

The training process writes one checkpoint and execs this module's lightweight
supervisor. Replacing that process releases its entire CUDA context before the
two independent Python training processes start on the inherited GPU devices.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def parse_retrieval_mode(value):
    """Accept YAML booleans and True/False/Both strings; reject truthy typos."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        modes = {"true": True, "false": False, "both": "Both"}
        normalized = value.strip().lower()
        if normalized in modes:
            return modes[normalized]
    raise ValueError(f"Retrieval.enable must be True, False, or Both; got {value!r}")


def split_retrieval_override(extra_args, key):
    """Remove an enable override before YACS enforces its original bool type."""
    if len(extra_args) % 2:
        raise ValueError("Configuration overrides must be KEY VALUE pairs")
    remaining = []
    mode = None
    for index in range(0, len(extra_args), 2):
        name, value = extra_args[index:index + 2]
        if name == key:
            mode = parse_retrieval_mode(value)
        else:
            remaining.extend((name, value))
    return remaining, mode


def capture_rng_state():
    # Lazy imports keep the supervisor free of torch/CUDA initialization.
    import random
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    import random
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"].cpu())
    if state.get("cuda_all"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_all"]])
    elif "cuda" in state and torch.cuda.is_available():
        # Compatibility with the original STORM resume checkpoint format.
        torch.cuda.set_rng_state(state["cuda"].cpu())


def launch_training_branches(checkpoint_dir, enabled_command, disabled_command):
    """Replace this process with a supervisor of two fresh training processes.

    Callers must close environments and flush/close loggers first. Each command
    must explicitly select its boolean branch so a child cannot branch again.
    The shared checkpoint is intentionally retained for inspection and resume.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest = checkpoint_dir / "branches.json"
    manifest.write_text(json.dumps({
        "checkpoint_dir": str(checkpoint_dir),
        "cwd": os.getcwd(),
        "commands": {"retrieval_on": list(enabled_command), "retrieval_off": list(disabled_command)},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Starting concurrent Retrieval True/False branches from {checkpoint_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), "--supervise", str(manifest)])


def _stop_processes(processes):
    """Terminate whole branch groups, including their environment workers."""
    for process in processes:
        # A failed group leader can leave environment workers behind.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_branch_supervisor(manifest_path):
    """Launch both branches concurrently and propagate failures/interrupts."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    processes = {}
    exit_codes = {}

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(signum)

    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    result = 0
    try:
        for name, command in manifest["commands"].items():
            processes[name] = subprocess.Popen(command, cwd=manifest["cwd"], start_new_session=True)
        while len(exit_codes) < len(processes):
            for name, process in processes.items():
                if name in exit_codes:
                    continue
                code = process.poll()
                if code is None:
                    continue
                exit_codes[name] = code
                if code:
                    print(f"{name} failed with exit code {code}; stopping the other branch.", file=sys.stderr, flush=True)
                    result = code if code > 0 else 128 - code
                    return result
            if len(exit_codes) < len(processes):
                time.sleep(0.1)
        return 0
    except KeyboardInterrupt as error:
        signum = error.args[0] if error.args and isinstance(error.args[0], int) else signal.SIGINT
        result = 128 + signum
        return result
    except Exception:
        result = 1
        raise
    finally:
        # Ignore repeated shutdown signals while reaping subprocesses.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        _stop_processes(processes.values())
        exit_codes.update({name: process.returncode for name, process in processes.items()})
        (manifest_path.parent / "branch_results.json").write_text(
            json.dumps({"exit_codes": exit_codes, "supervisor_exit_code": result}, indent=2) + "\n",
            encoding="utf-8",
        )
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervise", required=True)
    raise SystemExit(run_branch_supervisor(parser.parse_args().supervise))
