"""Run paired mini-SWE-agent coding tasks in Docker's Linux environment.

Install mini-swe-agent==2.4.6 in a separate evaluation environment.
On macOS, Docker Desktop executes these containers in its Linux VM.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import time

import yaml
import minisweagent
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_model import LitellmModel


TASKS = {
    "paid_total": {
        "prompt": "Fix paid_total in ledger.py. Sum the amounts of only paid orders. Empty input must return 0. Preserve the input records. Inspect the code, fix it, and run tests before submitting.",
        "source": 'def paid_total(orders):\n    return sum(order["amount"] for order in orders)\n',
        "module": "ledger",
        "visible": 'from ledger import paid_total\nassert paid_total([{"amount": 5, "paid": True}, {"amount": 9, "paid": False}]) == 5\nprint("visible test passed")\n',
        "tests": 'import copy\nfrom ledger import paid_total\nassert paid_total([]) == 0\nassert paid_total([{"amount": 4, "paid": False}]) == 0\na = [{"amount": 11, "paid": True}, {"amount": -3, "paid": True}, {"amount": 100, "paid": False}]\nb = copy.deepcopy(a)\nassert paid_total(a) == 8\nassert a == b\nprint("held-out tests passed")\n',
    },
    "latest_status": {
        "prompt": "Fix latest_status in records.py. Records have ISO YYYY-MM-DD date strings and status strings. Return the status of the newest date, regardless of input order. Return None for empty input. Do not modify the input list. Inspect the code, fix it, and run tests before submitting.",
        "source": 'def latest_status(records):\n    return records[-1]["status"]\n',
        "module": "records",
        "visible": 'from records import latest_status\nassert latest_status([{"date": "2026-02-03", "status": "ready"}, {"date": "2026-01-01", "status": "pending"}]) == "ready"\nprint("visible test passed")\n',
        "tests": 'import copy\nfrom records import latest_status\nassert latest_status([]) is None\nassert latest_status([{"date":"2026-01-01","status":"one"}]) == "one"\na = [{"date":"2025-12-31","status":"old"},{"date":"2026-04-01","status":"new"},{"date":"2026-02-01","status":"middle"}]\nb = copy.deepcopy(a)\nassert latest_status(a) == "new"\nassert a == b\nassert latest_status(list(reversed(a))) == "new"\nprint("held-out tests passed")\n',
    },
}


def command(env, text):
    return env.execute({"command": text})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--vector", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image", default="python:3.13-slim")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--reverse-order", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    vector_hash = hashlib.sha256(a.vector.read_bytes()).hexdigest()
    cfg = yaml.safe_load((Path(minisweagent.__file__).parent / "config/mini.yaml").read_text())
    cfg["agent"].pop("mode", None)
    cfg["agent"].update(step_limit=a.steps, cost_limit=0, wall_time_limit_seconds=480)
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", a.image]))[0]
    metadata = {"harness": importlib.metadata.version("mini-swe-agent"),
                "vector_sha256": vector_hash, "image_id": image["Id"],
                "docker": json.loads(subprocess.check_output(["docker", "info", "--format", "{{json .OperatingSystem}}"])),
                "temperature": 0, "thinking": False, "max_tokens": 1024, "steps": a.steps,
                "reverse_order": a.reverse_order}
    (a.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    results = []
    for i, (name, task) in enumerate(TASKS.items()):
        order = [False, True] if i % 2 == 0 else [True, False]
        for steer in (order[::-1] if a.reverse_order else order):
            run_name = f"{name}-{'steered' if steer else 'base'}"
            print(f"Starting {run_name}", flush=True)
            env = DockerEnvironment(image=image["Id"], cwd="/workspace", timeout=20,
                run_args=["--rm", "--network=none", "--read-only", "--user=1000:1000",
                    "--cap-drop=ALL", "--security-opt=no-new-privileges", "--memory=512m", "--cpus=2",
                    "--pids-limit=128", "--tmpfs=/workspace:rw,size=32m,uid=1000,gid=1000",
                    "--tmpfs=/tmp:rw,size=32m,uid=1000,gid=1000"])
            started = time.monotonic()
            try:
                for filename, content in [(task["module"] + ".py", task["source"]), ("test_visible.py", task["visible"])]:
                    result = command(env, f"printf %s {shlex.quote(content)} > {shlex.quote(filename)}")
                    if result["returncode"]: raise RuntimeError(result)
                before = command(env, "python test_visible.py")
                if before["returncode"] == 0: raise RuntimeError("Fixture must fail before repair")
                model_cfg = cfg["model"] | {"model_name": "openai/qwen3.8-flash-next-q2_0", "cost_tracking": "ignore_errors"}
                model_cfg["model_kwargs"] = {"api_base": f"http://127.0.0.1:{a.port}/v1", "api_key": "local",
                    "temperature": 0, "max_tokens": 1024, "timeout": 180,
                    "extra_body": {"experimental_speed_projection": steer,
                                   "chat_template_kwargs": {"enable_thinking": False}}}
                model = LitellmModel(**model_cfg)
                agent = DefaultAgent(model, env, **cfg["agent"], output_path=a.out / f"{run_name}.traj.json")
                kernel = command(env, "uname -r")["output"].strip()
                result = agent.run(task["prompt"], system="Linux", release=kernel, version="Docker Linux VM", machine="aarch64")
                # Held-out tests are supplied only after the agent has finished.
                test = command(env, "python -c " + shlex.quote(task["tests"]))
                source = command(env, "cat " + task["module"] + ".py")["output"]
                (a.out / f"{run_name}.py").write_text(source)
                row = {"task": name, "steer": steer, "exit_status": result.get("exit_status"),
                       "calls": agent.n_calls, "test": test, "passed": test["returncode"] == 0,
                       "seconds": round(time.monotonic() - started, 2)}
                results.append(row)
                (a.out / "results.json").write_text(json.dumps(results, indent=2))
                print(json.dumps(row), flush=True)
                if hashlib.sha256(a.vector.read_bytes()).hexdigest() != vector_hash:
                    raise RuntimeError("Vector changed during evaluation")
            finally:
                subprocess.run(["docker", "rm", "-f", env.container_id], check=True, capture_output=True)
                env.container_id = None


if __name__ == "__main__":
    main()
