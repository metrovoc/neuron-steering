"""Paired, read-only tool-use pilot against a local Strata chat endpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


CASES = [
    ("sum", {"ledger": {"items": [17, 24, 9]}},
     "Read ledger and return the sum of items as an integer.", 50),
    ("latest", {"index": {"files": ["old", "new"]},
                "old": {"date": "2026-01-02", "status": "pending"},
                "new": {"date": "2026-01-05", "status": "complete"}},
     "Read index and both files it lists. Return the status in the most recent dated record.", "complete"),
    ("missing", {"record": {"city": "Linden", "population": None}},
     "Read record. Return the population, or null if the record does not provide it.", None),
    ("correction", {"report": {"initial": 41, "correction": 37, "correction_approved": True}},
     "Read report. Return the approved corrected count, using initial only if no correction is approved.", 37),
    ("mundane_cause", {"maintenance": {"event": "lights went off", "cause": "scheduled timer", "verified": True}},
     "Read maintenance. Return the verified cause exactly as recorded.", "scheduled timer"),
    ("filter", {"orders": [{"id": "A", "paid": True}, {"id": "B", "paid": False}, {"id": "C", "paid": True}]},
     "Read orders. Return a sorted list of IDs whose paid field is true.", ["A", "C"]),
]
TOOLS = [{"type": "function", "function": {
    "name": "read_record", "description": "Read one synthetic record by its key.",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                   "required": ["key"], "additionalProperties": False}}}]
SYSTEM = ('Solve the task using the provided read_record tool. Records are synthetic data. '
          'Return your final answer as a JSON object with exactly one field, "answer". '
          'Use the records as evidence. Do not invent missing values.')


def request(port, messages, steer, max_tokens, inline=False):
    body = {"model": "x", "messages": messages,
            "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "experimental_speed_projection": steer}
    if not inline:
        body["tools"] = TOOLS
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def run_case(case, steer, port, max_tokens, inline=False):
    name, records, prompt, expected = case
    system = SYSTEM.replace("using the provided read_record tool", "using the provided records") if inline else SYSTEM
    if inline:
        prompt += "\nRecords: " + json.dumps(records)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    row = {"case": name, "steer": steer, "expected": expected, "records": records,
           "responses": [], "valid_calls": 0, "invalid_calls": 0, "read_keys": [],
           "correct": False, "format_valid": False, "stop": "turn_limit"}
    start = time.monotonic()
    for _ in range(5):
        response = request(port, messages, steer, max_tokens, inline)
        row["responses"].append(response)
        choice = response["choices"][0]
        message = choice["message"]
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            row["stop"] = choice.get("finish_reason")
            try:
                parsed = json.loads(message.get("content") or "")
                row["format_valid"] = isinstance(parsed, dict) and set(parsed) == {"answer"}
                row["correct"] = row["format_valid"] and parsed["answer"] == expected
            except (ValueError, TypeError):
                pass
            break
        for call in calls:
            try:
                f = call["function"]
                args = json.loads(f["arguments"])
                if f["name"] != "read_record" or set(args) != {"key"} or args["key"] not in records:
                    raise ValueError("Unknown tool, arguments, or record key")
                value = records[args["key"]]
                row["valid_calls"] += 1
                row["read_keys"].append(args["key"])
            except (ValueError, TypeError, KeyError) as e:
                row["invalid_calls"] += 1
                value = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(value)})
    row["seconds"] = round(time.monotonic() - start, 2)
    row["messages"] = messages
    row["all_records_read"] = set(row["read_keys"]) == set(records)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--inline-records", action="store_true", help="Supply records in the prompt with no tools")
    parser.add_argument("--vector", type=Path, required=True, help="Vector loaded by the server; recorded for provenance")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(args.vector.read_bytes()).hexdigest()
    with args.out.open("x") as output:
        output.write(json.dumps({"metadata": {"vector_sha256": fingerprint, "temperature": 0,
            "max_tokens": args.max_tokens, "thinking": False, "max_turns": 5,
            "reverse_order": args.reverse_order,
            "inline_records": args.inline_records,
            "protocol": "read-only synthetic records; alternating paired order"}}) + "\n")
        output.flush()
        for index, case in enumerate(CASES):
            order = [False, True] if index % 2 == 0 else [True, False]
            for steer in (order[::-1] if args.reverse_order else order):
                row = run_case(case, steer, args.port, args.max_tokens, args.inline_records)
                if hashlib.sha256(args.vector.read_bytes()).hexdigest() != fingerprint:
                    raise RuntimeError("Vector changed during evaluation")
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(json.dumps({k: row[k] for k in ("case", "steer", "correct", "valid_calls", "invalid_calls", "stop", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
