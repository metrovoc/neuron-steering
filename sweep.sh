#!/bin/sh
# usage: sweep.sh GGUF NAME SCALE...: one server per scale, every prompt steered and not
# (extra `fn_steer.py ask` arguments, e.g. --prompt, go in the environment as ASK_ARGS)
cd "$(dirname "$0")"
g=$1; n=$2; shift 2
for s in "$@"; do
  pkill -f "serve/server.py --engine strata" ; sleep 4
  uv run python fn_steer.py config --cvec "$g" --scale "$s" -o out/strata-steer.json
  vendor/Strata/.venv/bin/python vendor/Strata/serve/server.py --engine strata --config out/strata-steer.json --port 8080 > out/server-steer.log 2>&1 &
  sleep 15; until curl -s -m 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; do sleep 5; done
  eval uv run python fn_steer.py ask --name "\"$n-x$s\"" $ASK_ARGS > "out/$n-x$s.txt" 2>&1
done
pkill -f "serve/server.py --engine strata"
