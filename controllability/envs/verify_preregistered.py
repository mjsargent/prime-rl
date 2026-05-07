from __future__ import annotations

import argparse
import sys

from controllability.envs.env_loader import load_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", nargs="+")
    args = parser.parse_args()

    failures = []
    for env_id in args.env_id:
        print(f"=== {env_id}")
        try:
            env = load_environment(env_id)
            eval_dataset = env.get_eval_dataset()
            first = eval_dataset[0] if len(eval_dataset) else None
            print("status: pass")
            print("type:", type(env).__name__)
            print("eval_len:", len(eval_dataset))
            print("first_keys:", sorted(first.keys()) if isinstance(first, dict) else None)
            print("rubric:", type(env.rubric).__name__)
        except Exception as exc:
            failures.append((env_id, exc))
            print("status: fail")
            print("error_type:", type(exc).__name__)
            print("error:", exc)

    if failures:
        print("failures:", ", ".join(env_id for env_id, _ in failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
