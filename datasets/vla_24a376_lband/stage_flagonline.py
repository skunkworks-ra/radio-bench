"""Stage input.flagonline.txt into compiled 24A-376 raw-input Harbor tasks via YAML AST."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

TASKS_DIR = Path("/mnt/fast/radio-bench/tasks_24a376_orch_opt")
FLAGONLINE_SRC = Path("/mnt/fast/radio_bench_data/24a376_clean/raw/input.flagonline.txt")
RAW_PREFIXES = ("r1_24a376_inspect__", "r1_24a376_precal__", "r2_24a376_e2e__")
ANCHOR = "cp -a --no-preserve=ownership /seed_data/input.ms /workspace/input.ms"
INJECT = (
    "cp -a --no-preserve=ownership /seed_data/input.flagonline.txt /workspace/input.flagonline.txt"
)


def main() -> None:
    count = 0
    for task_dir in sorted(TASKS_DIR.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith(RAW_PREFIXES):
            continue
        data_dir = task_dir / "environment" / "data"
        compose_path = task_dir / "environment" / "docker-compose.yaml"
        if not data_dir.is_dir() or not compose_path.is_file():
            continue
        shutil.copy2(FLAGONLINE_SRC, data_dir / "input.flagonline.txt")
        doc = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        cmd = doc["services"]["main"]["command"]
        script = cmd[2]
        if INJECT not in script:
            if ANCHOR not in script:
                raise RuntimeError(f"Missing anchor in {compose_path}")
            cmd[2] = script.replace(ANCHOR, f"{ANCHOR}\n{INJECT}", 1)
            compose_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        count += 1
    print(f"Staged input.flagonline.txt into {count} raw-input Harbor tasks in {TASKS_DIR}")


if __name__ == "__main__":
    main()
