"""Convert the 11 ISPRAS ISCAS85 Verilog files and prove each AIG equivalent."""

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.abc import ActionBudget, evaluate_recipe
from common.actions import MAX_TOOL_CALLS

CIRCUITS = ("c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552")


def yosys(commands, log):
    try:
        process = subprocess.run(
            ["yosys", "-Q", "-T", "-p", commands], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=60, check=False,
        )
        log.write_text(process.stdout)
        if process.returncode:
            raise RuntimeError(f"Yosys exited {process.returncode}; see {log}")
        return process.stdout
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        log.write_text(output.decode(errors="replace") if isinstance(output, bytes) else output)
        raise RuntimeError(f"Yosys timed out; see {log}") from exc


def main():
    destination = ROOT / "benchmarks/converted/iscas85"
    evidence = ROOT / "results/iscas85_validation"
    destination.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in CIRCUITS:
        source = f"benchmarks/iscas85/iscas85/verilog/{name}.v"
        row = {"circuit": name, "source": source, "top": name,
               "conversion": "FAIL", "equivalence": "UNVERIFIED"}
        try:
            row["source_sha256"] = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
            with tempfile.TemporaryDirectory(prefix=f"{name}-aiger-") as directory:
                directory = Path(directory)
                aig = directory / f"{name}.aig"
                source_json = directory / "source.json"
                yosys(
                    f"read_verilog {source}; hierarchy -check -top {name}; check -assert; "
                    f"write_json {source_json}; techmap; aigmap; write_aiger -symbols {aig}",
                    evidence / f"{name}_conversion.log",
                )
                gold = json.loads(source_json.read_text())["modules"][name]
                row["inputs"] = sum(len(p["bits"]) for p in gold["ports"].values() if p["direction"] == "input")
                row["outputs"] = sum(len(p["bits"]) for p in gold["ports"].values() if p["direction"] == "output")
                types = sorted({cell["type"] for cell in gold["cells"].values()})
                row["source_cell_types"] = " ".join(types)
                if not set(types) <= {"$and", "$not", "$or", "$xor", "$xnor"}:
                    raise RuntimeError(f"Unexpected source cells: {types}")
                row["combinational"] = "YES"
                row["technology_cells"] = "NONE"
                header = aig.read_bytes().split(b"\n", 1)[0].split()
                if header[0] != b"aig" or int(header[3]) != 0:
                    raise RuntimeError("Expected binary combinational AIGER")
                if (int(header[2]), int(header[4])) != (row["inputs"], row["outputs"]):
                    raise RuntimeError("Source/AIGER input-output count mismatch")
                result = evaluate_recipe(aig, [], budget=ActionBudget(MAX_TOOL_CALLS), metric="aig", timeout=30)
                (evidence / f"{name}_abc.log").write_text(result["abc_output"])
                row.update(result["metrics"])
                row["aig_sha256"] = hashlib.sha256(aig.read_bytes()).hexdigest()
                shutil.copyfile(aig, destination / f"{name}.aig")
                row["conversion"] = "PASS"
                # Independent truth-table BLIF export; gold never uses aigmap.
                source_blif = directory / "source.blif"
                yosys(
                    f"read_verilog {source}; hierarchy -check -top {name}; check -assert; "
                    f"simplemap; write_blif {source_blif}",
                    evidence / f"{name}_source_blif.log",
                )
                process = subprocess.run(
                    ["yosys-abc", "-s", "-c", f"cec {source_blif} {aig}; echo CEC_COMPLETE"],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, timeout=60, check=False,
                )
                (evidence / f"{name}_cec.log").write_text(process.stdout)
                # ABC CEC matches primary inputs/outputs by name by default.
                if process.returncode or "CEC_COMPLETE" not in [line.strip() for line in process.stdout.splitlines()]:
                    raise RuntimeError("ABC equivalence command failed")
                if not re.search(r"(?m)^Networks are equivalent[ .]", process.stdout):
                    raise RuntimeError("ABC did not prove source BLIF and converted AIG equivalent")
                row["equivalence"] = "PASS"
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            row["error"] = str(exc)
            print(f"FAIL {name}: {exc}", file=sys.stderr)
        print(f"{name}: conversion={row['conversion']}, equivalence={row['equivalence']}")
        rows.append(row)
    fields = ["circuit", "source", "top", "inputs", "outputs", "combinational", "technology_cells",
              "source_cell_types", "aig_nodes", "aig_levels", "conversion", "equivalence",
              "source_sha256", "aig_sha256", "error"]
    with (evidence / "checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(row["equivalence"] != "PASS" for row in rows)
    print(f"{len(rows) - failures}/{len(rows)} equivalent; evidence: {evidence}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
