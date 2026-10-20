"""
Deterministic scan of recorded AutoDiscovery records for substituted data.

A script that cannot read its data in the AutoDiscovery sandbox often carries
on with a stand-in and reports a result for it: on new-split-1 (2026-10-20)
id 46 lost its tensorstore open and computed "SNR" from radius and gap, id 89
had no tensorstore and used a "line-integral proxy", id 90 never read an image
and set intensity to 1/(1+0.08*gap); on new-merge-1 (2026-10-19) id 95 "fell
back to a coordinate representation" and named it a Hessian. Each was found
only by an agent reading the output. This scan names them up front.

It is evidence, not a verdict: a flagged record is one whose output or code
shows the marks of a stand-in, or whose hypothesis is about the image while
nothing shows an image was read. The predictive selector and the verifier
decide what the marks mean.

Per record (``audit_record``):
- ``output_markers``: output lines (glog lines aside) naming a proxy, fallback,
  mock, simulation, dummy, an unavailable or inaccessible source, or a missing
  module;
- ``code_substitutes``: code lines naming a proxy, simulation or synthetic value
  (the stand-in itself); ``code_mocks``: lines naming a mock, dummy or fallback
  (often an unpickling shim, so informational);
- ``image_hypothesis``, ``image_reader_in_code``, ``image_read_evidence``;
- ``flags`` and ``suspected`` (an output marker, a code substitute, or an image
  hypothesis without an image reader or without read evidence).

Usage
-----
    python agentic/recorded_audit.py autodiscovery/RUN.json [--only-suspected]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_diff import recorded_stdout  # noqa: E402

_GLOG = re.compile(r"^[EWIF]\d{4} \d\d:\d\d:\d\d")
_OUTPUT_MARKERS = re.compile(
    r"\b(proxy|proxies|fallback|fall(?:s|ing)? back|fell back|mock|synthetic|simulat\w*|"
    r"dummy|unavailable|inaccessible|No module named|could not (?:open|read|load)|"
    r"using .{0,40}\binstead\b)",
    re.IGNORECASE,
)
_CODE_SUBSTITUTE = re.compile(r"\b(proxy|proxies|simulat\w*|synthetic)\b", re.IGNORECASE)
_CODE_MOCK = re.compile(r"\b(mock\w*|dummy|fallback|fell back|fall back)\b", re.IGNORECASE)
_IMAGE_HYPOTHESIS = re.compile(
    r"\b(fluoresc\w*|intensit\w*|image|imagery|raw signal|brightness|contrast|SNR|"
    r"voxel\w*|tubular\w*|vesselness|sheetness|Hessian|OME-Zarr|zarr|patch(?:es)?)\b",
    re.IGNORECASE,
)
_IMAGE_READER = re.compile(r"\b(tensorstore|zarr\.open|s3fs|boto3|fsspec|imageio|tifffile)\b")
_IMAGE_READ_EVIDENCE = re.compile(
    r"(opened (?:successfully|tensorstore|ome-zarr|remote|zarr)|"
    r"shape=\(1, 1, \d|shape: \(1, 1, \d|"
    r"patch(?:es)? (?:read|extracted|sampled|fetched)|patch_reads_success=[1-9]|"
    r"total_patches_read=[1-9]|profiles extracted|labeled patches|"
    r"successfully extracted|\d[\d,]* (?:image )?patches\b|patches/s|probe latency)",
    re.IGNORECASE,
)
_MAX_LINES = 8
_LINE_CHARS = 160


def _lines(text: str, pattern: re.Pattern, *, skip_glog: bool = False) -> list[str]:
    found: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or (skip_glog and _GLOG.match(line)) or not pattern.search(line):
            continue
        short = line[:_LINE_CHARS]
        if short not in found:
            found.append(short)
        if len(found) >= _MAX_LINES:
            break
    return found


def audit_record(record: dict) -> dict:
    """The substitution marks of one recorded hypothesis (see the module docstring)."""
    hypothesis = str(record.get("hypothesis") or "")
    code = str(record.get("code") or "")
    output = recorded_stdout(str(record.get("codeOutput") or ""))
    image_hypothesis = bool(_IMAGE_HYPOTHESIS.search(hypothesis))
    reader = bool(_IMAGE_READER.search(code))
    evidence = bool(_IMAGE_READ_EVIDENCE.search(output))
    output_markers = _lines(output, _OUTPUT_MARKERS, skip_glog=True)
    code_substitutes = _lines(code, _CODE_SUBSTITUTE)
    code_mocks = _lines(code, _CODE_MOCK)
    flags = []
    if output_markers:
        flags.append("output-fallback-marker")
    if code_substitutes:
        flags.append("code-substitute-marker")
    if image_hypothesis and not reader:
        flags.append("image-hypothesis-without-reader")
    elif image_hypothesis and not evidence:
        flags.append("image-hypothesis-without-read-evidence")
    if code_mocks and not code_substitutes:
        flags.append("code-mock-marker")
    suspected = any(flag != "code-mock-marker" for flag in flags)
    return {
        "suspected": suspected,
        "flags": flags,
        "output_markers": output_markers,
        "code_substitutes": code_substitutes,
        "code_mocks": code_mocks,
        "image_hypothesis": image_hypothesis,
        "image_reader_in_code": reader,
        "image_read_evidence": evidence,
    }


def audit_run(records: list[dict]) -> dict:
    entries = {str(r.get("id")): audit_record(r) for r in records if isinstance(r, dict)}
    suspected = [rid for rid, entry in entries.items() if entry["suspected"]]
    return {
        "n": len(entries),
        "n_suspected": len(suspected),
        "suspected_ids": suspected,
        "entries": entries,
    }


def audit_text(audit: dict, file_rel: str | None = None) -> str:
    """The audit as the sentences an instruction carries."""
    where = f", written to {file_rel}" if file_rel else ""
    ids = ", ".join(audit["suspected_ids"]) or "none"
    return (
        "RECORDED-DATA AUDIT: the driver scanned every record's code and output for "
        "substituted data (a proxy, fallback, mock or simulated value in the output "
        "or code; an image hypothesis without an image reader or without evidence "
        f"that patches were read){where}. Flagged ids: {ids} "
        f"({audit['n_suspected']} of {audit['n']}). A flag is evidence, not a "
        "verdict: read the flagged record's output and code, and where the measured "
        "quantity is a stand-in for the hypothesized one (intensity set from the gap, "
        "coordinate scatter named a Hessian) treat the result as invalidly computed. "
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--only-suspected", action="store_true",
                        help="List only the flagged records' entries.")
    args = parser.parse_args(argv)
    records = json.loads(args.run_json.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = next((v for v in records.values() if isinstance(v, list)), [])
    audit = audit_run(records)
    audit["run_json"] = str(args.run_json)
    if args.only_suspected:
        audit["entries"] = {rid: audit["entries"][rid] for rid in audit["suspected_ids"]}
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"[recorded-audit] {audit['n_suspected']} of {audit['n']} flagged: "
          f"{', '.join(audit['suspected_ids']) or 'none'}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
