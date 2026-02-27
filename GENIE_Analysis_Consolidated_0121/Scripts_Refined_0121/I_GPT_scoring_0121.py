#!/usr/bin/env python3
"""
End-to-end pipeline (scoring medical relatedness):

Input:  CSV with concept-pair columns (defaults below)
Modes:  --mode sync  (concurrent real-time calls)
        --mode batch (OpenAI Batch API)

Output: CSV with relationship_score + has_medical_relationship
        + optional audit JSONL (full input/request/response per pair)

Expected CSV columns (defaults):
  pair_id (optional)
  a_cui, a_term, a_semantic_type, a_assertion_status (optional -> default "unknown")
  b_cui, b_term, b_semantic_type, b_assertion_status (optional -> default "unknown")

Example:
  pair_id,a_cui,a_term,a_semantic_type,a_assertion_status,b_cui,b_term,b_semantic_type,b_assertion_status
  1,C1261155,aspartate aminotransferase serum measurement,Laboratory Procedure,PRESENT,
    C0428327,alanine aminotransferase - blood measurement,Laboratory Procedure,PRESENT

Prereqs:
  pip install openai
  export OPENAI_API_KEY="sk-..."

Run (batch):
  python GPT_scoring.py --input input.csv --output output.csv --model gpt-4o-mini --mode batch --audit-jsonl audit.jsonl

Run (sync):
  python GPT_scoring.py --input input.csv --output output.csv --model gpt-4o-mini --mode sync --concurrency 20 --audit-jsonl audit.jsonl
"""

import argparse
import asyncio
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI, AsyncOpenAI
from openai import APIError, RateLimitError, APITimeoutError


# ----------------------------
# Scoring rubric + schema
# ----------------------------
SYSTEM_PROMPT = """You are acting as a cautious clinician-informaticist.
Task: assess whether two UMLS concepts have a meaningful medical/clinical relationship in real-world practice.

Return a numeric medical relationship score in [0, 100] and a boolean has_medical_relationship.

Score guidance (use your judgment; these are anchors, not hard rules):
- 0–9: essentially unrelated
- 10–29: weak / tangential (same broad setting only, rare co-mention)
- 30–59: moderate (plausible linkage, sometimes co-managed)
- 60–89: strong (common linkage in care pathway, same condition management, typical co-ordering)
- 90–100: near-equivalent / direct coupling (same concept/synonym; one directly delivers/implements the other)

Assertion status handling:
- present: treat as actively applicable.
- absent (negated/denied): do not treat as active; usually reduces linkage unless relationship concerns exclusion.
- possible/suspected/uncertain: treat as uncertain; typically downgrade.
- historical: not currently active; typically downgrade unless relationship is about history/risk.
- family_history: not present in patient; typically downgrade.
- unknown: do not assume presence.

If you are uncertain, provide a conservative score and lower confidence.
Keep rationale concise and clinician-style.
Return only JSON that matches the provided schema exactly.
"""

JSON_SCHEMA = {
    "name": "medical_relationship_scoring",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relationship_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100
            },
            "has_medical_relationship": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
            "linking_conditions": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5
            },
            "assertion_adjustment": {
                "type": "string",
                "enum": [
                    "not_applicable",
                    "downgraded_due_to_absence_or_negation",
                    "downgraded_due_to_uncertainty",
                    "conditional_on_presence",
                    "conditional_on_current_use",
                    "conditional_on_diagnosis_status",
                ],
            },
        },
        "required": [
            "relationship_score",
            "has_medical_relationship",
            "confidence",
            "rationale",
            "linking_conditions",
            "assertion_adjustment",
        ],
    },
}


# ----------------------------
# Data model
# ----------------------------
@dataclass
class PairRecord:
    pair_id: str
    a_cui: str
    a_term: str
    a_semantic_type: str
    a_assertion_status: str
    b_cui: str
    b_term: str
    b_semantic_type: str
    b_assertion_status: str
    source_row: Dict[str, Any]


def _strip(v: Any) -> str:
    return ("" if v is None else str(v)).strip()


def normalize_assertion_status(s: str) -> str:
    """
    Normalizes common assertion vocab variants into a small controlled set.

    Output vocab:
      present, absent, possible, conditional, historical, family_history, unknown
    """
    x = _strip(s).lower().replace("-", "_").replace(" ", "_")
    if not x:
        return "unknown"

    mapping = {
        "present": "present",
        "pos": "present",

        "absent": "absent",
        "negated": "absent",
        "denied": "absent",
        "no": "absent",

        "possible": "possible",
        "suspected": "possible",
        "maybe": "possible",
        "probable": "possible",
        "uncertain": "possible",

        "conditional": "conditional",
        "if": "conditional",

        "historical": "historical",
        "history": "historical",
        "past": "historical",

        "family_history": "family_history",
        "fh": "family_history",
        "familyhx": "family_history",
    }
    return mapping.get(x, "unknown")


def make_user_prompt(p: PairRecord) -> str:
    return (
        "Concept A:\n"
        f"- cui: {p.a_cui}\n"
        f"- term: {p.a_term}\n"
        f"- semantic_type: {p.a_semantic_type}\n"
        f"- assertion_status: {p.a_assertion_status}\n\n"
        "Concept B:\n"
        f"- cui: {p.b_cui}\n"
        f"- term: {p.b_term}\n"
        f"- semantic_type: {p.b_semantic_type}\n"
        f"- assertion_status: {p.b_assertion_status}\n\n"
        "Question: Provide (1) relationship_score in [0,100], (2) has_medical_relationship boolean, "
        "(3) confidence in [0,1], (4) short rationale, (5) up to 5 linking_conditions, "
        "and (6) assertion_adjustment reflecting any downgrades due to assertion status."
    )


def make_chat_body(model: str, user_prompt: str) -> Dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_schema", "json_schema": JSON_SCHEMA},
    }


def safe_json_loads(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return json.loads(s), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"


# ----------------------------
# CSV I/O
# ----------------------------
def read_pairs_from_csv(
    input_csv: str,
    id_col: str,
    a_cui_col: str, a_term_col: str, a_sem_col: str, a_assert_col: str,
    b_cui_col: str, b_term_col: str, b_sem_col: str, b_assert_col: str,
    assertion_optional: bool,
) -> List[PairRecord]:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise SystemExit("Input CSV has no header row.")

        required = [a_cui_col, a_term_col, a_sem_col, b_cui_col, b_term_col, b_sem_col]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise SystemExit(
                f"CSV missing required columns: {missing}\n"
                f"Found columns: {reader.fieldnames}\n"
                "Use --col-* arguments to map your columns."
            )

        if not assertion_optional:
            missing_assert = [c for c in [a_assert_col, b_assert_col] if c not in reader.fieldnames]
            if missing_assert:
                raise SystemExit(
                    f"CSV missing required assertion columns: {missing_assert}\n"
                    f"Found columns: {reader.fieldnames}\n"
                    "Either add the columns or use --assertion-optional to default to 'unknown'."
                )

        out: List[PairRecord] = []
        for i, row in enumerate(reader, start=1):
            pair_id = _strip(row.get(id_col)) or f"p{i:06d}"

            a_assert = normalize_assertion_status(row.get(a_assert_col, "unknown"))
            b_assert = normalize_assertion_status(row.get(b_assert_col, "unknown"))

            pr = PairRecord(
                pair_id=pair_id,
                a_cui=_strip(row.get(a_cui_col)),
                a_term=_strip(row.get(a_term_col)),
                a_semantic_type=_strip(row.get(a_sem_col)),
                a_assertion_status=a_assert,
                b_cui=_strip(row.get(b_cui_col)),
                b_term=_strip(row.get(b_term_col)),
                b_semantic_type=_strip(row.get(b_sem_col)),
                b_assertion_status=b_assert,
                source_row=row,
            )

            if not pr.a_cui or not pr.b_cui:
                raise SystemExit(f"Row {i} (pair_id={pair_id}) missing a_cui or b_cui.")
            out.append(pr)

        return out


def write_output_csv(output_csv: str, results: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "pair_id",
        "a_cui", "a_term", "a_semantic_type", "a_assertion_status",
        "b_cui", "b_term", "b_semantic_type", "b_assertion_status",

        "relationship_score",
        "has_medical_relationship",
        "confidence",
        "assertion_adjustment",
        "rationale",
        "linking_conditions",

        "status",
        "error_type", "error_message",
        "request_json", "response_json",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def append_audit_jsonl(audit_path: str, obj: Dict[str, Any]) -> None:
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------
# SYNC mode (concurrent)
# ----------------------------
async def call_one_sync(
    client: AsyncOpenAI,
    pr: PairRecord,
    model: str,
    max_retries: int,
    audit_jsonl: Optional[str],
) -> Dict[str, Any]:
    user_prompt = make_user_prompt(pr)
    body = make_chat_body(model, user_prompt)
    request_envelope = {"url": "/v1/chat/completions", "body": body}

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.chat.completions.create(**body)
            raw = resp.model_dump()
            content = (resp.choices[0].message.content or "")
            parsed, parse_err = safe_json_loads(content)

            row = {
                "pair_id": pr.pair_id,
                "a_cui": pr.a_cui, "a_term": pr.a_term, "a_semantic_type": pr.a_semantic_type,
                "a_assertion_status": pr.a_assertion_status,
                "b_cui": pr.b_cui, "b_term": pr.b_term, "b_semantic_type": pr.b_semantic_type,
                "b_assertion_status": pr.b_assertion_status,

                "relationship_score": (parsed or {}).get("relationship_score", ""),
                "has_medical_relationship": (parsed or {}).get("has_medical_relationship", ""),
                "confidence": (parsed or {}).get("confidence", ""),
                "assertion_adjustment": (parsed or {}).get("assertion_adjustment", ""),
                "rationale": (parsed or {}).get("rationale", ""),
                "linking_conditions": ";".join((parsed or {}).get("linking_conditions", []) or []),

                "status": "ok" if (parsed is not None and parse_err is None) else "bad_json",
                "error_type": "",
                "error_message": parse_err or "",
                "request_json": json.dumps(request_envelope, ensure_ascii=False),
                "response_json": json.dumps(raw, ensure_ascii=False),
            }

            if audit_jsonl:
                append_audit_jsonl(audit_jsonl, {
                    "pair_id": pr.pair_id,
                    "input": pr.source_row,
                    "request": request_envelope,
                    "response_raw": raw,
                    "assessment": parsed,
                    "assessment_parse_error": parse_err,
                })

            return row

        except (RateLimitError, APITimeoutError, APIError) as e:
            if attempt == max_retries:
                err = {"type": e.__class__.__name__, "message": str(e)}
                if audit_jsonl:
                    append_audit_jsonl(audit_jsonl, {
                        "pair_id": pr.pair_id,
                        "input": pr.source_row,
                        "request": request_envelope,
                        "error": err,
                    })
                return {
                    "pair_id": pr.pair_id,
                    "a_cui": pr.a_cui, "a_term": pr.a_term, "a_semantic_type": pr.a_semantic_type,
                    "a_assertion_status": pr.a_assertion_status,
                    "b_cui": pr.b_cui, "b_term": pr.b_term, "b_semantic_type": pr.b_semantic_type,
                    "b_assertion_status": pr.b_assertion_status,

                    "relationship_score": "", "has_medical_relationship": "", "confidence": "",
                    "assertion_adjustment": "", "rationale": "", "linking_conditions": "",

                    "status": "error",
                    "error_type": err["type"],
                    "error_message": err["message"],
                    "request_json": json.dumps(request_envelope, ensure_ascii=False),
                    "response_json": "",
                }
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 20.0)

        except Exception as e:
            err = {"type": e.__class__.__name__, "message": str(e)}
            if audit_jsonl:
                append_audit_jsonl(audit_jsonl, {
                    "pair_id": pr.pair_id,
                    "input": pr.source_row,
                    "request": request_envelope,
                    "error": err,
                })
            return {
                "pair_id": pr.pair_id,
                "a_cui": pr.a_cui, "a_term": pr.a_term, "a_semantic_type": pr.a_semantic_type,
                "a_assertion_status": pr.a_assertion_status,
                "b_cui": pr.b_cui, "b_term": pr.b_term, "b_semantic_type": pr.b_semantic_type,
                "b_assertion_status": pr.b_assertion_status,

                "relationship_score": "", "has_medical_relationship": "", "confidence": "",
                "assertion_adjustment": "", "rationale": "", "linking_conditions": "",

                "status": "error",
                "error_type": err["type"],
                "error_message": err["message"],
                "request_json": json.dumps(request_envelope, ensure_ascii=False),
                "response_json": "",
            }

    return {
        "pair_id": pr.pair_id,
        "status": "error",
        "error_type": "Unknown",
        "error_message": "unreachable",
        "request_json": json.dumps(request_envelope, ensure_ascii=False),
        "response_json": "",
    }


async def run_sync_mode(
    pairs: List[PairRecord],
    model: str,
    concurrency: int,
    max_retries: int,
    audit_jsonl: Optional[str],
) -> List[Dict[str, Any]]:
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    async def bounded(pr: PairRecord) -> Dict[str, Any]:
        async with sem:
            return await call_one_sync(client, pr, model, max_retries, audit_jsonl)

    tasks = [asyncio.create_task(bounded(pr)) for pr in pairs]
    results: List[Dict[str, Any]] = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)

    results.sort(key=lambda r: str(r.get("pair_id", "")))
    return results


# ----------------------------
# BATCH mode
# ----------------------------
def write_batch_input_jsonl(
    pairs: List[PairRecord],
    model: str,
    batchinput_path: str,
) -> Dict[str, PairRecord]:
    custom_id_to_pair: Dict[str, PairRecord] = {}
    with open(batchinput_path, "w", encoding="utf-8") as fout:
        for pr in pairs:
            custom_id = pr.pair_id
            if custom_id in custom_id_to_pair:
                raise SystemExit(f"Duplicate pair_id/custom_id: {custom_id}. Ensure pair_id is unique.")
            custom_id_to_pair[custom_id] = pr

            user_prompt = make_user_prompt(pr)
            body = make_chat_body(model, user_prompt)

            line_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            fout.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
    return custom_id_to_pair


def run_batch_mode(
    pairs: List[PairRecord],
    model: str,
    batchinput_path: str,
    poll_seconds: int,
    audit_jsonl: Optional[str],
) -> List[Dict[str, Any]]:
    client = OpenAI()

    custom_id_to_pair = write_batch_input_jsonl(pairs, model, batchinput_path)
    uploaded = client.files.create(file=open(batchinput_path, "rb"), purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"job": "medical_relationship_scoring"},
    )

    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch.id)
        print(
            f"Batch status={batch.status} "
            f"(completed={batch.request_counts.completed}, failed={batch.request_counts.failed})"
        )
        if batch.status in terminal:
            break
        time.sleep(poll_seconds)

    output_by_id: Dict[str, Dict[str, Any]] = {}
    error_by_id: Dict[str, Dict[str, Any]] = {}

    if batch.output_file_id:
        out_text = client.files.content(batch.output_file_id).text
        for line in out_text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            output_by_id[obj["custom_id"]] = obj

    if batch.error_file_id:
        err_text = client.files.content(batch.error_file_id).text
        for line in err_text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            error_by_id[obj["custom_id"]] = obj

    results: List[Dict[str, Any]] = []

    for custom_id, pr in custom_id_to_pair.items():
        user_prompt = make_user_prompt(pr)
        body = make_chat_body(model, user_prompt)
        request_envelope = {"url": "/v1/chat/completions", "body": body}

        base_row = {
            "pair_id": pr.pair_id,
            "a_cui": pr.a_cui, "a_term": pr.a_term, "a_semantic_type": pr.a_semantic_type,
            "a_assertion_status": pr.a_assertion_status,
            "b_cui": pr.b_cui, "b_term": pr.b_term, "b_semantic_type": pr.b_semantic_type,
            "b_assertion_status": pr.b_assertion_status,
            "request_json": json.dumps(request_envelope, ensure_ascii=False),
        }

        if custom_id in error_by_id:
            err = error_by_id[custom_id].get("error") or {}
            if audit_jsonl:
                append_audit_jsonl(audit_jsonl, {
                    "pair_id": pr.pair_id,
                    "batch_id": batch.id,
                    "input": pr.source_row,
                    "request": request_envelope,
                    "error": err,
                })
            results.append({
                **base_row,
                "relationship_score": "", "has_medical_relationship": "", "confidence": "",
                "assertion_adjustment": "", "rationale": "", "linking_conditions": "",
                "status": "error",
                "error_type": (err.get("type") or ""),
                "error_message": (err.get("message") or json.dumps(err, ensure_ascii=False)),
                "response_json": "",
            })
            continue

        line_obj = output_by_id.get(custom_id)
        if not line_obj:
            err = {"type": "MissingOutput", "message": "No output line found for this custom_id."}
            if audit_jsonl:
                append_audit_jsonl(audit_jsonl, {
                    "pair_id": pr.pair_id,
                    "batch_id": batch.id,
                    "input": pr.source_row,
                    "request": request_envelope,
                    "error": err,
                })
            results.append({
                **base_row,
                "relationship_score": "", "has_medical_relationship": "", "confidence": "",
                "assertion_adjustment": "", "rationale": "", "linking_conditions": "",
                "status": "missing",
                "error_type": err["type"],
                "error_message": err["message"],
                "response_json": "",
            })
            continue

        if line_obj.get("error"):
            err = line_obj["error"]
            if audit_jsonl:
                append_audit_jsonl(audit_jsonl, {
                    "pair_id": pr.pair_id,
                    "batch_id": batch.id,
                    "input": pr.source_row,
                    "request": request_envelope,
                    "error": err,
                })
            results.append({
                **base_row,
                "relationship_score": "", "has_medical_relationship": "", "confidence": "",
                "assertion_adjustment": "", "rationale": "", "linking_conditions": "",
                "status": "error",
                "error_type": (err.get("type") if isinstance(err, dict) else "BatchError"),
                "error_message": (err.get("message") if isinstance(err, dict) else str(err)),
                "response_json": "",
            })
            continue

        resp_body = line_obj["response"]["body"]
        content = resp_body["choices"][0]["message"]["content"]
        parsed, parse_err = safe_json_loads(content)

        if audit_jsonl:
            append_audit_jsonl(audit_jsonl, {
                "pair_id": pr.pair_id,
                "batch_id": batch.id,
                "input": pr.source_row,
                "request": request_envelope,
                "response_raw": resp_body,
                "assessment": parsed,
                "assessment_parse_error": parse_err,
            })

        results.append({
            **base_row,
            "relationship_score": (parsed or {}).get("relationship_score", ""),
            "has_medical_relationship": (parsed or {}).get("has_medical_relationship", ""),
            "confidence": (parsed or {}).get("confidence", ""),
            "assertion_adjustment": (parsed or {}).get("assertion_adjustment", ""),
            "rationale": (parsed or {}).get("rationale", ""),
            "linking_conditions": ";".join((parsed or {}).get("linking_conditions", []) or []),
            "status": "ok" if (parsed is not None and parse_err is None) else "bad_json",
            "error_type": "",
            "error_message": parse_err or "",
            "response_json": json.dumps(resp_body, ensure_ascii=False),
        })

    results.sort(key=lambda r: str(r.get("pair_id", "")))
    return results


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV file")
    ap.add_argument("--output", required=True, help="Output CSV file")
    ap.add_argument("--model", required=True, help="Model name")
    ap.add_argument("--mode", choices=["sync", "batch"], required=True)

    # Sync knobs
    ap.add_argument("--concurrency", type=int, default=10, help="Only for --mode sync")
    ap.add_argument("--max-retries", type=int, default=5, help="Only for --mode sync")

    # Batch knobs
    ap.add_argument("--batchinput", default="batchinput.jsonl", help="Only for --mode batch")
    ap.add_argument("--poll-seconds", type=int, default=10, help="Only for --mode batch")

    # Column mapping
    ap.add_argument("--id-col", default="pair_id")
    ap.add_argument("--a-cui-col", default="a_cui")
    ap.add_argument("--a-term-col", default="a_term")
    ap.add_argument("--a-semantic-type-col", default="a_semantic_type")
    ap.add_argument("--a-assertion-col", default="a_assertion_status")
    ap.add_argument("--b-cui-col", default="b_cui")
    ap.add_argument("--b-term-col", default="b_term")
    ap.add_argument("--b-semantic-type-col", default="b_semantic_type")
    ap.add_argument("--b-assertion-col", default="b_assertion_status")
    ap.add_argument(
        "--assertion-optional",
        action="store_true",
        help="If set, missing assertion columns default to 'unknown' instead of erroring.",
    )

    # Audit trail
    ap.add_argument("--audit-jsonl", default="", help="Optional path for full input/request/response audit JSONL")

    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")

    audit_path = args.audit_jsonl.strip() or None
    if audit_path:
        # Truncate existing audit
        with open(audit_path, "w", encoding="utf-8") as _:
            pass

    pairs = read_pairs_from_csv(
        input_csv=args.input,
        id_col=args.id_col,
        a_cui_col=args.a_cui_col,
        a_term_col=args.a_term_col,
        a_sem_col=args.a_semantic_type_col,
        a_assert_col=args.a_assertion_col,
        b_cui_col=args.b_cui_col,
        b_term_col=args.b_term_col,
        b_sem_col=args.b_semantic_type_col,
        b_assert_col=args.b_assertion_col,
        assertion_optional=args.assertion_optional,
    )

    if args.mode == "sync":
        results = asyncio.run(
            run_sync_mode(
                pairs=pairs,
                model=args.model,
                concurrency=max(1, args.concurrency),
                max_retries=max(1, args.max_retries),
                audit_jsonl=audit_path,
            )
        )
    else:
        results = run_batch_mode(
            pairs=pairs,
            model=args.model,
            batchinput_path=args.batchinput,
            poll_seconds=max(1, args.poll_seconds),
            audit_jsonl=audit_path,
        )

    write_output_csv(args.output, results)
    print(f"Wrote output CSV: {args.output}")
    if audit_path:
        print(f"Wrote audit JSONL: {audit_path}")


if __name__ == "__main__":
    main()
