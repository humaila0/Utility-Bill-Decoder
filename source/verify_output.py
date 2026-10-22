from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# This file is intended to live at project_root/source/verify_output.py.
try:
    from ocr_helper import extract_ocr_text
except ImportError:
    # Also support importing/running this file from a different working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ocr_helper import extract_ocr_text

GEMINI_MODEL = "gemini-3.1-flash-lite"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

LEVEL1_FIELDS = [
    "provider", "tariff", "sanctioned_load_kw", "bill_month",
    "reading_date", "issue_date", "due_date", "previous_reading",
    "current_reading", "units_consumed", "charges", "total_charges",
    "taxes", "total_taxes", "current_bill", "arrears",
    "payable_within_due_date", "payable_after_due_date",
]

NUMERIC_FIELDS = {
    "sanctioned_load_kw", "previous_reading", "current_reading",
    "units_consumed", "total_charges", "total_taxes", "current_bill",
    "arrears", "payable_within_due_date", "payable_after_due_date",
}
REQUIRED_REVIEW_FIELDS = [field for field in LEVEL1_FIELDS if field not in {"charges", "taxes"}]
DATE_FIELDS = {"reading_date", "issue_date", "due_date"}
PROVIDERS = {"KE", "LESCO", "IESCO"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_bill_image(bills_dir: Path, bill_id: str) -> Path | None:
    if not bills_dir.exists():
        return None
    for path in bills_dir.iterdir():
        if path.is_file() and path.stem == bill_id and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def parse_json_cell(cell: str) -> dict[str, Any]:
    try:
        obj = json.loads(cell or "{}")
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def numeric_value(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("PKR", "").replace("Rs.", "").replace("Rs", "").strip()
        if not cleaned:
            return None
        try:
            number = float(cleaned)
            if not math.isfinite(number):
                return None
            return int(number) if number.is_integer() else number
        except ValueError:
            return None
    return None


def normalise_candidate(field: str, value: Any) -> tuple[bool, Any, str]:
    """Return (valid, normalized value, explanation). Never invent a value."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return False, None, "candidate is empty/null"
    if field == "provider":
        text = str(value).strip().upper().replace("-", "").replace(" ", "")
        aliases = {"KE": "KE", "KELECTRIC": "KE", "KARACHIELECTRIC": "KE", "LESCO": "LESCO", "IESCO": "IESCO"}
        normalized = aliases.get(text)
        return (normalized in PROVIDERS, normalized, "provider must be KE, LESCO, or IESCO")
    if field in NUMERIC_FIELDS:
        number = numeric_value(value)
        if number is None:
            return False, None, "not a valid finite number"
        # Physical quantities should not be negative; arrears may be negative if printed as a credit.
        if field in {"sanctioned_load_kw", "previous_reading", "current_reading", "units_consumed", "total_charges", "total_taxes", "current_bill", "payable_within_due_date", "payable_after_due_date"} and number < 0:
            return False, None, "negative value is implausible for this field"
        return True, number, "valid numeric value"
    if field == "bill_month":
        text = str(value).strip()
        return (bool(re.fullmatch(r"\d{4}-\d{2}", text)), text, "expected YYYY-MM")
    if field in DATE_FIELDS:
        text = str(value).strip()
        return (bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text)), text, "expected YYYY-MM-DD")
    if field in {"tariff"}:
        return (isinstance(value, (str, int, float)) and bool(str(value).strip()), str(value).strip(), "tariff must be readable text")
    if field in {"charges", "taxes"}:
        return (isinstance(value, list), value if isinstance(value, list) else [], "expected a list")
    return True, value, "accepted"


def validate_list_field(field: str, value: Any) -> tuple[bool, str]:
    if not isinstance(value, list):
        return False, "expected list"
    for index, item in enumerate(value):
        if not isinstance(item, dict) or "amount" not in item or numeric_value(item.get("amount")) is None:
            return False, f"item {index} is not a valid object with numeric amount"
    return True, "valid list"


def field_issues(data: dict[str, Any]) -> dict[str, str]:
    issues: dict[str, str] = {}
    for field in REQUIRED_REVIEW_FIELDS:
        value = data.get(field)
        if blank(value):
            issues[field] = "missing value"
            continue
        valid, _, why = normalise_candidate(field, value)
        if not valid:
            issues[field] = f"invalid value: {why}"

    for field in ("charges", "taxes"):
        valid, why = validate_list_field(field, data.get(field, []))
        if not valid:
            issues[field] = why

    # These patterns are warnings, not proof that a value is wrong.
    if isinstance(data.get("arrears"), (int, float)) and data["arrears"] < 0:
        issues["arrears"] = "suspicious negative arrears/possible credit; verify against printed bill"
    if isinstance(data.get("payable_within_due_date"), (int, float)) and isinstance(data.get("payable_after_due_date"), (int, float)):
        if data["payable_after_due_date"] < data["payable_within_due_date"]:
            issues["payable_after_due_date"] = "after-due amount is lower than within-due amount; verify printed figures"
    if isinstance(data.get("previous_reading"), (int, float)) and isinstance(data.get("current_reading"), (int, float)):
        if data["current_reading"] < data["previous_reading"]:
            issues["current_reading"] = "current reading is lower than previous reading; possible TOU/register layout or extraction error"
    return issues


def extract_ocr_safely(image_path: Path) -> dict[str, Any]:
    try:
        result = extract_ocr_text(image_path)
        if not isinstance(result, dict):
            return {"text": "", "lines": [], "error": "OCR returned unexpected result"}
        return result
    except Exception as exc:
        return {"text": "", "lines": [], "error": f"{type(exc).__name__}: {exc}"}


def extract_response_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Gemini did not return a JSON object")
        result = json.loads(cleaned[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Gemini response is not a JSON object")
    return result


def targeted_gemini_review(client: Any, image_path: Path, fields: list[str], current: dict[str, Any], ocr_text: str) -> dict[str, Any]:
    from google.genai import types

    schema_values = {field: current.get(field) for field in fields}
    prompt = f"""You are verifying selected fields from one Pakistani electricity bill.
Inspect the original bill image first and use the OCR text only as supporting evidence.
Return ONLY a JSON object containing exactly these requested fields (no prose): {json.dumps(fields)}.
Current extracted values: {json.dumps(schema_values, ensure_ascii=False)}
OCR text (may contain recognition errors):
{ocr_text[:18000]}

Rules:
- Correct a field only when the original image visibly supports the value.
- If the value cannot be read confidently, return null for that field.
- Numeric fields must be JSON numbers, not strings. Do not include commas or currency labels.
- Dates must use YYYY-MM-DD; bill_month must use YYYY-MM.
- provider must be KE, LESCO, or IESCO (K-Electric is KE).
- Preserve printed amounts; do not calculate a replacement for a printed amount.
- A small negative arrears value may be a printed credit/rounding adjustment. Read what is actually printed; do not automatically turn it into zero.
- For a time-of-use bill, do not merge peak and off-peak readings into one reading. If a single schema field cannot represent the bill accurately, return null for that field.
- Do not return account numbers, meter numbers, names, addresses, CNICs, or other personal identifiers.
"""
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[image_path.suffix.lower()]
    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, image_part],
        config={"response_mime_type": "application/json", "temperature": 0.1},
    )
    if not getattr(response, "text", None):
        raise ValueError("Gemini returned an empty response")
    return extract_response_json(response.text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and conservatively correct existing WTQ utility-bill CSV outputs.")
    parser.add_argument("--data-dir", default="data", help="Project data folder (default: ./data)")
    parser.add_argument("--input-dir", default="output_ocr_final", help="Existing output folder to verify (default: ./output_ocr_final)")
    parser.add_argument("--output-dir", default="output_verified", help="Separate destination for verified outputs (default: ./output_verified)")
    parser.add_argument("--delay", type=float, default=4.2, help="Seconds between Gemini verification calls")
    parser.add_argument("--no-gemini", action="store_true", help="Run OCR + validation only; do not call Gemini")
    args = parser.parse_args()

    project_root = Path.cwd()
    data_dir = Path(args.data_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    source_dir = Path(__file__).resolve().parent
    load_dotenv(source_dir / ".env")

    # Resolve relative paths against the directory from which the command was run.
    if not input_dir.is_absolute(): input_dir = project_root / input_dir
    if not output_dir.is_absolute(): output_dir = project_root / output_dir
    if not data_dir.is_absolute(): data_dir = project_root / data_dir

    level1_path = input_dir / "level1.csv"
    level2_path = input_dir / "level2.csv"
    bills_dir = data_dir / "test" / "bills"
    for required in (level1_path, level2_path, bills_dir):
        if not required.exists():
            print(f"ERROR: Required path not found: {required}")
            print("Run this command from the project root and check --input-dir / --data-dir.")
            return 1

    level1_rows = read_csv(level1_path)
    level2_rows = read_csv(level2_path)
    if not level1_rows:
        print(f"ERROR: No rows found in {level1_path}")
        return 1
    if "bill_id" not in (level1_rows[0].keys() if level1_rows else []):
        print("ERROR: level1.csv must contain a bill_id column.")
        return 1
    if "json" not in (level1_rows[0].keys() if level1_rows else []):
        print("ERROR: level1.csv must contain a json column.")
        return 1

    client = None
    api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not args.no_gemini and api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
        except Exception as exc:
            print(f"WARNING: Could not initialise Gemini; continuing with OCR-only checks: {exc}")
    elif not args.no_gemini:
        print("WARNING: API_KEY / GEMINI_API_KEY not found in source/.env; running OCR-only checks.")

    output_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir = output_dir / "ocr_text"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    audit: list[dict[str, Any]] = []
    corrected_level1: list[dict[str, Any]] = []
    total_corrected = 0
    total_unresolved = 0

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Gemini targeted verification: {'ON' if client else 'OFF'}")
    print(f"Bills to audit: {len(level1_rows)}\n")

    for index, row in enumerate(level1_rows, start=1):
        bill_id = (row.get("bill_id") or "").strip()
        print(f"[{index}/{len(level1_rows)}] Checking {bill_id} ...")
        data = parse_json_cell(row.get("json", ""))
        # Keep the expected schema fields present without changing good values.
        for field in LEVEL1_FIELDS:
            if field not in data:
                data[field] = [] if field in {"charges", "taxes"} else None

        image_path = find_bill_image(bills_dir, bill_id)
        ocr_result: dict[str, Any] = {"text": "", "lines": [], "error": "image not found"}
        if image_path:
            ocr_result = extract_ocr_safely(image_path)
            ocr_payload = {
                "bill_id": bill_id,
                "image": image_path.name,
                "ocr_error": ocr_result.get("error"),
                "text": ocr_result.get("text", ""),
                "lines": ocr_result.get("lines", []),
            }
            (ocr_dir / f"{bill_id}.json").write_text(json.dumps(ocr_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            audit.append({"bill_id": bill_id, "field": "__image__", "original_value": "", "final_value": "", "status": "UNRESOLVED", "method": "file-check", "reason": f"No bill image found in {bills_dir}", "ocr_evidence": ""})

        ocr_text = str(ocr_result.get("text", "") or "")
        # OCR can directly confirm the provider only when its explicit label is present.
        provider_match = None
        if re.search(r"\bIESCO\b", ocr_text, re.IGNORECASE):
            provider_match = "IESCO"
        elif re.search(r"\bLESCO\b", ocr_text, re.IGNORECASE):
            provider_match = "LESCO"
        elif re.search(r"K\s*[- ]?\s*ELECTRIC|KESC|K\s*ELECTRIC", ocr_text, re.IGNORECASE):
            provider_match = "KE"
        if provider_match and data.get("provider") != provider_match:
            old = data.get("provider")
            data["provider"] = provider_match
            total_corrected += 1
            audit.append({"bill_id": bill_id, "field": "provider", "original_value": json.dumps(old, ensure_ascii=False), "final_value": json.dumps(provider_match), "status": "CORRECTED", "method": "explicit OCR label", "reason": "Provider label explicitly found in OCR text", "ocr_evidence": provider_match})

        issues = field_issues(data)
        flagged_fields = list(issues.keys())
        # Ask Gemini only for fields that fail checks or have a defined suspicious pattern.
        if client and image_path and flagged_fields:
            try:
                print(f"   Targeted Gemini review: {', '.join(flagged_fields)}")
                candidates = targeted_gemini_review(client, image_path, flagged_fields, data, ocr_text)
                for field in flagged_fields:
                    old = data.get(field)
                    if field not in candidates:
                        audit.append({"bill_id": bill_id, "field": field, "original_value": json.dumps(old, ensure_ascii=False), "final_value": json.dumps(old, ensure_ascii=False), "status": "UNRESOLVED", "method": "OCR + Gemini", "reason": "Gemini response omitted requested field; " + issues[field], "ocr_evidence": ocr_text[:1000]})
                        total_unresolved += 1
                        continue
                    valid, new_value, reason = normalise_candidate(field, candidates.get(field))
                    if field in {"charges", "taxes"}:
                        valid, reason = validate_list_field(field, candidates.get(field))
                        new_value = candidates.get(field) if valid else None
                    # Null is not a correction; preserve original unless it was already missing.
                    if not valid:
                        audit.append({"bill_id": bill_id, "field": field, "original_value": json.dumps(old, ensure_ascii=False), "final_value": json.dumps(old, ensure_ascii=False), "status": "UNRESOLVED", "method": "OCR + Gemini", "reason": f"No safe correction accepted ({reason}); original issue: {issues[field]}", "ocr_evidence": ocr_text[:1000]})
                        total_unresolved += 1
                        continue
                    # For suspicious-but-valid values, accept Gemini only if it returned a non-null value.
                    if new_value != old:
                        data[field] = new_value
                        total_corrected += 1
                        status = "CORRECTED"
                        reason_text = f"Gemini candidate accepted after type/schema validation; original issue: {issues[field]}"
                    else:
                        status = "REVIEWED_UNCHANGED"
                        reason_text = f"Gemini returned same value; original warning remains for caution: {issues[field]}"
                        if "missing value" in issues[field] or "invalid value" in issues[field]:
                            total_unresolved += 1
                    audit.append({"bill_id": bill_id, "field": field, "original_value": json.dumps(old, ensure_ascii=False), "final_value": json.dumps(data.get(field), ensure_ascii=False), "status": status, "method": "OCR + targeted Gemini", "reason": reason_text, "ocr_evidence": ocr_text[:1000]})
                time.sleep(max(0.0, args.delay))
            except Exception as exc:
                print(f"   WARNING: Gemini verification failed: {exc}")
                for field in flagged_fields:
                    audit.append({"bill_id": bill_id, "field": field, "original_value": json.dumps(data.get(field), ensure_ascii=False), "final_value": json.dumps(data.get(field), ensure_ascii=False), "status": "UNRESOLVED", "method": "OCR + Gemini error", "reason": f"{issues[field]}; Gemini error: {type(exc).__name__}: {exc}", "ocr_evidence": ocr_text[:1000]})
                    total_unresolved += 1
        else:
            for field in flagged_fields:
                # If a field was already corrected from explicit provider OCR, don't duplicate the record.
                if field == "provider" and provider_match and data.get(field) == provider_match:
                    continue
                audit.append({"bill_id": bill_id, "field": field, "original_value": json.dumps(data.get(field), ensure_ascii=False), "final_value": json.dumps(data.get(field), ensure_ascii=False), "status": "REVIEW_REQUIRED", "method": "OCR + rules only", "reason": issues[field] + ("; no image available" if not image_path else "; Gemini disabled/unavailable"), "ocr_evidence": ocr_text[:1000]})
                total_unresolved += 1

        corrected_level1.append({"bill_id": bill_id, "json": json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)})
        print(f"   Flagged {len(flagged_fields)} field(s); OCR characters: {len(ocr_text)}")

    # Preserve Level 2 exactly as received; only flag blank answers in the audit.
    for row in level2_rows:
        if blank(row.get("answer")):
            audit.append({"bill_id": row.get("bill_id", ""), "field": f"level2:{row.get('question_id', '')}", "original_value": "", "final_value": "", "status": "REVIEW_REQUIRED", "method": "blank-answer check", "reason": "Level 2 answer is blank; this verifier does not rewrite question answers automatically", "ocr_evidence": ""})
            total_unresolved += 1

    write_csv(output_dir / "level1.csv", ["bill_id", "json"], corrected_level1)
    # Keep the Level 2 CSV unchanged, including exact columns and row order.
    shutil.copy2(level2_path, output_dir / "level2.csv")
    audit_fields = ["bill_id", "field", "original_value", "final_value", "status", "method", "reason", "ocr_evidence"]
    write_csv(output_dir / "audit.csv", audit_fields, audit)

    print("\nVerification complete.")
    print(f"Level 1 bills: {len(corrected_level1)}")
    print(f"Level 2 answers copied unchanged: {len(level2_rows)}")
    print(f"Corrections accepted: {total_corrected}")
    print(f"Unresolved/review items: {total_unresolved}")
    print(f"Corrected CSVs: {output_dir / 'level1.csv'} and {output_dir / 'level2.csv'}")
    print(f"Audit report: {output_dir / 'audit.csv'}")
    print(f"OCR text files: {ocr_dir}")
    print("IMPORTANT: Check audit.csv. A correction is only as reliable as the image evidence; unresolved fields are not guessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
