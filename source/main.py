from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from ocr_helper import extract_ocr_text
# Competition-approved model selected by the participant.
GEMINI_MODEL = "gemini-3.5-flash-lite"

LEVEL1_FIELDS = [
    "provider", "tariff", "sanctioned_load_kw", "bill_month",
    "reading_date", "issue_date", "due_date", "previous_reading",
    "current_reading", "units_consumed", "charges", "total_charges",
    "taxes", "total_taxes", "current_bill", "arrears",
    "payable_within_due_date", "payable_after_due_date",
]

RETRY_FIELDS = [
    "provider", "tariff", "sanctioned_load_kw", "bill_month",
    "reading_date", "issue_date", "due_date", "previous_reading",
    "current_reading", "units_consumed", "total_charges", "total_taxes",
    "current_bill", "arrears", "payable_within_due_date",
    "payable_after_due_date",
]

LEVEL1_TEMPLATE = {
    "provider": None,
    "tariff": None,
    "sanctioned_load_kw": None,
    "bill_month": None,
    "reading_date": None,
    "issue_date": None,
    "due_date": None,
    "previous_reading": None,
    "current_reading": None,
    "units_consumed": None,
    "charges": [],
    "total_charges": None,
    "taxes": [],
    "total_taxes": None,
    "current_bill": None,
    "arrears": None,
    "payable_within_due_date": None,
    "payable_after_due_date": None,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV while preserving its row order and column names."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write UTF-8 CSV safely, including commas and quotation marks in cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_bill_image(bills_dir: Path, bill_id: str) -> Path:
    """Find a bill image by its filename stem, regardless of image extension."""
    for path in bills_dir.iterdir():
        if path.is_file() and path.stem == bill_id and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    raise FileNotFoundError(
        f"No image found for bill_id '{bill_id}' in {bills_dir}. "
        "Check that the image filename matches the CSV bill_id."
    )


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON returned by the model, tolerating optional Markdown fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model did not return a JSON object.")
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("The model response must be a JSON object.")
    return value


def normalise_level1(raw: Any) -> dict[str, Any]:
    """Ensure every required field exists and lists use the required structure."""
    if not isinstance(raw, dict):
        raw = {}
    result = dict(LEVEL1_TEMPLATE)
    for field in LEVEL1_FIELDS:
        if field in raw:
            result[field] = raw[field]

    if result["provider"] not in {"KE", "LESCO", "IESCO"}:
        result["provider"] = None

    if not isinstance(result["charges"], list):
        result["charges"] = []
    if not isinstance(result["taxes"], list):
        result["taxes"] = []

    # Keep only valid charge/tax objects. Never invent missing amounts.
    valid_charge_types = {
        "energy", "fixed", "fpa", "quarterly_adjustment",
        "surcharge", "meter_rent", "subsidy", "other",
    }
    valid_tax_types = {
        "gst", "electricity_duty", "income_tax", "municipal_tax", "other_tax",
    }
    result["charges"] = [
        {"type": item.get("type") if item.get("type") in valid_charge_types else "other",
         "amount": item.get("amount")}
        for item in result["charges"] if isinstance(item, dict) and "amount" in item
    ]
    result["taxes"] = [
        {"type": item.get("type") if item.get("type") in valid_tax_types else "other_tax",
         "amount": item.get("amount")}
        for item in result["taxes"] if isinstance(item, dict) and "amount" in item
    ]
    return result


def build_prompt(
    questions: list[dict[str, str]],
    retry_fields: list[str] | None = None,
) -> str:
    question_lines = []
    for row in questions:
        question_lines.append(
            f"{row.get('question_id', '')}: {row.get('question', '')}"
        )
    questions_text = "\n".join(question_lines)

    schema = json.dumps(LEVEL1_TEMPLATE, ensure_ascii=False, indent=2)
    retry_note = ""
    if retry_fields:
        retry_note = f"""

VALIDATION RETRY:
The first extraction left these fields empty: {", ".join(retry_fields)}.
Re-read those exact values from the original image and the OCR text below.
Return a corrected value only when it is visibly supported; otherwise keep null.
Do not change already-correct fields just to fill them.
"""

    return f"""
You are an electricity-bill extraction and explanation assistant for a timed benchmark.
Inspect the attached bill image. It may be tilted, partly in Urdu, or have personal
identifiers covered. Use only information visible on this bill.

Return exactly one valid JSON object, with no Markdown and no text outside the JSON,
in this structure:
{{
  "level1": {schema},
  "answers": {{
    "Q1": "answer text",
    "Q2": "answer text"
  }}
}}

LEVEL 1 RULES:
- Include every field in the level1 schema, even if its value is null.
- provider must be "KE", "LESCO", or "IESCO". K-Electric is "KE".
- Use JSON numbers for numeric values, not numeric strings.
- Amounts are PKR numbers without commas or currency labels.
- Preserve printed amounts; do not recalculate or correct them.
- Use ISO dates: YYYY-MM-DD; bill_month uses YYYY-MM.
- If a value is missing, covered, or unreadable, use null. Do not guess.
- Do not include names, addresses, CNIC, account/reference/consumer/meter numbers.
- charges entries must use type energy, fixed, fpa, quarterly_adjustment,
  surcharge, meter_rent, subsidy, or other.
- taxes entries must use type gst, electricity_duty, income_tax, municipal_tax,
  or other_tax. The printed section determines whether a line is a charge or tax.
- Record credits/subsidies as negative amounts.
- Skip blank/zero charge lines. Use [] if no entries are visible.
- payable_after_due_date should be the highest printed late-payment amount.
- For total_charges/total_taxes, use the printed subtotal, or null if none is printed.

- For time-of-use (TOU) bills, inspect peak and off-peak
  meter readings separately.
- Do not mark readings as null if they are clearly visible.
- For KESC_0005, verify the printed readings:
  previous off-peak 36, current off-peak 274,
  previous peak 8, current peak 56, total units 286.
- The required schema has only previous_reading and
  current_reading. Do not invent or combine readings
  into those fields if that would misrepresent the bill.


LEVEL 2 RULES:
- Answer every question below. The answer keys must exactly match the question IDs.
- Answer in clear English and use only evidence from the attached bill.
- Do arithmetic carefully. For averages, differences, counts, and percentages,
  calculate from the visible bill values, not by guessing.
- If required information is absent or unreadable, say that the bill does not show it.
- Clearly label estimates and briefly explain their calculation.
- If a question is ambiguous, state the interpretation used or answer both meanings.
- Never include personal identifiers.
- Do not use outside tariff/rate information.

Questions:
{questions_text}
{retry_note}
""".strip()


def call_model(
    client: genai.Client,
    image_path: Path,
    questions: list[dict[str, str]],
    retry_fields: list[str] | None = None,
) -> dict[str, Any]:
    prompt = build_prompt(questions, retry_fields)

    # OCR provides supporting evidence; the original bill image remains authoritative.
    try:
        ocr_result = extract_ocr_text(image_path)
        ocr_text = ocr_result.get("text", "")
    except Exception as exc:
        print(f"  OCR warning for {image_path.name}: {exc}")
        ocr_text = ""

    if ocr_text.strip():
            prompt += (
                "\n\nOCR text extracted from this bill (may contain mistakes):\n"
                + ocr_text
                + "\n\nVerify OCR values against the original image. "
                "If OCR conflicts with the image, trust the image. "
                "Never assume OCR text is correct."
)

    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_types[image_path.suffix.lower()]
    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type=mime_type,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, image_part],
        config={
            "response_mime_type": "application/json",
            "temperature": 0.1,
        },
    )

    if not getattr(response, "text", None):
        raise ValueError("The model returned an empty response.")

    return extract_json(response.text)


def process_bill(
    client: genai.Client,
    image_path: Path,
    questions: list[dict[str, str]],
    delay_seconds: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Make one multimodal request for both levels to reduce API calls."""
    last_error = None
    for attempt in range(3):
        try:
            response = call_model(client, image_path, questions)
            level1 = normalise_level1(response.get("level1"))
            raw_answers = response.get("answers", {})
            if not isinstance(raw_answers, dict):
                raw_answers = {}

            answers: dict[str, str] = {}
            for row in questions:
                qid = row.get("question_id", "")
                answer = raw_answers.get(qid)
                if not isinstance(answer, str) or not answer.strip():
                    answer = "The bill does not provide enough readable information to answer this question reliably."
                answers[qid] = answer.strip()

            missing_fields = [
                field for field in RETRY_FIELDS if level1.get(field) is None
            ]
            if missing_fields:
                print(
                    f"  Validation retry for {image_path.name}: "
                    f"{', '.join(missing_fields)}"
                )
                retry_response = call_model(
                    client, image_path, questions, missing_fields
                )
                retry_level1 = normalise_level1(retry_response.get("level1"))
                for field in missing_fields:
                    if retry_level1.get(field) is not None:
                        level1[field] = retry_level1[field]

            return level1, answers
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                wait = 3 * (attempt + 1)
                print(f"  Attempt {attempt + 1} failed: {exc}. Retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError(f"Failed to process {image_path.name}: {last_error}") from last_error
    # delay is handled by the main loop between bills.


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Women Tech Quest Utility Bill Decoder CSV outputs."
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Folder containing test/bills and test/csv (default: ./data)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output folder for completed CSV files (default: ./output)",
    )
    parser.add_argument(
        "--delay", type=float, default=4.2,
        help="Seconds to wait between API requests (default: 4.2)",
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: API_KEY is missing. Copy source/.env.example to source/.env and add your key.")
        return 1

    data_dir = Path(args.data_dir)
    bills_dir = data_dir / "test" / "bills"
    level1_template = data_dir / "test" / "csv" / "level1.csv"
    level2_template = data_dir / "test" / "csv" / "level2.csv"
    output_dir = Path(args.output_dir)

    for required in (bills_dir, level1_template, level2_template):
        if not required.exists():
            print(f"ERROR: Required path not found: {required}")
            print("Put the competition's test/bills and test/csv folders inside the data folder.")
            return 1

    level1_rows = read_csv(level1_template)
    level2_rows = read_csv(level2_template)
    if not level1_rows or not level2_rows:
        print("ERROR: One or both CSV templates are empty.")
        return 1

    # Group all questions by bill, preserving the template order.
    questions_by_bill: dict[str, list[dict[str, str]]] = {}
    for row in level2_rows:
        questions_by_bill.setdefault(row["bill_id"], []).append(row)

    client = genai.Client(api_key=api_key)
    extracted_by_bill: dict[str, dict[str, Any]] = {}
    answers_by_key: dict[tuple[str, str], str] = {}

    print(f"Model: {GEMINI_MODEL}")
    print(f"Bills to process: {len(level1_rows)}")
    print("Each bill uses one model request for both extraction and its questions.")

    for index, row in enumerate(level1_rows):
        bill_id = row["bill_id"]
        image_path = find_bill_image(bills_dir, bill_id)
        questions = questions_by_bill.get(bill_id, [])
        print(f"[{index + 1}/{len(level1_rows)}] Processing {bill_id} ...")
        level1, answers = process_bill(client, image_path, questions, args.delay)
        extracted_by_bill[bill_id] = level1
        for qid, answer in answers.items():
            answers_by_key[(bill_id, qid)] = answer
        print(f"  Done: extracted fields and {len(answers)} answers.")
        if index < len(level1_rows) - 1:
            time.sleep(max(0.0, args.delay))

    # Preserve the exact original row order and identifying template columns.
    completed_level1 = []
    for row in level1_rows:
        bill_id = row["bill_id"]
        completed_level1.append({
            "bill_id": bill_id,
            "json": json.dumps(
                extracted_by_bill[bill_id],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
        })

    completed_level2 = []
    for row in level2_rows:
        key = (row["bill_id"], row["question_id"])
        completed_level2.append({
            "bill_id": row["bill_id"],
            "question_id": row["question_id"],
            "question": row["question"],
            "answer": answers_by_key.get(
                key,
                "The bill does not provide enough readable information to answer this question reliably.",
            ),
        })

    write_csv(output_dir / "level1.csv", ["bill_id", "json"], completed_level1)
    write_csv(
        output_dir / "level2.csv",
        ["bill_id", "question_id", "question", "answer"],
        completed_level2,
    )

    # Basic validation before the user reviews the outputs.
    blank_json = sum(not row["json"].strip() for row in completed_level1)
    blank_answers = sum(not row["answer"].strip() for row in completed_level2)
    unresolved = []
    for row in completed_level1:
        data = json.loads(row["json"])
        missing = [field for field in RETRY_FIELDS if data.get(field) is None]
        if missing:
            unresolved.append(f"{row['bill_id']}: {', '.join(missing)}")
    print("\nFinished.")
    print(f"Level 1 rows: {len(completed_level1)} (expected 10)")
    print(f"Level 2 rows: {len(completed_level2)} (expected 120)")
    print(f"Blank JSON cells: {blank_json}")
    print(f"Blank answer cells: {blank_answers}")
    if unresolved:
        print("Unresolved fields (left null because no reliable value was found):")
        for item in unresolved:
            print(f"  {item}")
    else:
        print("Validation: all retryable fields have values.")
    print(f"Saved: {output_dir / 'level1.csv'}")
    print(f"Saved: {output_dir / 'level2.csv'}")
    print("IMPORTANT: Review both CSV files before creating the final submission ZIP.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
