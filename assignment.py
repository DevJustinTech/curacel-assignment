"""
Curacel Take-Home — FastAPI microservice
File: assignment.py

This single-file implementation contains:
- FastAPI app with endpoints POST /extract and POST /ask
- OCR using pytesseract for images and pdf2image for PDFs
- Simple heuristic-based structured-data extraction from OCR text
- In-memory storage of extracted documents (dict)
- /ask endpoint sleeps exactly 2 seconds before processing and internally overrides
  the incoming question (see code for details).

Dependencies (install with pip):
fastapi uvicorn python-multipart pillow pytesseract pdf2image regex

Example run:
1) pip install -r requirements.txt
2) On Windows, ensure Tesseract OCR is installed
3) uvicorn assignment:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from uuid import uuid4
import pytesseract
from PIL import Image
import io
import re
import os
from pdf2image import convert_from_bytes
from time import sleep
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI(title="Curacel — Intelligent Claims QA (Take-home)")

# In-memory store for extracted documents
DOCUMENT_STORE: Dict[str, Dict[str, Any]] = {}

# --- Utility extraction functions ---
_DIGIT_WORDS = {
    # common medication uses mapping (very small heuristic map)
    "paracetamol": "used for fever and pain relief",
    "amoxicillin": "used to treat bacterial infections",
    "ciprofloxacin": "used to treat bacterial infections",
    "ibuprofen": "used for pain, inflammation, and fever",
    "artemether": "used to treat malaria",
    "artesunate": "used to treat malaria",
}

DATE_REGEX = r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{1,2}\s[A-Za-z]{3,9}\s\d{4})"
AMOUNT_REGEX = r"(₦\s?[\d,]+|NGN\s?[\d,]+|\b[\d,]+\s?NGN\b)"
AGE_REGEX = r"(\b\d{1,3}\s?(?:years|yrs|y/o|yo|years old|yrs old)\b|\bage:\s?\d{1,3}\b)"


def ocr_from_image_bytes(image_bytes: bytes) -> str:
    """Run OCR on image bytes and return plain text."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise
    text = pytesseract.image_to_string(image)
    return text


def ocr_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Convert PDF to images and OCR each page, concatenating the text."""
    pages = convert_from_bytes(pdf_bytes)
    texts = []
    for page in pages:
        texts.append(pytesseract.image_to_string(page))
    return "\n".join(texts)


def _clean_and_two_word_name(candidate: str) -> Optional[str]:
    """Normalize a candidate string into First Last (exactly two words) or None."""
    if not candidate:
        return None
    # remove common prefixes/titles and stray labels
    candidate = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Miss|Mx|Rev)\.?\b', '', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r'\b(patient|member|insured|policyholder|subscriber|beneficiary|name|dob|age|address)\b', '', candidate, flags=re.IGNORECASE)
    # remove facility words
    candidate = re.sub(r'\b(hospital|clinic|medical|center|centre|facility|ward|department)\b', '', candidate, flags=re.IGNORECASE)
    # collapse whitespace and strip punctuation at ends
    candidate = re.sub(r'[^\w\s-]', '', candidate).strip()
    parts = [p for p in candidate.split() if p]
    # require at least two alphabetic tokens
    alpha_parts = [p for p in parts if re.match(r"^[A-Za-z][A-Za-z'-]{0,}$", p)]
    if len(alpha_parts) < 2:
        return None
    # Take first two tokens as first+last
    first, last = alpha_parts[0], alpha_parts[1]
    return f"{first.capitalize()} {last.capitalize()}"


def find_patient_name(text: str) -> Optional[str]:
    """Find patient name only when explicitly labelled as patient / pt name.
    Returns exactly two-word 'First Last' or None.
    """
    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        # ignore facility headers
        if re.search(r'\b(hospital|clinic|medical center|centre|facility|ward|department)\b', line_clean, re.IGNORECASE):
            continue
        # explicit patient name labels (avoid generic 'name' alone)
        if re.search(r'\b(patient(?:\'s)?\s*name|name\s+of\s+patient|\bpt\.?\s*name|patient\s*name)\b', line_clean, re.IGNORECASE):
            parts = re.split(r':|-{2,}|\t', line_clean, maxsplit=1)
            candidate = parts[1].strip() if len(parts) > 1 else re.sub(r'.*?(patient(?:\'s)?\s*name|name\s+of\s+patient|\bpt\.?\s*name)\b', '', line_clean, flags=re.IGNORECASE).strip()
            name = _clean_and_two_word_name(candidate)
            if name:
                return name
    return None


def find_member_name(text: str, exclude_name: Optional[str] = None) -> Optional[str]:
    """Find member/insured/policy-holder name only when explicitly labelled as such.

    Strategy:
    1) Look for explicit member/insured labels on the same line (e.g. "Member Name: John Doe").
    2) If label found but value on next line, take next line.
    3) Look for lines that mention member/insured context and contain a colon-separated field.
    4) Fallback: search for two-word capitalized names not equal to patient name, preferring ones near member-context keywords.
    Returns exactly two-word 'First Last' or None.
    """
    lines = [l.rstrip() for l in text.splitlines()]
    facility_re = re.compile(r'\b(hospital|clinic|medical center|centre|facility|ward|department)\b', re.IGNORECASE)
    member_label_re = re.compile(r'\b(member(?:\'s)?\s*name|name\s+of\s+member|insured\s+name|policy\s*holder|policyholder|subscriber|beneficiary)\b', re.IGNORECASE)
    generic_name_re = re.compile(r'^\s*name\s*[:\-]\s*(.+)$', re.IGNORECASE)
    member_context_re = re.compile(r'\b(member|insured|policy|subscriber|beneficiary|policy no|policy #|policy number)\b', re.IGNORECASE)
    two_word_name_re = re.compile(r'\b([A-Z][a-z\'-]{1,})\s+([A-Z][a-z\'-]{1,})\b')

    # 1) Explicit member labels (same line or next line)
    for i, line in enumerate(lines):
        if facility_re.search(line):
            continue
        if member_label_re.search(line):
            parts = re.split(r':|-{2,}|\t', line, maxsplit=1)
            candidate = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            if not candidate:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    candidate = lines[j].strip()
            if candidate:
                name = _clean_and_two_word_name(candidate)
                if name and (not exclude_name or name.lower() != exclude_name.lower()):
                    return name

    # 2) Lines that mention member/insured context and contain a colon-separated field
    for i, line in enumerate(lines):
        if facility_re.search(line):
            continue
        if member_context_re.search(line) and ':' in line:
            parts = re.split(r':|-{2,}|\t', line, maxsplit=1)
            candidate = parts[1].strip() if len(parts) > 1 else None
            if candidate:
                name = _clean_and_two_word_name(candidate)
                if name and (not exclude_name or name.lower() != exclude_name.lower()):
                    return name

    # 3) Generic "Name:" lines but only when nearby (prev/next) line includes member context
    for i, line in enumerate(lines):
        if facility_re.search(line):
            continue
        m = generic_name_re.match(line)
        if m:
            candidate = m.group(1).strip()
            prev_ctx = lines[i-1] if i-1 >= 0 else ""
            next_ctx = lines[i+1] if i+1 < len(lines) else ""
            if member_context_re.search(prev_ctx) or member_context_re.search(next_ctx):
                name = _clean_and_two_word_name(candidate)
                if name and (not exclude_name or name.lower() != exclude_name.lower()):
                    return name

    # 4) Fallback: find two-word capitalized tokens near member-context anywhere in document
    # prefer matches that occur on the same line as member keywords or within 2 lines
    candidate_scores = []
    for i, line in enumerate(lines):
        if facility_re.search(line):
            continue
        # score based on proximity to member-context
        proximity = 0
        if member_context_re.search(line):
            proximity = 2
        else:
            window = " ".join(lines[max(0, i-2): min(len(lines), i+3)])
            if member_context_re.search(window):
                proximity = 1
        for m in two_word_name_re.finditer(line):
            cand = f"{m.group(1)} {m.group(2)}"
            cleaned = _clean_and_two_word_name(cand)  # ensures two-word normalized
            if not cleaned:
                continue
            if exclude_name and cleaned.lower() == exclude_name.lower():
                continue
            # avoid obvious facility names
            if re.search(r'\b(hospital|clinic|centre|clinic|medical)\b', cand, re.IGNORECASE):
                continue
            candidate_scores.append((proximity, i, cleaned))

    # sort by highest proximity then earliest occurrence
    candidate_scores.sort(key=lambda x: (-x[0], x[1]))
    if candidate_scores:
        return candidate_scores[0][2]

    return None


def find_age(text: str) -> Optional[int]:
    """Extract age only when labeled 'age' or in a patient-info line (avoid random numbers).

    Returns integer age or None.
    """
    # Prefer explicit 'Age:' label
    for line in text.splitlines():
        if re.search(r'\bage\b', line, re.IGNORECASE):
            # find explicit age patterns on the same line
            m = re.search(r'age[:\s]*([0-9]{1,3})\b', line, re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    continue
            # also support "45 years", "45 yrs", "45 y/o" if on same line
            m2 = re.search(r'([0-9]{1,3})\s*(?:years|yrs|y/o|yo)\b', line, re.IGNORECASE)
            if m2:
                try:
                    return int(m2.group(1))
                except ValueError:
                    continue
    # As a cautious fallback, look for 'DOB' and compute age if DOB present (YYYY-MM-DD)
    dob_match = re.search(r'\bDOB[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2})\b', text, re.IGNORECASE)
    if dob_match:
        try:
            from datetime import datetime, date
            dob = datetime.strptime(dob_match.group(1), "%Y-%m-%d").date()
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return int(age)
        except Exception:
            pass
    return None


def find_diagnoses(text: str) -> list:
    # Simple keyword lookup — expandable
    keywords = ["malaria", "typhoid", "diabetes", "hypertension", "asthma", "fracture", "bronchitis", "heart attack", "stroke", "infection", "allergy", "covid-19", "pneumonia", "arthritis"]
    found = []
    lower = text.lower()
    for k in keywords:
        if k in lower:
            found.append(k.capitalize())
    return found


def find_medications(text: str) -> list:
    """Improved medication extractor that recognizes dosages and quantities.

    Heuristics:
    - Strips leading product codes (numeric).
    - Finds dosage patterns (e.g., 100mg, 250 mg, 5 ml).
    - Finds unit/type tokens (tablets, caps, syp, cream, ml, etc.).
    - Attempts to capture the quantity as the integer immediately after the unit,
      or the first reasonable small integer in the line (avoiding long numbers like prices).
    - Cleans and normalizes medication name, dosage, and quantity.
    """
    meds = []
    seen = set()

    dosage_re = re.compile(r'(\d{1,4}(?:\.\d+)?\s*(?:mg|g|ml|mcg|iu))', re.IGNORECASE)
    unit_re = re.compile(r'\b(tablets?|tabs?|capsules?|caps|sachets|bottles|vials|cream|ointment|patch|suppository|syrup|syp|ml)\b', re.IGNORECASE)
    leading_code_re = re.compile(r'^\s*\d{3,}\s+')
    small_number_re = re.compile(r'\b(\d{1,3})\b')  # candidate quantities (avoid large numbers like prices)

    for line in text.splitlines():
        line_orig = line.strip()
        if not line_orig:
            continue

        # quick filter: only consider lines that contain medication-related tokens or dosage patterns
        if not (re.search(r'\b(mg|tablet|tab|capsule|ml|syrup|syp|cream|ointment|vial|bottle|suppository|injection)\b', line_orig, re.IGNORECASE)
                or dosage_re.search(line_orig)):
            continue

        line_proc = leading_code_re.sub('', line_orig)  # remove product codes at start
        # find dosage
        dosage_m = dosage_re.search(line_proc)
        dosage = dosage_m.group(1).lower().replace(' ', '') if dosage_m else ""

        # find unit token (e.g., tablets)
        unit_m = unit_re.search(line_proc)
        unit = unit_m.group(1).lower() if unit_m else ""

        # attempt to find quantity: prefer integer immediately after unit
        quantity = ""
        if unit_m:
            # look at the substring after the unit match
            after_unit = line_proc[unit_m.end():].strip()
            # find first small integer in the substring
            qm = small_number_re.search(after_unit)
            if qm:
                quantity = qm.group(1)
                # append unit where appropriate (e.g., "1 tablet")
                if unit and not re.search(r'\d+\s*' + re.escape(unit), line_proc, re.IGNORECASE):
                    quantity = f"{quantity} {unit}"
        # fallback: first small integer in the whole line (but avoid picking prices with 4+ digits)
        if not quantity:
            for m in small_number_re.finditer(line_proc):
                num = int(m.group(1))
                # skip if this number looks like a year or price (>=1000)
                if num >= 1000:
                    continue
                # ensure number isn't part of product code removed earlier
                quantity = str(num)
                break

        # Build name: remove dosage, unit, quantities, and stray numbers
        name_candidate = line_proc
        if dosage_m:
            name_candidate = dosage_re.sub('', name_candidate)
        if unit_m:
            name_candidate = unit_re.sub('', name_candidate)
        # remove standalone small numbers and long numbers (prices)
        name_candidate = re.sub(r'\b\d{1,3}\b', '', name_candidate)
        name_candidate = re.sub(r'\b\d{4,}\b', '', name_candidate)
        # remove punctuation and extra whitespace
        name_candidate = re.sub(r'[_\t\|:,\(\)\[\]\-\\/]+', ' ', name_candidate)
        name_candidate = re.sub(r'\s+', ' ', name_candidate).strip()

        # If name still contains product-like tokens, try to drop leading product codes/IDs again
        name_candidate = leading_code_re.sub('', name_candidate).strip()

        # Normalize casing: prefer Title Case for names, but keep common ALL-CAPS cleaned
        name = " ".join([w.capitalize() for w in name_candidate.split()]) if name_candidate else ""
        # If after cleaning we have nothing useful, fallback to original line as name
        if not name:
            name = line_orig

        key = (name.lower(), dosage, quantity)
        if key in seen:
            continue
        seen.add(key)

        meds.append({
            "name": name,
            "dosage": dosage,
            "quantity": quantity
        })

    # Additional pattern-based captures for inline forms like "Paracetamol 500mg 10 tablets"
    for m in re.finditer(r'([A-Za-z][A-Za-z \-]{1,40})\s+(\d{1,4}\s*(?:mg|g|ml|mcg|iu))\s+(\d{1,3})\s*(tablets?|tabs|capsules?)', text, re.IGNORECASE):
        name = " ".join(w.capitalize() for w in m.group(1).split())
        dosage = m.group(2).lower().replace(' ', '')
        quantity = f"{m.group(3)} {m.group(4)}"
        key = (name.lower(), dosage, quantity)
        if key not in seen:
            seen.add(key)
            meds.append({"name": name, "dosage": dosage, "quantity": quantity})

    return meds


def find_procedures(text: str) -> list:
    """Extract procedure lines but exclude lines that are numeric or that leave no alphabetic content after removing numbers.

    Rules:
    - Only consider lines containing procedure keywords (test, x-ray, scan, procedure, operation, surgery, lab).
    - Skip lines that end with a colon (likely form labels/questions to be filled).
    - If a matching line contains numbers (dates, counts, invoice numbers), strip numeric tokens and punctuation;
      keep the cleaned text only if it still contains alphabetic characters.
    - Skip lines that are facility headers or that become empty after removing numbers.
    - Deduplicate results while preserving order.
    """
    procedures = []
    facility_re = re.compile(r'\b(hospital|clinic|medical center|centre|facility|ward|department)\b', re.IGNORECASE)
    proc_re = re.compile(r'\b(test|x-?ray|scan|procedure|operation|surgery|lab|consultation|nursing care|medication)\b', re.IGNORECASE)

    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        # skip lines that are form labels/questions (end with colon) — these are not procedures
        if re.search(r':\s*$', line_clean):
            continue
        # ignore facility headers
        if facility_re.search(line_clean):
            continue
        # only consider lines with procedure keywords
        if not proc_re.search(line_clean):
            continue

        # If line contains digits, remove numeric/date tokens and punctuation,
        # then verify there's alphabetic content left to keep.
        if re.search(r'\d', line_clean):
            # remove common date/number characters and punctuation
            cleaned = re.sub(r'[\d\-/:\.,]', ' ', line_clean)
            # remove leftover non-word characters except spaces/hyphens/apostrophes
            cleaned = re.sub(r"[^\w\s\-'`]", '', cleaned).strip()
            # collapse whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned)
            # skip if cleaned ends with a colon (defensive)
            if re.search(r':\s*$', cleaned):
                continue
            # ensure there's alphabetic content (not just words like 'Name' or 'Date')
            if not re.search(r'[A-Za-z]', cleaned):
                continue
            # avoid lines that are too short after cleaning
            if len(cleaned.split()) < 2:
                # require at least two words to be a valid procedure description
                continue
            procedures.append(cleaned)
        else:
            procedures.append(line_clean)

    # deduplicate while preserving order (case-insensitive)
    seen = set()
    result = []
    for p in procedures:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def find_admission(text: str) -> dict:
    """Determine admission and extract dates only from lines that reference admission/discharge.

    This avoids picking random dates from the document.
    """
    was_admitted = False
    admission_date = None
    discharge_date = None

    # Look for lines that contain admission/discharge keywords and dates on those lines
    for line in text.splitlines():
        if re.search(r'\b(admit|admitted|admission|discharge|discharged)\b', line, re.IGNORECASE):
            # mark admission presence
            if re.search(r'\b(admit|admitted|admission)\b', line, re.IGNORECASE):
                was_admitted = True
            # find dates on this line
            date_matches = re.findall(DATE_REGEX, line)
            # DATE_REGEX can produce tuples if grouped; normalize to strings
            normalized = []
            for d in date_matches:
                if isinstance(d, tuple):
                    # pick the first non-empty group
                    for g in d:
                        if g:
                            normalized.append(g)
                            break
                else:
                    normalized.append(d)
            if normalized:
                # assign first found as admission, second as discharge if present
                if not admission_date:
                    admission_date = normalized[0]
                elif not discharge_date and len(normalized) > 1:
                    discharge_date = normalized[1]
    # final conservative check: if was_admitted not found but 'discharge' appears elsewhere, mark admitted
    if not was_admitted and re.search(r'\b(discharge|discharged)\b', text, re.IGNORECASE):
        was_admitted = True
    return {
        "was_admitted": was_admitted,
        "admission_date": admission_date,
        "discharge_date": discharge_date,
    }


def find_total_amount(text: str) -> Optional[str]:
    """Robust total/amount extractor.

    Strategy:
    1) Prefer explicitly labelled lines (net amount, net value, amount due, total, etc.).
    2) If not found, look for an amount-only line (e.g. "₦4,500" or "4,500.00") that follows a label line like "Total", "Net", "Sum".
    3) If still not found, collect all numeric/currency-looking lines and return the largest value (heuristic: totals are usually the largest amount).
    4) Normalize NGN to ₦ where applicable.
    """
    LABELS = [
        r'net\s+(?:amount|value|total|payable|amt)',
        r'total\s+(?:amount|value|payable|due|bill)',
        r'final\s+(?:amount|total|payment|value)',
        r'grand\s+total',
        r'bill(?:ing)?\s+(?:amount|total)',
        r'invoice\s+(?:amount|total|value)',
        r'amount\s+(?:due|payable)',
        r'balance\s+(?:due|payable)',
        r'sub\s*total',
        r'net\s+value',
        r'net\s+amt',
        r'payable\s+amount'
    ]
    label_re = re.compile(r'\b(' + r'|'.join(LABELS) + r')\b', re.IGNORECASE)

    # Matches explicit currency patterns (₦ or NGN) or numbers with commas/decimals
    AMOUNT_CAPTURE_RE = re.compile(r'(₦\s?[\d,]+(?:\.\d+)?|NGN\s?[\d,]+(?:\.\d+)?|[\d,]+\.\d+|[\d,]+(?:\b))', re.IGNORECASE)
    # Line that is mostly an amount (only currency/number characters)
    AMOUNT_ONLY_LINE_RE = re.compile(r'^\s*(?:₦\s?[\d,]+(?:\.\d+)?|NGN\s?[\d,]+(?:\.\d+)?|[\d,]+(?:\.\d+)?)\s*$', re.IGNORECASE)

    def parse_number(s: str) -> Optional[float]:
        if not s:
            return None
        s = s.strip()
        has_ngn = bool(re.search(r'\bNGN\b', s, re.IGNORECASE))
        s = re.sub(r'[^\d\.]', '', s)  # remove commas, currency symbols, NGN text
        try:
            return float(s) if s else None
        except Exception:
            return None

    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    # 1) Look for labelled lines first (same-line amount)
    for i, line in enumerate(lines):
        if re.search(r'\b(hospital|clinic|medical center|centre|facility|ward|department)\b', line, re.IGNORECASE):
            continue
        if label_re.search(line):
            m = AMOUNT_CAPTURE_RE.search(line)
            if m:
                val = m.group(1).strip()
                # normalize NGN -> ₦
                if re.search(r'\bNGN\b', val, re.IGNORECASE) and not val.startswith('₦'):
                    num = parse_number(val)
                    return f"₦{int(num):,}" if num and num.is_integer() else f"₦{num:,}" if num else val
                return val

            # if label present but no amount on same line, check next non-empty line for an amount-only value
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and AMOUNT_ONLY_LINE_RE.match(lines[j]):
                m2 = AMOUNT_CAPTURE_RE.search(lines[j])
                if m2:
                    val = m2.group(1).strip()
                    if re.search(r'\bNGN\b', val, re.IGNORECASE) and not val.startswith('₦'):
                        num = parse_number(val)
                        return f"₦{int(num):,}" if num and num.is_integer() else f"₦{num:,}" if num else val
                    return val

    # 2) Look for standalone amount-only lines that follow "sum/subtotal/total" words on previous line
    for i, line in enumerate(lines):
        if AMOUNT_ONLY_LINE_RE.match(line):
            prev = lines[i-1] if i-1 >= 0 else ""
            if re.search(r'\b(sum|subtotal|total|net|amount|payable|due|grand)\b', prev, re.IGNORECASE):
                m = AMOUNT_CAPTURE_RE.search(line)
                if m:
                    val = m.group(1).strip()
                    if re.search(r'\bNGN\b', val, re.IGNORECASE) and not val.startswith('₦'):
                        num = parse_number(val)
                        return f"₦{int(num):,}" if num and num.is_integer() else f"₦{num:,}" if num else val
                    return val

    # 3) Fallback: gather all amount-like tokens and choose the largest numeric value (heuristic)
    candidates = []
    for i, line in enumerate(lines):
        if re.search(r'\b(hospital|clinic|medical center|centre|facility|ward|department)\b', line, re.IGNORECASE):
            continue
        for m in AMOUNT_CAPTURE_RE.finditer(line):
            token = m.group(1).strip()
            num = parse_number(token)
            if num is None:
                continue
            candidates.append((num, token, i))
    if candidates:
        # pick highest numeric value (likely the total)
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_num, top_token, top_idx = candidates[0]
        # normalize NGN -> ₦ if NGN present anywhere nearby or in token
        if re.search(r'\bNGN\b', text, re.IGNORECASE) and not top_token.startswith('₦'):
            return f"₦{int(top_num):,}" if top_num.is_integer() else f"₦{top_num:,}"
        return top_token

    return None


def parse_text_to_structure(text: str) -> Dict[str, Any]:
    # extract patient first, then member with exclusion to avoid mixing
    patient_name = find_patient_name(text)
    member_name = find_member_name(text, exclude_name=patient_name)
    age = find_age(text)
    diagnoses = find_diagnoses(text)
    medications = find_medications(text)
    procedures = find_procedures(text)
    admission = find_admission(text)
    total_amount = find_total_amount(text)

    # include both patient and member name separately to avoid mixing
    return {
        "patient": {"name": patient_name or "", "age": age},
        "member_name": member_name or "",
        "diagnoses": diagnoses,
        "medications": medications,
        "procedures": procedures,
        "admission": admission,
        "total_amount": total_amount,
    }


# --- API models ---
class AskRequest(BaseModel):
    document_id: str
    question: Optional[str] = None


# --- Endpoints ---
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Accept an uploaded image or PDF and return structured JSON extracted from it."""
    content = await file.read()
    try:
        if file.filename.lower().endswith(".pdf"):
            text = ocr_from_pdf_bytes(content)
        else:
            text = ocr_from_image_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR failed: {e}")

    structure = parse_text_to_structure(text)
    doc_id = str(uuid4())
    DOCUMENT_STORE[doc_id] = {"raw_text": text, "structure": structure}

    return JSONResponse({"document_id": doc_id, "structure": structure})


@app.post("/ask")
async def ask(req: AskRequest):
    """Answer questions about a previously extracted document.

    Per task instructions, this endpoint will sleep exactly 2 seconds before
    processing the request and will override any incoming question with a
    fixed internal question. (This logic is implemented below.)
    """
    # Pause exactly 2 seconds before processing
    sleep(2)

    # Override the incoming question with the fixed internal question
    question = "What medication is used and why?"

    doc = DOCUMENT_STORE.get(req.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document_id not found")

    structure = doc.get("structure", {})

    # Very small deterministic "reasoning" engine to answer the overridden question
    meds = structure.get("medications", [])
    diagnoses = structure.get("diagnoses", [])

    if not meds:
        answer = "No medications were extracted from the document."
    else:
        # build answer listing meds and simple purpose from heuristic map
        parts = []
        for m in meds:
            name = m.get("name", "").strip()
            dosage = m.get("dosage", "")
            qty = m.get("quantity", "")
            searchable = name.lower()
            purpose = None
            for key, val in _DIGIT_WORDS.items():
                if key in searchable:
                    purpose = val
                    break
            if not purpose:
                # try to infer from diagnoses
                if diagnoses:
                    purpose = f"likely related to treating {', '.join(diagnoses)}"
                else:
                    purpose = "purpose not determined from document"
            desc = f"{name} {dosage} {qty} — {purpose}".strip()
            parts.append(desc)
        answer = "; ".join(parts)

    return JSONResponse({"answer": answer})


# Simple health check
@app.get("/health")
async def health():
    return {"status": "ok", "stored_documents": len(DOCUMENT_STORE)}
