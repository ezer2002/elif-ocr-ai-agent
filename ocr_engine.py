import os
import re
import json
import base64
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
import pytesseract
from dotenv import load_dotenv

load_dotenv()

TESSERACT_CMD = os.getenv(
    'TESSERACT_CMD',
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

logger = logging.getLogger(__name__)


def clean_value(value):
    if value is None:
        return None
    cleaned = re.sub(r'\s+', ' ', str(value)).strip()
    return cleaned or None

def preprocess_image(image_path: str):
    """Enhance image for better OCR — uses only Pillow."""
    try:
        img = Image.open(image_path).convert('L')
        # Increase size for better OCR
        w, h = img.size
        if w < 1000:
            img = img.resize((w * 2, h * 2), Image.LANCZOS)
        return img
    except Exception:
        return Image.open(image_path).convert('L')

def extract_with_tesseract(image_path: str,
                           document_type: str = 'UNKNOWN') -> dict:
    """Extract text using Tesseract OCR."""
    try:
        pil_image = preprocess_image(image_path)
        text = pytesseract.image_to_string(
            pil_image,
            lang='eng+fra',
            config='--psm 3 --oem 3'
        )
        return parse_document_text(text)
    except Exception as e:
        return {
            'documentNumber': None,
            'holderName': None,
            'issueDate': None,
            'expiryDate': None,
            'issuingOrganization': None,
            'detectedDocumentType': None,
            'confidence': 0.0,
            'rawExtractedText': '',
            'missingFields': [],
            'isExpired': False,
            'warnings': [f'Tesseract error: {str(e)}'],
            'source': 'tesseract_error'
        }

def extract_with_gemini(image_path: str,
                        document_type: str):
    text = ''
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        logger.info(f"[OCR] API key present: {bool(api_key)}")

        if not api_key:
            logger.warning("[OCR] No GEMINI_API_KEY - skipping")
            return None

        client = genai.Client(api_key=api_key)

        # Load image
        pil_img = Image.open(image_path).convert('RGB')

        prompt = f"""You are a veterinary document
analyzer for a pet travel compliance system.
Today: {datetime.now().strftime('%Y-%m-%d')}
Document type declared: {document_type}

Analyze this document and return ONLY valid JSON:

{{
  "isRelevantDocument": true or false,
  "documentNumber": "official cert/passport number -
    NOT microchip (15 digits), NOT chip number",
  "holderName": "OWNER full name (the human) -
    look for: Owner, Owner Name, Holder, Proprietaire.
    The name is often on the line AFTER the label.
    NEVER return pet name here.",
  "petName": "animal name if visible",
  "issueDate": "YYYY-MM-DD - date of issue only,
    NEVER date of birth or travel date",
  "expiryDate": "YYYY-MM-DD - valid until date only,
    NEVER date of birth",
  "issuingOrganization": "official issuing authority -
    Ministry, Direction, Veterinary Services, clinic.
    NEVER owner name or pet name.",
  "detectedDocumentType": "document type in English",
  "confidence": 0.0 to 1.0,
  "rawExtractedText": "all visible text verbatim",
  "missingFields": ["fields not found"],
  "isExpired": true if expiryDate before today
    {datetime.now().strftime('%Y-%m-%d')},
  "documentQuality": "GOOD" or "MEDIUM" or "POOR",
  "warnings": [],
  "rejectionReason": null or reason
}}

CRITICAL RULES:
- holderName: search label then value on NEXT line
- documentNumber: format like EU-2024-TN-xxx
  or VAC-2024-RB-xxx, NOT 15-digit microchip
- issueDate/expiryDate: ignore Date of Birth completely
- if issueDate > expiryDate: swap them
- isRelevantDocument false for: course notes,
  code, academic documents, menus, invoices
- confidence 0.85+ if holderName + documentNumber
  + expiryDate all found
Return ONLY the JSON. No markdown. No explanation."""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pil_img]
        )

        if not response or not getattr(response, 'text', None):
            return None

        logger.info(f"[OCR] Gemini response: {response.text[:300]}")
        text = response.text.strip()

        if '```' in text:
            text = re.sub(r'```json?\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()

        result = json.loads(text)
        result['source'] = 'gemini'
        logger.info(f"[OCR] Confidence: "
                    f"{result.get('confidence', 0)}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"[OCR] JSON parse error: {e}")
        logger.error(f"[OCR] Raw text was: {text[:200]}")
        return None
    except Exception as e:
        logger.error(f"[OCR] Gemini error: {e}")
        return None

def parse_document_text(text: str) -> dict:
    from datetime import datetime as dt

    result = {
        'isRelevantDocument': True,
        'documentNumber': None,
        'holderName': None,
        'petName': None,
        'issueDate': None,
        'expiryDate': None,
        'issuingOrganization': None,
        'detectedDocumentType': None,
        'confidence': 0.0,
        'rawExtractedText': text,
        'missingFields': [],
        'isExpired': False,
        'documentQuality': 'MEDIUM',
        'warnings': [],
        'source': 'tesseract',
        'rejectionReason': None
    }

    lines = [l.strip() for l in text.split('\n')
             if l.strip()]
    text_lower = text.lower()

    # --- Document Number ---
    doc_patterns = [
        r'(?:passport\s*no|certificate\s*no|'
        r'cert(?:ificate)?\s*(?:number|no|#)|'
        r'authorization\s*no|ref(?:erence)?\s*no|'
        r'no\.|number)[:\s#]+([A-Z0-9][A-Z0-9\-]{3,20})',
        r'\b([A-Z]{2,4}[\-][0-9]{4}[\-][A-Z]{0,4}[\-]?[0-9]+)\b',
    ]
    for pattern in doc_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            # Skip microchip (15 consecutive digits)
            if not re.match(r'^\d{15}$', val):
                result['documentNumber'] = val
                break

    # --- Holder Name (multi-line aware) ---
    owner_label_patterns = [
        r'(?:owner(?:\s+name)?|holder|proprietaire|'
        r'nom\s+du\s+proprietaire|issued\s+to|'
        r'owner\s+information)'
        r'[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'(?:owner(?:\s+name)?|holder|proprietaire)'
        r'[\s:]*\n+\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
    ]
    for pattern in owner_label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result['holderName'] = match.group(1).strip()
            break

    # Multi-line fallback: find "Owner" then next line
    if not result['holderName']:
        for i, line in enumerate(lines):
            if re.match(
                r'^(owner|holder|proprietaire'
                r'|owner\s+name|owner\s+information)$',
                line, re.IGNORECASE
            ):
                # Get next non-empty line
                for j in range(i+1, min(i+4, len(lines))):
                    candidate = lines[j].strip()
                    # Must look like a name
                    if (re.match(
                        r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)+$',
                        candidate
                    ) and len(candidate) > 4):
                        result['holderName'] = candidate
                        break
                if result['holderName']:
                    break

    # --- Issuing Organization ---
    org_patterns = [
        r'(?:issued\s+by|issuing\s+authority|'
        r'authority|autorite|direction|ministry|'
        r'ministere|veterinarian|veterinary\s+services|'
        r'clinique|clinic|cabinet)'
        r'[\s:]+([A-Za-zÀ-ÿ\s\.,]{5,60})',
    ]
    for pattern in org_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            org = match.group(1).strip()
            # Not a name or pet breed
            if (len(org) > 5
                    and org != result.get('holderName')
                    and org != result.get('petName')):
                result['issuingOrganization'] = org
                break

    # Fallback: look for official authority keywords
    if not result['issuingOrganization']:
        official_keywords = [
            'ministry', 'direction', 'ministere',
            'services vétérinaires', 'veterinary',
            'oaca', 'aviation', 'agriculture'
        ]
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower
                   for kw in official_keywords):
                result['issuingOrganization'] = line.strip()
                break

    # --- Dates with context ---
    birth_keywords = ['birth', 'born', 'naissance',
        'né', 'dob', 'date of birth', 'date naissance']
    issue_keywords = ['issue', 'issued', 'emission',
        'délivrance', 'delivrance', 'valid from', 'depuis',
        'date issue', 'date of issue']
    expiry_keywords = ['expir', 'valid until', 'valid to',
        'valable', "jusqu'au", 'expires', 'validity',
        'valid thru', 'valid through']

    date_pattern = (
        r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|'
        r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})\b'
    )

    issue_date = None
    expiry_date = None

    for match in re.finditer(date_pattern, text,
                              re.IGNORECASE):
        date_str = match.group(0)
        start = max(0, match.start() - 60)
        end = min(len(text), match.end() + 60)
        context = text[start:end].lower()

        is_birth = any(kw in context
            for kw in birth_keywords)
        if is_birth:
            continue  # Skip birth dates entirely

        is_expiry = any(kw in context
            for kw in expiry_keywords)
        is_issue = any(kw in context
            for kw in issue_keywords)

        normalized = normalize_date(date_str)
        if not normalized:
            continue

        if is_expiry and not expiry_date:
            expiry_date = normalized
        elif is_issue and not issue_date:
            issue_date = normalized
        elif not expiry_date:
            expiry_date = normalized
        elif not issue_date:
            issue_date = normalized

    # Swap if needed
    if issue_date and expiry_date:
        try:
            id_p = dt.strptime(issue_date, '%Y-%m-%d')
            ex_p = dt.strptime(expiry_date, '%Y-%m-%d')
            if id_p > ex_p:
                issue_date, expiry_date = (
                    expiry_date, issue_date)
        except:
            pass

    result['issueDate'] = issue_date
    result['expiryDate'] = expiry_date

    # isExpired check
    if expiry_date:
        try:
            expiry = dt.strptime(expiry_date, '%Y-%m-%d')
            result['isExpired'] = expiry < dt.now()
            if result['isExpired']:
                result['warnings'].append(
                    'Document is expired')
        except:
            pass

    # Irrelevant document detection
    irrelevant_kws = [
        'java', 'jpa', 'hibernate', 'spring boot',
        'entity', '@entity', 'annotation', 'esprit school',
        'lecture', 'chapter', 'sql', 'database schema',
        'class diagram', 'professor', 'student notes'
    ]
    irrelevant_count = sum(
        1 for kw in irrelevant_kws
        if kw in text_lower)
    if irrelevant_count >= 3:
        result['isRelevantDocument'] = False
        result['confidence'] = 0.0
        result['rejectionReason'] = (
            'This appears to be an academic document, '
            'not a pet travel document.')
        return normalize_result(result)

    # Confidence
    found_fields = sum([
        bool(result['documentNumber']),
        bool(result['holderName']),
        bool(result['issueDate']),
        bool(result['expiryDate']),
        bool(result['issuingOrganization'])
    ])
    result['confidence'] = round(found_fields / 5, 2)
    result['documentQuality'] = (
        'GOOD' if result['confidence'] >= 0.6
        else 'MEDIUM' if result['confidence'] >= 0.4
        else 'POOR'
    )

    fields_to_check = [
        'documentNumber', 'holderName',
        'issueDate', 'expiryDate', 'issuingOrganization'
    ]
    result['missingFields'] = [
        f for f in fields_to_check
        if not result.get(f)]

    return normalize_result(result)

def normalize_date(date_str: str) -> str:
    """Normalize date to YYYY-MM-DD."""
    formats = [
        '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
        '%Y/%m/%d', '%Y-%m-%d', '%m/%d/%Y',
        '%d/%m/%y',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(
                date_str.strip(), fmt
            ).strftime('%Y-%m-%d')
        except:
            continue
    return date_str


def normalize_result(result: dict,
                     document_type: str = 'UNKNOWN') -> dict:
    text_fields = [
        'documentNumber', 'holderName', 'petName',
        'issueDate', 'expiryDate', 'issuingOrganization',
        'detectedDocumentType', 'rawExtractedText',
        'documentQuality', 'rejectionReason', 'source'
    ]
    for field in text_fields:
        if field in result:
            result[field] = clean_value(result.get(field))

    if result.get('documentNumber'):
        result['documentNumber'] = result['documentNumber'].upper()

    if result.get('issuingOrganization'):
        result['issuingOrganization'] = re.sub(
            r'\s+', ' ', result['issuingOrganization']
        ).strip()

    generic_owner_tokens = {
        'owner', 'owner name', 'name of owner',
        'nom du proprietaire', 'holder', 'holder name'
    }
    holder_value = (result.get('holderName') or '').strip().lower()
    if holder_value in generic_owner_tokens:
        result['holderName'] = None

    if (result.get('holderName') and result.get('petName')
            and result['holderName'].lower() == result['petName'].lower()):
        result['holderName'] = None

    issue = result.get('issueDate')
    expiry = result.get('expiryDate')
    if issue and expiry:
        try:
            issue_dt = datetime.strptime(issue, '%Y-%m-%d')
            expiry_dt = datetime.strptime(expiry, '%Y-%m-%d')
            if issue_dt > expiry_dt:
                result['issueDate'], result['expiryDate'] = expiry, issue
        except:
            pass

    if result.get('expiryDate'):
        try:
            expiry_dt = datetime.strptime(result['expiryDate'], '%Y-%m-%d')
            result['isExpired'] = expiry_dt < datetime.now()
        except:
            result['isExpired'] = False
    else:
        result['isExpired'] = False

    if 'isRelevantDocument' not in result:
        result['isRelevantDocument'] = True

    if not result.get('documentQuality'):
        confidence = float(result.get('confidence') or 0)
        result['documentQuality'] = (
            'GOOD' if confidence >= 0.85
            else 'MEDIUM' if confidence >= 0.40
            else 'POOR'
        )

    missing = result.get('missingFields') or []
    if not isinstance(missing, list):
        missing = []
    normalized_missing = []
    for field in missing:
        cleaned = clean_value(field)
        if cleaned:
            normalized_missing.append(cleaned)
    result['missingFields'] = normalized_missing

    if not normalized_missing:
        required = [
            'documentNumber', 'holderName', 'issueDate',
            'expiryDate', 'issuingOrganization'
        ]
        result['missingFields'] = [
            field for field in required if not result.get(field)
        ]

    if result.get('documentNumber'):
        if re.match(r'^\d{15}$', str(result['documentNumber'])):
            result['missingFields'] = [
                f for f in result['missingFields'] if f != 'documentNumber'
            ]
            result['documentNumber'] = None
            if 'documentNumber' not in result['missingFields']:
                result['missingFields'].append('documentNumber')

    return result

def analyze_document(image_path: str,
                     document_type: str = 'UNKNOWN') -> dict:

    gemini_result = extract_with_gemini(
        image_path, document_type)

    # If Gemini detected wrong document
    if (gemini_result
            and isinstance(gemini_result, dict)
            and gemini_result.get(
                'isRelevantDocument') == False):
        return normalize_result({
            'isRelevantDocument': False,
            'documentNumber': None,
            'holderName': None,
            'petName': None,
            'issueDate': None,
            'expiryDate': None,
            'issuingOrganization': None,
            'detectedDocumentType':
                gemini_result.get(
                    'detectedDocumentType',
                    'Unknown document'),
            'confidence': 0.0,
            'rawExtractedText':
                gemini_result.get(
                    'rawExtractedText', ''),
            'missingFields': [],
            'isExpired': False,
            'documentQuality': 'POOR',
            'warnings': ['Wrong document type'],
            'rejectionReason':
                gemini_result.get(
                    'rejectionReason',
                    'This is an academic document, '
                    'not a pet travel document.'),
            'source': 'gemini'
        }, document_type)

    # Gemini success with confidence > 0.4
    if (gemini_result
            and isinstance(gemini_result, dict)
            and gemini_result.get('confidence', 0) > 0.4):
        gemini_result['isRelevantDocument'] = True
        return normalize_result(gemini_result, document_type)

    # Tesseract fallback
    result = extract_with_tesseract(image_path, document_type)
    result['source'] = 'tesseract_fallback'
    return normalize_result(result, document_type)
