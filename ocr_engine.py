import os
import re
import json
import base64
from datetime import datetime
from pathlib import Path
from PIL import Image
import pytesseract
from dotenv import load_dotenv

load_dotenv()

TESSERACT_CMD = os.getenv(
    'TESSERACT_CMD',
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


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
        return parse_document_text(text, document_type)
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
    """Extract structured data using Gemini Vision."""
    try:
        import logging
        import time
        import google.generativeai as genai

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info(f"Gemini called for: {image_path}")
        logger.info(f"GEMINI_API_KEY present: "
                    f"{bool(os.getenv('GEMINI_API_KEY'))}")

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        with open(image_path, 'rb') as f:
            image_data = f.read()

        ext = Path(image_path).suffix.lower()
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
        }
        mime_type = mime_map.get(ext, 'image/jpeg')

        prompt = f"""You are a veterinary document analyzer
for a pet travel compliance system.

    Document type declared: {document_type}.
    Use this declared type as a strong hint for label selection.
Today's date: {datetime.now().strftime('%Y-%m-%d')}

IMPORTANT: This document contains multiple dates.
You must identify each date by its LABEL/CONTEXT:
- issueDate = "Issue date", "Date of issue",
    "Date delivrance", "Date d'emission"
- expiryDate = "Expiry", "Valid until", "Valid to",
    "Date d'expiration", "Expires", "Valable jusqu'au"
- DO NOT use "Date of birth", "Date de naissance",
    "Birth date" as issueDate or expiryDate

Return ONLY this JSON (no markdown, no explanation):
{{
    "isRelevantDocument": true or false,
    "documentNumber": "the FULL official number exactly
        as printed, including all hyphenated segments.
        Examples: HC-TN-2024-003421, VAC-2024-RB-45621,
        EU-2024-TN-78523, TA-2024-AIR-78234.
        NEVER truncate. NEVER return only the year part.
        NEVER return microchip number",
    "holderName": "owner full name (not pet name)",
    "petName": "name of the pet if visible",
    "issueDate": "YYYY-MM-DD - issue date only,
        NOT birth date",
    "expiryDate": "YYYY-MM-DD - expiry/valid-until
        date only, NOT birth date",
    "issuingOrganization": "official authority,
        ministry, clinic or vet name",
    "detectedDocumentType": "what this document is",
    "confidence": 0.0 to 1.0,
    "rawExtractedText": "all visible text",
    "missingFields": ["fields not found in document"],
    "isExpired": compare expiryDate to today
        {datetime.now().strftime('%Y-%m-%d')},
    "documentQuality": "GOOD" or "MEDIUM" or "POOR",
    "warnings": [],
    "rejectionReason": null or explanation
}}

RELEVANCE RULES:
isRelevantDocument = false if document is:
- University course notes or slides
- Academic papers or textbooks
- Food menus, invoices, receipts
- News articles or books
- Any non-veterinary/non-travel document

isRelevantDocument = true if document is:
- Pet passport
- Vaccination certificate (any vaccine)
- Health certificate
- Transport authorization
- Any official animal/veterinary document

CONFIDENCE RULES:
0.85+ : 4 or more key fields extracted clearly
0.60-0.84 : 2-3 key fields found
0.40-0.59 : 1-2 fields found, quality issues
0.00-0.39 : cannot read or wrong document type

DATE RULES (CRITICAL):
- DATE FIELD MAPPING - read labels carefully, in this exact priority order:
    issueDate labels (in order of priority):
        'Issue Date', 'Date of Issue', 'Date de delivrance', 'Date of Vaccination',
        'Valid From', 'Valable a partir du', 'Date d'emission', 'Issued on', 'Date de vaccination'
    expiryDate labels (in order of priority):
        'Expiry Date', 'Valid Until', 'Valable jusqu\'au', 'Valid To', 'Date d\'expiration',
        'Date de validite', 'Expires', 'Validity end'
- NEVER use: 'Date of Birth', 'Date de naissance', 'Born', 'DOB', 'Birth date'.
- Dates may be followed by parenthetical notes like '(Subject to transport conditions)'.
    Extract only the date part and ignore parenthetical notes.
- If a date is labeled 'Valid From' and another is labeled 'Valid Until' in the same document:
    Valid From = issueDate, Valid Until = expiryDate.
- If issueDate > expiryDate after parsing: swap them.
- isExpired = true ONLY if expiryDate < today.

OWNER/PET RULES (CRITICAL):
- holderName = ONLY the human owner's name.
    Look ONLY in labels: 'Owner Name', 'Owner', 'Nom du proprietaire',
    'Owner & Authorization', 'Owner Information', 'Name of Owner'.
- petName = animal name only from labels: 'Pet Name', 'Nom de l\'animal',
    'Animal Name', or 'Name' under Pet Details.
- NEVER use petName/species/breed as holderName.
- If owner field exists but is blank: return holderName = null.

TRANSPORT AUTHORIZATION RULES:
- issueDate = 'Issue Date'
- expiryDate = 'Travel Date' (effective expiry for single-journey permits)
- issuingOrganization = 'Issuing Authority' or 'Authorized by' or 'Issuing body'
- documentNumber = 'Authorization Number'
- NEVER use 'Validity: Single journey only' as a date.

ISSUING ORGANIZATION RULES:
- issuingOrganization = official body/ministry/clinic/authority issuing document.
- Label priority: 'Issuing Authority', 'Autorite de delivrance', 'Issuing body',
    'Issued by', 'Authorized by', 'Ministry', 'Direction Generale', 'Clinic'.
- For vet-signed certificates, combine vet name + clinic name.
    Example: 'Dr. Ahmed Ben Ali - Happy Paws Veterinary Clinic, Tunis'.

Return ONLY JSON, no markdown, no explanation."""

        image_part = {
            'mime_type': mime_type,
            'data': base64.b64encode(image_data)
                         .decode('utf-8')
        }

        response = None
        for attempt in range(2):
            try:
                response = model.generate_content(
                    [prompt, image_part])
                if response.text:
                    break
            except Exception as e:
                logger.error(f"Gemini attempt {attempt+1} failed: {e}")
                if attempt == 0:
                    time.sleep(1)
                    continue
                return None

        if not response or not response.text:
            return None

        logger.info(f"Gemini raw response: {(response.text or '')[:200]}")
        text = response.text.strip()

        # Clean markdown if present
        if '```' in text:
            text = re.sub(r'```json?\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()

        result = json.loads(text)
        result['source'] = 'gemini'
        return normalize_result(result, document_type)

    except json.JSONDecodeError:
        return None
    except Exception:
        return None

def parse_document_text(text: str,
                        document_type: str = 'UNKNOWN') -> dict:
    """Parse raw text to extract document fields."""
    clean_lines = []
    for line in text.splitlines():
        clean_lines.append(re.sub(r'\s*\([^)]*\)', '', line))
    clean_text = '\n'.join(clean_lines)

    result = {
        'documentNumber': None,
        'holderName': None,
        'petName': None,
        'issueDate': None,
        'expiryDate': None,
        'issuingOrganization': None,
        'detectedDocumentType': None,
        'confidence': 0.0,
        'rawExtractedText': clean_text,
        'missingFields': [],
        'isExpired': False,
        'isRelevantDocument': True,
        'documentQuality': 'POOR',
        'warnings': [],
        'rejectionReason': None,
        'source': 'tesseract'
    }

    # Document number
    doc_patterns = [
        r'(?:Certificate\s*(?:No|Number)|Passport\s*(?:No|Number)|Authorization\s*(?:No|Number)|No[.:\s]+|N[°o][:\s]+|Numero[:\s]+)([A-Z0-9][A-Z0-9\-/]{5,30})',
        r'\b([A-Z]{2,4}[-][A-Z0-9]{2,6}[-][A-Z0-9]{2,6}[-][A-Z0-9]{2,10})\b',
        r'\b([A-Z]{2,4}[-]\d{4}[-][A-Z]{2,5}[-]\d{3,6})\b'
    ]
    for pattern in doc_patterns:
        for match in re.finditer(pattern, clean_text, re.IGNORECASE):
            candidate = clean_value(match.group(1))
            if not candidate:
                continue
            line_start = clean_text.rfind('\n', 0, match.start()) + 1
            line_end = clean_text.find('\n', match.end())
            if line_end == -1:
                line_end = len(clean_text)
            line_context = clean_text[line_start:line_end].lower()
            if any(kw in line_context for kw in ['microchip', 'puce', 'chip']):
                continue
            result['documentNumber'] = candidate.upper()
            break
        if result['documentNumber']:
            break

    # Dates
    date_patterns = [
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})(?:\s*\([^)]*\))?',
        r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})(?:\s*\([^)]*\))?',
    ]
    from datetime import datetime as dt
    doc_type_norm = str(document_type or 'UNKNOWN').upper()

    # Find dates with their context
    issue_keywords = ['issue', 'issued', 'emission',
        'delivrance', 'from', 'depuis',
        'valid from', 'valable a partir',
        'date of vaccination', 'vaccinated on',
        'date de vaccination']
    expiry_keywords = ['expir', 'valid until', 'valid to',
        'valid unti', 'valid unti',
        'valable', 'jusqu', 'expires', 'validity',
        'date d expiration', 'date de validite']
    if 'TRANSPORT_AUTHORIZATION' in doc_type_norm:
        expiry_keywords.append('travel date')
    birth_keywords = ['birth', 'born', 'naissance',
        'ne', 'dob', 'date of birth']

    issue_date = None
    expiry_date = None
    birth_dates = []

    # Search with line-first context (avoids cross-line keyword bleed)
    unlabeled_dates = []
    for line in clean_lines:
        line_lower = line.lower()
        for pattern in date_patterns:
            for match in re.finditer(pattern, line, re.IGNORECASE):
                date_str = match.group(1)
                normalized = normalize_date(date_str)
                if not re.match(r'^\d{4}-\d{2}-\d{2}$', normalized):
                    continue

                is_birth = any(kw in line_lower for kw in birth_keywords)
                is_expiry = any(kw in line_lower for kw in expiry_keywords)
                is_issue = any(kw in line_lower for kw in issue_keywords)

                if is_birth:
                    birth_dates.append(normalized)
                elif is_expiry and not expiry_date:
                    expiry_date = normalized
                elif is_issue and not issue_date:
                    issue_date = normalized
                else:
                    unlabeled_dates.append(normalized)

    if not issue_date and unlabeled_dates:
        issue_date = unlabeled_dates[0]
    if not expiry_date and len(unlabeled_dates) > 1:
        expiry_date = unlabeled_dates[1]

    if issue_date and not expiry_date:
        for pattern in date_patterns:
            for match in re.finditer(pattern, clean_text, re.IGNORECASE):
                date_str = match.group(1)
                start = max(0, match.start() - 120)
                end = min(len(clean_text), match.end() + 120)
                context = clean_text[start:end].lower()
                normalized = normalize_date(date_str)
                if normalized == issue_date:
                    continue
                if any(kw in context for kw in expiry_keywords):
                    expiry_date = normalized
                    break
            if expiry_date:
                break

    # Swap if issue > expiry
    if issue_date and expiry_date:
        try:
            id_parsed = dt.strptime(issue_date, '%Y-%m-%d')
            ex_parsed = dt.strptime(expiry_date, '%Y-%m-%d')
            if id_parsed > ex_parsed:
                issue_date, expiry_date = expiry_date, issue_date
        except:
            pass

    result['issueDate'] = issue_date
    result['expiryDate'] = expiry_date

    # isExpired check
    if result['expiryDate']:
        try:
            expiry = dt.strptime(result['expiryDate'], '%Y-%m-%d')
            result['isExpired'] = expiry < dt.now()
            if result['isExpired']:
                result['warnings'].append('Document is expired')
        except:
            result['isExpired'] = False

    # Name detection
    name_patterns = [
        r'(?im)^\s*(?:Owner\s*Name|Name\s*of\s*Owner|Nom\s*du\s*propri.?taire|Proprietaire)\s*:\s*([A-Z][a-zA-Z\-]+(?:[ \t]+[A-Z][a-zA-Z\-]+){1,3})\s*$',
        r'(?im)^\s*(?:4\.1\.\s*Name\s*of\s*Owner|Owner\s*Name)\s*:\s*([A-Z][a-zA-Z\-]+(?:[ \t]+[A-Z][a-zA-Z\-]+){1,3})\s*$',
    ]
    for pattern in name_patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            result['holderName'] = clean_value(match.group(1))
            break

    pet_name_patterns = [
        r'(?:Pet\s*Name|Nom\s*de\s*l\s*animal|Animal\s*Name)[:\s]+([A-Z][a-zA-Z\-]+(?:[ \t]+[A-Z][a-zA-Z\-]+){0,2})',
    ]
    for pattern in pet_name_patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            result['petName'] = clean_value(match.group(1))
            break

    if (result['holderName'] and result['petName']
            and result['holderName'].lower() == result['petName'].lower()):
        result['holderName'] = None

    issuing_patterns = [
        r'(?:Issuing\s*Authority|Autorite\s*de\s*delivrance|Authorized\s*by|Issued\s*by|Issuing\s*body)\s*:\s*([^\n]{5,80})',
        r'(Direction\s+[A-Z][^\n]{5,80})',
        r'(Ministry\s+of\s+[A-Z][^\n]{5,80})',
        r'(OACA\s*[-\u2014]?\s*Office\s*de\s*l\s*Aviation\s*Civile)',
    ]

    vet_match = re.search(r'(?:Veterinarian|Veterinaire)\s*:\s*([^\n]{3,80})', clean_text, re.IGNORECASE)
    clinic_match = re.search(r'(?:Clinic|Clinique)\s*:\s*([^\n]{3,80})', clean_text, re.IGNORECASE)
    if vet_match and clinic_match:
        result['issuingOrganization'] = clean_value(
            f"{vet_match.group(1)} - {clinic_match.group(1)}"
        )
    elif clinic_match:
        result['issuingOrganization'] = clean_value(clinic_match.group(1))

    if not result['issuingOrganization']:
        for pattern in issuing_patterns:
            match = re.search(pattern, clean_text, re.IGNORECASE)
            if match:
                result['issuingOrganization'] = clean_value(match.group(1))
                break

    # Calculate confidence
    fields_to_check = [
        'documentNumber', 'holderName',
        'issueDate', 'expiryDate', 'issuingOrganization'
    ]
    result['missingFields'] = [
        f for f in fields_to_check if not result.get(f)]
    found = len(fields_to_check) - len(result['missingFields'])
    result['confidence'] = round(found / len(fields_to_check), 2)
    result['documentQuality'] = (
        'GOOD' if result['confidence'] >= 0.85
        else 'MEDIUM' if result['confidence'] >= 0.40
        else 'POOR'
    )

    # Detect non-relevant documents
    irrelevant_keywords = [
        'java', 'jpa', 'hibernate', 'spring boot',
        'entity', 'annotation', 'university', 'esprit',
        'course', 'lecture', 'chapter', 'sql', 'database',
        'class diagram', 'uml', 'professor', 'student'
    ]

    text_lower = clean_text.lower()
    irrelevant_count = sum(
        1 for kw in irrelevant_keywords
        if kw in text_lower
    )

    if irrelevant_count >= 3:
        result['isRelevantDocument'] = False
        result['rejectionReason'] = (
            'This appears to be an academic or '
            'technical document, not a pet travel document.'
        )
        result['confidence'] = 0.0
    else:
        result['isRelevantDocument'] = True
        result['rejectionReason'] = None

    return normalize_result(result, document_type)

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
