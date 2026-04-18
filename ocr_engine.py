import os
import re
import json
import base64
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
from google import genai
import pytesseract
from dotenv import load_dotenv

try:
    import openai
except ImportError:
    openai = None

load_dotenv()

TESSERACT_CMD = os.getenv(
    'TESSERACT_CMD',
    '/usr/bin/tesseract' if os.name != 'nt' else r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    'documentNumber',
    'holderName',
    'issueDate',
    'expiryDate',
    'issuingOrganization',
]


def clean_value(value):
    if value is None:
        return None
    cleaned = re.sub(r'\s+', ' ', str(value)).strip()
    return cleaned or None


def preprocess_image(image_path: str):
    try:
        img = Image.open(image_path).convert('L')
        w, h = img.size
        if w < 1000:
            img = img.resize((w * 2, h * 2), Image.LANCZOS)
        return img
    except Exception:
        return Image.open(image_path).convert('L')


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        return base64.standard_b64encode(image_file.read()).decode('utf-8')


def extract_with_openai(image_path: str, document_type: str) -> dict:
    if not openai:
        logger.warning("[OCR] OpenAI not installed - skipping")
        return None

    try:
        api_key = os.getenv('OPENAI_API_KEY')
        logger.info(f"[OCR] OpenAI key present: {bool(api_key)}")

        if not api_key:
            logger.warning("[OCR] No OPENAI_API_KEY - skipping")
            return None

        client = openai.OpenAI(api_key=api_key)
        image_data = encode_image_to_base64(image_path)
        file_ext = Path(image_path).suffix.lower()
        media_type = "image/jpeg" if file_ext in ['.jpg', '.jpeg'] else "image/png" if file_ext == '.png' else "image/jpeg"

        prompt = f"""You are a veterinary document analyzer for a pet travel compliance system.
Today: {datetime.now().strftime('%Y-%m-%d')}
Document type declared: {document_type}

Analyze this document and return ONLY valid JSON (no markdown):

{{
  "isRelevantDocument": true or false,
  "documentNumber": "FULL official number - Examples: EU-2024-TN-78523, VAC-2024-RB-45621. NEVER truncate. EXCLUDE microchip (15 digits)",
  "holderName": "ONLY owner name (NOT pet). Look in: 'Owner Name', 'Owner', 'Nom du propriétaire'. If blank → null, NOT pet name.",
  "petName": "animal name - look in: 'Pet Name', 'Nom de l'animal'",
  "issueDate": "YYYY-MM-DD. Labels: 'Issue Date', 'Date of Vaccination', 'Valid From'. NEVER birth dates.",
  "expiryDate": "YYYY-MM-DD. Labels: 'Expiry Date', 'Valid Until', 'Travel Date'. NEVER birth dates.",
  "issuingOrganization": "official body/ministry/clinic. Example: 'Dr. Ahmed Ben Ali — Happy Paws Veterinary Clinic, Tunis'",
  "detectedDocumentType": "document type in English",
  "confidence": 0.85-1.0 if all 5 fields; 0.60-0.84 if 3-4; 0.40-0.59 if 2; 0.0-0.39 if 0-1",
  "rawExtractedText": "all visible text",
  "missingFields": [],
  "isExpired": true if expiryDate before today, false otherwise,
  "documentQuality": "GOOD" or "MEDIUM" or "POOR",
  "warnings": [],
  "rejectionReason": null or reason
}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{media_type.split('/')[1]};base64,{image_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )

        if not response or not response.choices:
            logger.warning("[OCR] OpenAI returned empty response")
            return None

        text = response.choices[0].message.content.strip()
        logger.info(f"[OCR] OpenAI response: {text[:300]}")

        if '```' in text:
            text = re.sub(r'```json?\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()

        result = json.loads(text)
        result['source'] = 'openai'
        logger.info(f"[OCR] OpenAI Confidence: {result.get('confidence', 0)}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"[OCR] OpenAI JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"[OCR] OpenAI error: {e}")
        return None


def extract_with_gemini(image_path: str, document_type: str):
    text = ''
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        logger.info(f"[OCR] Gemini key present: {bool(api_key)}")

        if not api_key:
            logger.warning("[OCR] No GEMINI_API_KEY - skipping")
            return None

        client = genai.Client(api_key=api_key)
        pil_img = Image.open(image_path).convert('RGB')

        prompt = f"""You are a veterinary document analyzer. Analyze and return ONLY valid JSON:
{{
  "isRelevantDocument": true or false,
  "documentNumber": "FULL number (Examples: EU-2024-TN-78523, VAC-2024-RB-45621). NEVER truncate",
  "holderName": "ONLY owner name, NOT pet name",
  "petName": "animal name",
  "issueDate": "YYYY-MM-DD",
  "expiryDate": "YYYY-MM-DD",
  "issuingOrganization": "official body",
  "detectedDocumentType": "type",
  "confidence": 0.0-1.0,
  "rawExtractedText": "all text",
  "missingFields": [],
  "isExpired": false,
  "documentQuality": "GOOD",
  "warnings": [],
  "rejectionReason": null
}}"""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pil_img]
        )

        if not response or not getattr(response, 'text', None):
            logger.warning("[OCR] Gemini returned empty response")
            return None

        text = response.text.strip()
        if '```' in text:
            text = re.sub(r'```json?\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()

        result = json.loads(text)
        result['source'] = 'gemini'
        logger.info(f"[OCR] Gemini Confidence: {result.get('confidence', 0)}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"[OCR] Gemini JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"[OCR] Gemini error: {e}")
        return None


def extract_with_tesseract(image_path: str, document_type: str = 'UNKNOWN') -> dict:
    try:
        pil_image = preprocess_image(image_path)
        text = pytesseract.image_to_string(pil_image, lang='eng+fra', config='--psm 3 --oem 3')
        return parse_document_text(text)
    except Exception as e:
        logger.error(f"[OCR] Tesseract error: {e}")
        return {
            'documentNumber': None, 'holderName': None, 'issueDate': None,
            'expiryDate': None, 'issuingOrganization': None, 'detectedDocumentType': None,
            'confidence': 0.0, 'rawExtractedText': '', 'missingFields': REQUIRED_FIELDS.copy(),
            'isExpired': False, 'warnings': [f'Tesseract error: {str(e)}'], 'source': 'tesseract_error'
        }


def parse_document_text(text: str) -> dict:
    from datetime import datetime as dt

    result = {
        'isRelevantDocument': True, 'documentNumber': None, 'holderName': None, 'petName': None,
        'issueDate': None, 'expiryDate': None, 'issuingOrganization': None, 'detectedDocumentType': None,
        'confidence': 0.0, 'rawExtractedText': text, 'missingFields': [], 'isExpired': False,
        'documentQuality': 'MEDIUM', 'warnings': [], 'source': 'tesseract', 'rejectionReason': None
    }

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    text_lower = text.lower()

    # BUG FIX 1: Document Number - Full extraction
    doc_patterns = [
        r'\b([A-Z]{2,4}[-][A-Z0-9]{2,6}[-][A-Z0-9]{2,6}[-][A-Z0-9]{2,10})\b',
        r'\b([A-Z]{2,4}[-]\d{4}[-][A-Z]{2,5}[-]\d{3,6})\b',
        r'(?:Certificate\s*(?:No|Number)|Passport\s*(?:No|Number)|Authorization\s*(?:No|Number)|No[.:\s]+|N°[:\s]+|Numéro[:\s]+)([A-Z0-9][A-Z0-9\-\/]{5,30})',
    ]

    skip_microchip = any(kw in text_lower for kw in ['microchip', 'puce', 'chip'])

    for pattern in doc_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            val = match.group(1).strip()
            if re.match(r'^\d{15}$', val) or skip_microchip:
                continue
            result['documentNumber'] = val
            break
        if result['documentNumber']:
            break

    # BUG FIX 2 + 8: Holder Name - Owner only, supports 2-4 words
    owner_label_patterns = [
        r'(?:^|\n)(owner(?:\s+name)?|holder|proprietaire|nom\s+du\s+proprietaire|owner\s+information)(?:\s|:|$)[\s\n]+([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){1,3})',
        r'(?:owner(?:\s+name)?|holder|proprietaire|nom\s+du\s+proprietaire)[\s:]+([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){1,3})',
    ]

    for pattern in owner_label_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            result['holderName'] = match.group(1).strip()
            break

    if not result['holderName']:
        for i, line in enumerate(lines):
            if re.match(r'^(owner|holder|proprietaire|owner\s+name|owner\s+information)$', line, re.IGNORECASE):
                for j in range(i + 1, min(i + 4, len(lines))):
                    candidate = lines[j].strip()
                    if (re.match(r'^[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){1,3}$', candidate) and len(candidate) > 4):
                        result['holderName'] = candidate
                        break
                if result['holderName']:
                    break

    # BUG FIX 6: Issuing Organization
    org_patterns = [
        r'(?:Issued\s+by|Issuing\s+Authority|Autorité\s+de\s+délivrance|Authorized\s+by)[\s:]+([^\n]{5,80})',
        r'\b((?:Ministry|Ministère|Direction\s+)[A-Za-zÀ-ÿ\s]{5,60})',
        r'(?:Clinic|Clinique|Veterinary)[\s:]+([^\n]{5,60})',
    ]

    for pattern in org_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            org = match.group(1).strip()
            if len(org) > 5 and org != result.get('holderName') and org != result.get('petName'):
                result['issuingOrganization'] = org
                break

    # BUG FIX 3, 4, 5: Dates
    text_no_paren = re.sub(r'\s*\([^)]*\)', '', text)

    birth_keywords = ['birth', 'born', 'naissance', 'né', 'dob', 'date of birth']
    issue_keywords = ['issue', 'issued', 'date of vaccination', 'valid from']
    expiry_keywords = ['expir', 'valid until', 'valid to', 'travel date']

    date_pattern = r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})\b'

    issue_date = None
    expiry_date = None

    for match in re.finditer(date_pattern, text_no_paren, re.IGNORECASE):
        date_str = match.group(0)
        context = text_no_paren[max(0, match.start() - 120):min(len(text_no_paren), match.end() + 120)].lower()

        if any(kw in context for kw in birth_keywords):
            continue

        normalized = normalize_date(date_str)
        if not normalized:
            continue

        if any(kw in context for kw in expiry_keywords) and not expiry_date:
            expiry_date = normalized
        elif any(kw in context for kw in issue_keywords) and not issue_date:
            issue_date = normalized
        elif not expiry_date:
            expiry_date = normalized
        elif not issue_date:
            issue_date = normalized

    if issue_date and expiry_date and issue_date > expiry_date:
        issue_date, expiry_date = expiry_date, issue_date

    result['issueDate'] = issue_date
    result['expiryDate'] = expiry_date

    if expiry_date:
        try:
            result['isExpired'] = dt.strptime(expiry_date, '%Y-%m-%d') < dt.now()
        except:
            pass

    found_fields = sum([bool(result[f]) for f in ['documentNumber', 'holderName', 'issueDate', 'expiryDate', 'issuingOrganization']])
    result['confidence'] = round(found_fields / 5, 2)

    return normalize_result(result)


def normalize_date(date_str: str) -> str:
    formats = ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%Y/%m/%d', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%y']
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime('%Y-%m-%d')
        except:
            continue
    return None


def normalize_result(result: dict, document_type: str = 'UNKNOWN') -> dict:
    for field in ['documentNumber', 'holderName', 'petName', 'issueDate', 'expiryDate', 'issuingOrganization', 'detectedDocumentType', 'rawExtractedText', 'documentQuality', 'rejectionReason', 'source']:
        if field in result:
            result[field] = clean_value(result.get(field))

    if result.get('documentNumber'):
        result['documentNumber'] = result['documentNumber'].upper()

    if result.get('holderName') and result.get('holderName').lower() in ['owner', 'holder', 'proprietaire']:
        result['holderName'] = None

    issue = result.get('issueDate')
    expiry = result.get('expiryDate')
    if issue and expiry and issue > expiry:
        result['issueDate'], result['expiryDate'] = expiry, issue

    if result.get('expiryDate'):
        try:
            result['isExpired'] = datetime.strptime(result['expiryDate'], '%Y-%m-%d') < datetime.now()
        except:
            result['isExpired'] = False
    else:
        result['isExpired'] = False

    if not result.get('documentQuality'):
        confidence = float(result.get('confidence') or 0)
        result['documentQuality'] = 'GOOD' if confidence >= 0.85 else 'MEDIUM' if confidence >= 0.40 else 'POOR'

    result['missingFields'] = [f for f in REQUIRED_FIELDS if not result.get(f)]

    if result.get('documentNumber') and re.match(r'^\d{15}$', str(result['documentNumber'])):
        result['documentNumber'] = None
        if 'documentNumber' not in result['missingFields']:
            result['missingFields'].append('documentNumber')

    return result


def analyze_document(image_path: str, document_type: str = 'UNKNOWN') -> dict:
    # === LEVEL 1 TEMPORARILY DISABLED (OpenAI — uncomment to activate) ===
    # result = extract_with_openai(image_path, document_type)
    # if result and result.get('confidence', 0) > 0.4:
    #     return result
    # logger.warning("[OCR] OpenAI failed — trying Gemini")

    # Kept original OpenAI handling for quick re-activation.
    # logger.info("[OCR] === LEVEL 1: Trying OpenAI ===")
    # openai_result = extract_with_openai(image_path, document_type)
    #
    # if openai_result and isinstance(openai_result, dict) and openai_result.get('isRelevantDocument') == False:
    #     logger.warning("[OCR] OpenAI detected wrong document")
    #     return normalize_result({'isRelevantDocument': False, 'source': 'openai', 'confidence': 0.0, 'documentNumber': None, 'holderName': None, 'issueDate': None, 'expiryDate': None, 'issuingOrganization': None, 'detectedDocumentType': openai_result.get('detectedDocumentType'), 'rawExtractedText': openai_result.get('rawExtractedText', ''), 'missingFields': [], 'isExpired': False, 'documentQuality': 'POOR', 'warnings': [], 'rejectionReason': openai_result.get('rejectionReason')}, document_type)
    #
    # if openai_result and isinstance(openai_result, dict) and openai_result.get('confidence', 0) > 0.4:
    #     logger.info("[OCR] OpenAI confidence > 0.4")
    #     return normalize_result(openai_result, document_type)
    #
    # logger.warning("[OCR] OpenAI failed - trying Gemini")

    logger.warning("[OCR] OpenAI failed - trying Gemini")
    logger.info("[OCR] === LEVEL 2: Trying Gemini ===")
    gemini_result = extract_with_gemini(image_path, document_type)

    if gemini_result and isinstance(gemini_result, dict) and gemini_result.get('isRelevantDocument') == False:
        logger.warning("[OCR] Gemini detected wrong document")
        return normalize_result({'isRelevantDocument': False, 'source': 'gemini', 'confidence': 0.0, 'documentNumber': None, 'holderName': None, 'issueDate': None, 'expiryDate': None, 'issuingOrganization': None, 'detectedDocumentType': gemini_result.get('detectedDocumentType'), 'rawExtractedText': gemini_result.get('rawExtractedText', ''), 'missingFields': [], 'isExpired': False, 'documentQuality': 'POOR', 'warnings': [], 'rejectionReason': gemini_result.get('rejectionReason')}, document_type)

    if gemini_result and isinstance(gemini_result, dict) and gemini_result.get('confidence', 0) > 0.4:
        logger.info("[OCR] Gemini confidence > 0.4")
        return normalize_result(gemini_result, document_type)

    logger.warning("[OCR] Gemini failed - using Tesseract")
    result = extract_with_tesseract(image_path, document_type)
    result['source'] = 'tesseract_fallback'
    return normalize_result(result, document_type)
