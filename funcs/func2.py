import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
import json
from typing import List
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# OCR Decision
# -------------------------------
def needs_ocr(text):
    if text is None:
        return True

    text = text.strip()

    if len(text) < 20:
        return True

    valid_chars = sum(c.isalnum() for c in text)
    total_chars = len(text)

    if total_chars == 0:
        return True

    quality_ratio = valid_chars / total_chars

    words = text.split()
    valid_words = [w for w in words if re.match(r'^[a-zA-Z0-9]+$', w)]
    word_ratio = len(valid_words) / (len(words) + 1e-5)

    if quality_ratio < 0.5 or word_ratio < 0.4:
        return True

    return False


# -------------------------------
# OCR Extraction
# -------------------------------
def ocr_pdf(file_path):
    images = convert_from_path(file_path)
    pages = []

    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)

        pages.append({
            "page_num": i,
            "text": text.lower()
        })

    return pages


# -------------------------------
# Hybrid Extraction
# -------------------------------
def extract_pdf_pages(uploaded_file):
    pages = []
    use_ocr = False

    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if needs_ocr(text):
                use_ocr = True
                break

            pages.append({
                "page_num": i,
                "text": text.lower()
            })

    if use_ocr:
        print("⚠️ Using OCR fallback...")
        return ocr_pdf(uploaded_file)

    return pages


# -------------------------------
# Section Keywords
# -------------------------------
SECTION_KEYWORDS = {
    "balance_sheet": [
        "balance sheet",
        "statement of financial position"
    ],
    "income_statement": [
        "statement of profit and loss",
        "income statement",
        "profit and loss"
    ],
    "cash_flow": [
        "cash flow statement",
        "statement of cash flows"
    ]
}


# -------------------------------
# Stop Keywords
# -------------------------------
STOP_KEYWORDS = [
    "notes to financial statements",
    "the accompanying notes",
    "board of directors",
    "auditors report"
]

GENERIC_SECTION_BOUNDARY_PATTERNS = [
    r"\bnotes?\s+to\b",
    r"\bauditors?\b",
    r"\bdirectors?\s+report\b",
    r"\bmanagement\s+discussion\b",
    r"\bstatement\s+of\b",
    r"\breport\s+of\b"
]


def is_generic_section_boundary(text):
    # Use the first lines because headings usually appear at the top.
    top_text = "\n".join(text.splitlines()[:8]).strip()
    if not top_text:
        return False
    return any(re.search(pattern, top_text) for pattern in GENERIC_SECTION_BOUNDARY_PATTERNS)


# -------------------------------
# Detect Sections (fixed)
# -------------------------------
def detect_sections(pages):
    detected = []
    last_seen = {}

    for page in pages:
        text = page["text"]

        for section, keywords in SECTION_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):

                # skip duplicate detections on nearby pages
                if section in last_seen and page["page_num"] - last_seen[section] <= 1:
                    continue

                detected.append({
                    "section": section,
                    "start_page": page["page_num"]
                })

                last_seen[section] = page["page_num"]

    return detected


# -------------------------------
# Assign Section Ranges (fixed)
# -------------------------------
def assign_section_ranges(detected_sections, pages):
    detected_sections = sorted(detected_sections, key=lambda x: x["start_page"])

    results = []
    section_keywords_flat = {
        keyword
        for keywords in SECTION_KEYWORDS.values()
        for keyword in keywords
    }

    for i, sec in enumerate(detected_sections):
        start = sec["start_page"]

        # default end = next section start
        if i + 1 < len(detected_sections):
            end = detected_sections[i + 1]["start_page"]
        else:
            end = len(pages)

        for j in range(start + 1, end):
            page_text = pages[j]["text"]

            # Hard boundaries
            if any(k in page_text for k in STOP_KEYWORDS):
                end = j
                break
            if any(k in page_text for k in section_keywords_flat):
                end = j
                break
            if is_generic_section_boundary(page_text):
                end = j
                break

        # Guarantee a non-empty slice for extraction.
        if end <= start:
            end = min(start + 1, len(pages))

        results.append({
            "section": sec["section"],
            "start_page": start,
            "end_page": end
        })

    return results


# -------------------------------
# Extract Section Text
# -------------------------------
def extract_sections_text(pages, section_ranges):
    extracted = {}

    for sec in section_ranges:
        if sec["section"] in ["balance_sheet", "income_statement", "cash_flow"]:
            section_name = sec["section"]
            start = sec["start_page"]
            end = sec["end_page"]

            lines = []

            for i in range(start, end):
                page_text = pages[i]["text"]
                lines.append(page_text.strip())

            extracted[section_name] = "\n\n".join(lines)
        else:
            print(f"Skipping section {sec['section']} for now")
    return extracted


# -------------------------------
# Validation
# -------------------------------
def validate_section(section_name, text):
    checks = {
        "balance_sheet": ["assets", "liabilities", "equity"],
        "income_statement": ["revenue", "expenses", "profit"],
        "cash_flow": ["operating", "investing", "financing"]
    }

    required = checks.get(section_name, [])

    score = sum(1 for word in required if word in text)

    return score / len(required) if required else 0


# -------------------------------
# Final Pipeline
# -------------------------------
def extract_financial_statements(uploaded_file):
    pages = extract_pdf_pages(uploaded_file)

    detected = detect_sections(pages)

    if not detected:
        print("❌ No sections detected")
        return {}

    section_ranges = assign_section_ranges(detected, pages)

    # 🔍 Debug ranges
    print("\n📊 Detected Section Ranges:")
    for sec in section_ranges:
        print(sec)

    extracted = extract_sections_text(pages, section_ranges)

    json_string = json.dumps(extracted)
    splits = json.loads(json_string)

    # 🔍 Validation scores
    print("\n📊 Validation Scores:")
    for sec, text in extracted.items():
        score = validate_section(sec, text)
        print(f"{sec}: {score:.2f}")

    return splits

def create_vectorstore(documents: List[str]) -> Chroma:
    """Create vector store from documents"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.create_documents(documents)

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vector_store.persist()

    return 0
def vector_store_init(uploaded_file):
    # file_path = r"data\Annual_Report_FY25-152-157.pdf"
    results = extract_financial_statements(uploaded_file)

    documents = []

    for section, content in results.items():
        if content.strip():  # avoid empty
            documents.append(f"{section.upper()}\n\n{content}")

    create_vectorstore(documents)


    print("\n✅ Vector store created successfully!")
    return results

if __name__ == "__main__":
    file_path = "data\Annual_Report_FY25-152-157.pdf"

    vector_store_init(file_path)
    

    # for section, content in results.items():
    #     print("\n====================")
    #     print("SECTION:", section)
    #     print("====================\n")
    #     print(content[:10000])

    pass