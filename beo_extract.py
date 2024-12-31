import sys
import os
import re
import fitz  # pymupdf
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageOps
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"  # Choose the model size based on your resources
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# OCR-based PDF parsing
def parse_pdf_with_ocr(file_path):
    """Extract text from a PDF file, including OCR for scanned images, with enhanced accuracy."""
    pdf_text = []
    doc = fitz.open(file_path)  # Open the PDF file
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if not text.strip():  # If no text is extracted, use OCR
            print(f"Performing OCR on page {page_num + 1} with enhanced settings...")
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = img.convert("L")
            img = ImageOps.autocontrast(img)
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
        pdf_text.append(text)
    doc.close()
    return pdf_text

# Save and load OCR output
def save_ocr_output(pdf_text, output_file="ocr_output.json"):
    """Save OCR output to a file."""
    with open(output_file, "w") as f:
        json.dump(pdf_text, f)

def load_ocr_output(input_file="ocr_output.json"):
    """Load OCR output from a file."""
    with open(input_file, "r") as f:
        return json.load(f)

def clean_with_t5(text):
    # Prefix the task (e.g., summarization or structure conversion)
    input_text = f"structure text: {text}"
    
    # Tokenize the input
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate output
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    
    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# Get the latest PDF file
def get_latest_pdf(directory):
    """Find the most recent PDF in the specified directory."""
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the directory.")
    pdf_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return os.path.join(directory, pdf_files[0])

# Extract events
def extract_events_from_text(pdf_text):
    """Separate text into blocks based on each event."""
    event_blocks = []
    current_event = []
    page_pattern = re.compile(r"Page \d+ of \d+", re.IGNORECASE)
    for i, page_text in enumerate(pdf_text):
        if page_pattern.search(page_text):
            current_event.append(page_text)
        else:
            if current_event:
                event_blocks.append("\n".join(current_event))
                current_event = []
    if current_event:
        event_blocks.append("\n".join(current_event))
    return event_blocks

# Parse a single event
def parse_event(event_text, patterns):
    """Parse a single event text block into a structured dictionary."""
    event_data = {
        "General": {},
        "Equipment": {},
        "Layout": {},
        "Details": {
            "Food": {},
            "Beverage": {}
        }
    }
    
    lines = event_text.splitlines()  # Split the text into individual lines
    
    for attribute, pattern in patterns.items():
        if attribute == "Date":  # Special handling for unlabeled lines
            for line in lines:
                match = re.match(pattern, line.strip())
                if match:
                    event_data["General"]["Date"] = line.strip()
                    break
        else:
            match = re.search(pattern, event_text)
            value = match.group(1).strip() if match else None
            
            if attribute in ["BEO", "Booking Type"]:
                event_data["General"][attribute] = value
            elif attribute in ["F&B", "Tables", "Decor"]:
                event_data["Equipment"][attribute] = value
            elif attribute in ["Table Settings", "Descriptions of Table Locations", "Buffet Setup"]:
                event_data["Layout"][attribute] = value
            elif attribute in ["Music", "Provided Stuff"]:
                event_data["Details"][attribute] = value
            elif attribute in ["Menu", "Prices"]:
                event_data["Details"]["Food"][attribute] = value
            elif attribute in ["Package", "Tickets", "Special Requests"]:
                event_data["Details"]["Beverage"][attribute] = value
                

    return event_data

# Parse all events
def parse_all_events(event_blocks, patterns):
    """Parse all event blocks into a JSON list."""
    events_json = []
    for event_text in event_blocks:
        event_json = parse_event(event_text, patterns)
        events_json.append(event_json)
    return events_json

if __name__ == "__main__":
    directory = '/Users/brayden/Library/Mobile Documents/com~apple~CloudDocs/TGCC/BEO Ingest'
    ocr_output_file = "ocr_output.json"
    try:
        # Step 1: Extract or load OCR text
        if os.path.exists(ocr_output_file):
            print(f"Loading OCR output from {ocr_output_file}")
            pdf_text = load_ocr_output(ocr_output_file)
        else:
            file_path = get_latest_pdf(directory)
            print(f"Processing file: {file_path}")
            pdf_text = parse_pdf_with_ocr(file_path)
            save_ocr_output(pdf_text, ocr_output_file)

        # Step 2: Extract events
        event_blocks = extract_events_from_text(pdf_text)

        # Step 3: Define attribute patterns
        patterns = {
            "BEO": r"BEO#\s*([^\n]+)",
            "Bar Type": r"Bar Type:\s*([^\n]+)",
            "Timeline": r"Timeline:\s*([^\n]+)",
            "Title": r"Post As:\s*([^\n]+)",
            "Contact": r"On-Site Contact\s*([^\n]+)",
            "Location": r"Location:\s*([^\n]+)",
            "Date": r"^\w{6,9},\s\w{3,9}\s\d{1,2},\s\d{4}$",
            "Catering Manager": r"Catering Manager\s*([^\n]+)",
            "Booking Type": r"Booking Type:\s*([^\n]+)",
            "Expected People": r"Expected People:\s*(\d+)",
            "F&B": r"F&B:\s*([^\n]+)",
            "Tables": r"Tables:\s*([^\n]+)",
            "Decor": r"Decor:\s*([^\n]+)",
            "Table Settings": r"Table Settings:\s*([^\n]+)",
            "Descriptions of Table Locations": r"Descriptions of Table Locations:\s*([^\n]+)",
            "Buffet Setup": r"Buffet Setup:\s*([^\n]+)",
            "Music": r"Music:\s*([^\n]+)",
            "Provided Stuff": r"Provided Stuff:\s*([^\n]+)",
            "Menu": r"Menu:\s*([^\n]+)",
            "Prices": r"Prices:\s*([^\n]+)",
            "Package": r"Package:\s*([^\n]+)",
            "Tickets": r"Tickets:\s*([^\n]+)",
            "Special Requests": r"Special Requests:\s*([^\n]+)"
        }

        # Step 4: Parse events into structured JSON
        events_json = parse_all_events(event_blocks, patterns)

        # Step 5: Output JSON
        json_output = json.dumps(events_json, indent=4)
        print(json_output)

        # Optional: Save JSON to a file
        output_file = "nested_parsed_events.json"
        with open(output_file, "w") as f:
            f.write(json_output)
        print(f"JSON output saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        