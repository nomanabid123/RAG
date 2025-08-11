import fitz




def parse_document(file_path):
    doc = fitz.open(file_path)

    for page in doc:
        text = page.get_text()
        print(text)