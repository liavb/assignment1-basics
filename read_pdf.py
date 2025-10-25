import pdfplumber

with pdfplumber.open('cs336_spring2025_assignment1_basics.pdf') as pdf:
    text = ''
    for page in pdf.pages:
        text += page.extract_text()

    # Find section 7.2
    if '7.2' in text:
        start_idx = text.find('7.2')
        # Get surrounding context
        section = text[max(0, start_idx-200):start_idx+4000]
        print(section)
    else:
        print("Section 7.2 not found. Searching for 'Task 7':")
        if 'Task 7' in text:
            start_idx = text.find('Task 7')
            section = text[start_idx:start_idx+5000]
            print(section)
        else:
            print("Full text:")
            print(text[:10000])

