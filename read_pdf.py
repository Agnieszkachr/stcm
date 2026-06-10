import sys
import fitz

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

doc = fitz.open(r'c:\DATA\DOCS\POSTDOC\Q\Q\stcm\TheFormationOfQ.pdf')

start = int(sys.argv[1])
end = int(sys.argv[2]) if len(sys.argv) > 2 else start + 1

for i in range(start, min(end, len(doc))):
    text = doc[i].get_text()
    print(f'\n=== PAGE {i} (PDF p.{i+1}) ===')
    print(text)
