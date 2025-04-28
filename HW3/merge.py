from PyPDF2 import PdfMerger

pdf_list = [
    "3.1.pdf",
    "3.2.pdf",
    "3.3.pdf",
    "3.4.pdf",
    "3.5.pdf",
    "3.6.pdf",
    "3.7.pdf"
]

merger = PdfMerger()

for pdf in pdf_list:
    merger.append(pdf)

merger.write("HW3.pdf")
merger.close()
