from PyPDF2 import PdfMerger

pdf_list = [

]

merger = PdfMerger()

for pdf in pdf_list:
    merger.append(pdf)

merger.write(".pdf")
merger.close()
