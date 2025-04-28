import os
from PyPDF2 import PdfReader, PdfWriter

input_dir = "."
output_dir = "Figs"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.startswith("3") and filename.endswith(".pdf"):
        input_pdf_path = os.path.join(input_dir, filename)
        reader = PdfReader(input_pdf_path)
        base_name = os.path.splitext(filename)[0]

        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)

            output_filename = f"{base_name}-{i+1}.pdf"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, "wb") as f_out:
                writer.write(f_out)

        print(f"成功将 {filename} 拆分为 {len(reader.pages)} 个 PDF 文件，保存在 '{output_dir}' 文件夹中。")

print("所有符合条件的文件都处理完毕。")
