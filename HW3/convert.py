from PIL import Image

image_path = "3.1.png"
image = Image.open(image_path)

if image.mode == "RGBA":
    image = image.convert("RGB")

pdf_path = "3.1.pdf"
image.save(pdf_path)

print(f"成功将 {image_path} 转成 {pdf_path}")
