import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# تشخیص GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# مدل کپشن‌ساز (BLIP)
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# تابع تولید کپشن انگلیسی
def generate_caption(image):
    # پیش‌پردازش تصویر: تغییر اندازه برای بهبود عملکرد مدل
    image = image.convert("RGB")
    image = image.resize((384, 384))  # BLIP معمولاً روی این سایز عملکرد خوبی دارد

    # آماده‌سازی ورودی
    inputs = caption_processor(images=image, return_tensors="pt").to(device)

    # تولید کپشن
    output = caption_model.generate(**inputs, max_length=50)
    caption_en = caption_processor.decode(output[0], skip_special_tokens=True)
    return caption_en

# رابط Gradio
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="📷 Please upload The image"),
    outputs=gr.Textbox(label="📝 Caption (English)"),
    title="🖼️ Image Captioning (Optimized)",
    description="Optimized version of BLIP with GPU support and image preprocessing"
)

if __name__ == "__main__":
    iface.launch()
