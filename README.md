# Image Captioning with BLIP (Optimized)

This project is a simple **Gradio** interface for generating English captions from images using the **BLIP** model.  
It is optimized with **GPU support** and **image preprocessing** for faster and more stable caption generation.

## Features
- Generate English captions for uploaded images
- Image preprocessing (convert to RGB and resize to 384x384)
- Automatic GPU support if available
- Simple and user-friendly Gradio interface

## Installation & Usage

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>

2.(Optional but recommended) Create and activate a virtual environment:

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate


3.Install dependencies:

pip install -r requirements.txt


4.Run the application:

python image.py


After running, Gradio will provide a local URL (usually http://127.0.0.1:7860) to open in your browser and upload images.