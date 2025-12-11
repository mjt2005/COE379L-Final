FROM python:3.11

RUN pip install --no-cache-dir \
    tensorflow==2.15 \
    Flask==3.0 \
    Pillow \
    numpy \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    huggingface_hub \
    torch --index-url https://download.pytorch.org/whl/cpu \
    torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy model and Flask app
RUN mkdir /app
WORKDIR /app
COPY lenet5_model_realCIFAR.keras /model/lenet5_model_realCIFAR.keras
COPY model10.keras /model/model10.keras
COPY Deployment.py /app/api.py

# Start Flask server
CMD ["python", "api.py"]