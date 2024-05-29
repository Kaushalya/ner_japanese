FROM python:3.11-slim
WORKDIR /
RUN apt-get update && apt-get install -y git \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*
# Install uv for faster installation of pip packages
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
CMD ["python", "gradio_app.py"]