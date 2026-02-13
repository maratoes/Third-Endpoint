FROM ghcr.io/huggingface/text-generation-inference:2.0.3

RUN pip install --no-cache-dir runpod==1.6.2 requests

WORKDIR /app
COPY handler.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV MODEL_ID="Qwen/Qwen3-32B-Instruct"
ENV MAX_INPUT_LENGTH=4096
ENV MAX_TOTAL_TOKENS=8192
ENV QUANTIZE="awq"
ENV NUM_SHARD=1

EXPOSE 80
CMD ["python", "-u", "handler.py"]
