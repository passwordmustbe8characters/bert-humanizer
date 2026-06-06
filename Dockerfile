FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bert_humanizer.py .

CMD ["python", "-u", "bert_humanizer.py"]