FROM python:3.8-slim

WORKDIR /app

# Dependencies first so this layer stays cached when the code changes
RUN pip install --no-cache-dir \
    numpy==1.20.1 \
    scikit-learn==0.22.1 \
    pandas==1.4.1 \
    scipy

COPY . .

CMD ["python", "demo.py", "--dataset=ad", "--threshold=403", "--branch=0"]
