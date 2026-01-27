FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default to using host.docker.internal to reach n8n running on the host (Windows)
ENV N8N_WEBHOOK_URL=http://host.docker.internal:5678/webhook/23cb51ff-48ee-48fd-8cde-447af19a99ea

CMD ["python", "App.py"]
