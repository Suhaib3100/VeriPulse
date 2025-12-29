FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY core/ ./core/
COPY apps/backend/ ./apps/backend/

CMD ["uvicorn", "apps.backend.main:app", "--host", "0.0.0.0"]
