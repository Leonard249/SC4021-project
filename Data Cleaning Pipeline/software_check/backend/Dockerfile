FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY profiles.json /app/profiles.json
COPY db_*.json /app/
COPY selected_data.json /app/selected_data.json
COPY scripts /app/scripts

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
