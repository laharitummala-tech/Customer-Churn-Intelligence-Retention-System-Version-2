FROM python:3.11-slim

WORKDIR /churn_app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY frontend/ ./frontend/

EXPOSE 8000
EXPOSE 7860

CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/streamlit_app.py --server.port 7860 --server.address 0.0.0.0 & wait