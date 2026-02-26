# ---- Stage 1: Build Angular frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npx ng build --configuration=production

# ---- Stage 2: Python backend + serve frontend ----
FROM python:3.12-slim
WORKDIR /app

COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY src/ ./src/
COPY data/ ./data/
COPY main.py ./

# Copy Angular build output
COPY --from=frontend-build /app/frontend/dist/frontend/browser ./static

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
