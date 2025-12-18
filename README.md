# Kakinada Traffic AI Analytics

## Description
FastAPI backend for video traffic analysis with helmet, fight, weapon detection and tracking.

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | / | Health check |
| POST | /analyze | Upload & analyze video |
| GET | /download/video/{file} | Download output video |
| GET | /download/csv/{file} | Download event CSV |
| GET | /docs | Swagger UI |
