# Authenti.AI

Authenti.AI is a comprehensive deepfake detection system designed to ensure media integrity. It provides advanced forensic analysis for images, videos, and audio, along with evidence integrity tracking using blockchain-ready hashing.

## Features

- **Multi-Modal Detection**: Analyze Images, Videos, and Audio files for deepfake manipulation.
- **Forensic Breakdown**: Detailed explainable AI reports covering:
  - Facial & Visual Analysis (GAN artifacts, landmarks)
  - Temporal Consistency (for video)
  - Lighting & Shadow Analysis
  - Noise & Compression Analysis
  - Audio Signal Analysis (spectral, breathing patterns)
- **Evidence Integrity**: SHA-256 hashing and timestamping for chain-of-custody verification.
- **Heatmap Generation**: Visual localization of manipulated regions in images.
- **URL Analysis**: Direct analysis of media from social media URLs.
- **WhatsApp Integration**: Support for analyzing media forwarded via WhatsApp.

## Tech Stack

- **Backend**: Python, FastAPI, PyTorch/TensorFlow (for models), OpenCV.
- **Frontend**: React, Vite, TailwindCSS.
- **Containerization**: Docker, Docker Compose.

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Node.js & npm (for local frontend dev)
- Python 3.9+ (for local backend dev)

### Quick Start (Docker)

The easiest way to run Authenti.AI is using Docker Compose.

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd authentic-ai-test
    ```

2.  Start the services:
    ```bash
    docker-compose up --build
    ```

3.  Access the application:
    - Frontend: `http://localhost:5173`
    - Backend API Docs: `http://localhost:8000/docs`

### Manual Setup

#### Backend

1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the server:
    ```bash
    uvicorn main:app --reload
    ```

#### Frontend

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Run the development server:
    ```bash
    npm run dev
    ```

## API Documentation

The backend exposes a RESTful API. Full Swagger documentation is available at `/docs` when the backend is running.

### Key Endpoints

- `POST /api/analyze`: Analyze an uploaded image.
- `POST /api/analyze-video`: Analyze an uploaded video.
- `POST /api/analyze-url`: Analyze media from a URL.
- `GET /health`: System health check.

## License

[License Name] - See LICENSE file for details.
