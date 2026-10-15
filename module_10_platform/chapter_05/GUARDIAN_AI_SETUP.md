# Guardian AI — Local Setup

## Requirements

- Python 3 with the repository's virtual environment
- Node.js and npm
- Python dependencies: fastapi, uvicorn, httpx, pytest
- Frontend dependencies from chapter_03/package.json

Ollama was used in previous research modules. It is
not required to view the completed Module 10 dashboard.

## 1. Start the backend

From the repository root:

```bash
source venv/bin/activate
cd module_10_platform/chapter_02
python3 -m uvicorn guardian_research_detail_api:app \
  --host 127.0.0.1 --port 8000
```

Backend health: http://127.0.0.1:8000/health
API documentation: http://127.0.0.1:8000/docs

## 2. Start the frontend

In a second terminal:

```bash
cd ~/Downloads/guardian-ai/module_10_platform/chapter_03
npm install
npm run dev
```

Open: http://127.0.0.1:5173

The Vite development server proxies /api and /health
to local FastAPI on port 8000.

## 3. Build and test the frontend

```bash
cd ~/Downloads/guardian-ai/module_10_platform/chapter_03
npm test
npm run build
```

## 4. Test the APIs

```bash
cd ~/Downloads/guardian-ai/module_10_platform/chapter_02
python3 -m pytest test_guardian_research_detail_api.py -v
python3 -u c2_detail_api_benchmark.py
```

## 5. Run the final integration checks

```bash
cd ~/Downloads/guardian-ai/module_10_platform/chapter_04
python3 -m pytest test_guardian_integration.py -v
python3 -u c4_integration_benchmark.py
```

## 6. Final scientific evaluation

```bash
cd ~/Downloads/guardian-ai/module_10_platform/chapter_05
python3 -m pytest test_guardian_final_evaluation.py -v
python3 -u guardian_final_evaluation.py
```

## Troubleshooting

ECONNREFUSED 127.0.0.1:8000:
Start the Chapter 10.2 FastAPI backend first.

HTTP 503 EVIDENCE_VERIFICATION_FAILED:
Inspect the upstream research evidence and manifests.
Do not bypass verification by inventing results.

Frontend does not start:
Check that Node.js is installed and run npm install
inside chapter_03.

## Usage restrictions

- Local research prototype only
- No public-network API exposure
- No automatic candidate promotion
- No lesson activation through the dashboard
- No Guardian permission changes through the UI
