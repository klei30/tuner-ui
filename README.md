# Tuner UI

![Stars](https://img.shields.io/github/stars/klei30/tuner-ui)
![Forks](https://img.shields.io/github/forks/klei30/tuner-ui)
![Last Commit](https://img.shields.io/github/last-commit/klei30/tuner-ui)
![Contributors](https://img.shields.io/github/contributors/klei30/tuner-ui)

Tuner UI is an open-source interface for running and inspecting model tuning
workflows with Tinker. It combines a Next.js workspace with a FastAPI backend
for datasets, training recipes, run monitoring, checkpoints, model testing, and
Hugging Face deployment.

## Repository Signals

GitHub traffic analytics are private to the repository owner. For public
evidence, use the repository badges above and a screenshot of the GitHub
`Insights → Traffic` page when you need clone and view counts.

> Real training requires a valid `TINKER_API_KEY` and access to the Tinker SDK.
> The SDK package is not distributed in this repository; the repository
> includes the application-side cookbook integration.

## What It Does

- Registers Hugging Face datasets or uploaded JSONL data.
- Configures SFT, DPO, RL, distillation, Math RL, and related recipes.
- Recommends learning rates, batch sizes, and LoRA parameters.
- Starts and monitors training runs from one interface.
- Tracks progress, metrics, logs, checkpoints, and failures.
- Tests base and registered models through chat and sampling workflows.
- Registers trained models and publishes available artifacts to Hugging Face.

## Architecture

```text
Browser
  |
  v
Next.js frontend (port 3000)
  |
  v
FastAPI backend (port 8000)
  |-- SQLAlchemy database
  |-- Tinker SDK and cookbook
  |-- local dataset/checkpoint artifacts
  `-- Hugging Face integration
```

The frontend calls the FastAPI API. The backend validates requests, persists
project state, launches recipe execution, records training output, and exposes
the resulting metrics and artifacts to the UI.

## Requirements

- Python 3.11
- Node.js 18 or newer
- pnpm
- A valid Tinker API key
- Access to the Tinker SDK
- A Hugging Face token for private datasets or Hub deployment

Install pnpm if needed:

```powershell
npm install -g pnpm
```

## Installation

```powershell
git clone https://github.com/klei30/tuner-ui.git
cd tuner-ui
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip install -r backend\tests\requirements-test.txt
```

Install the Tinker SDK using the installation method provided with your Tinker
access. Confirm Python can import the SDK and bundled cookbook integration:

```powershell
cd backend
python -c "import tinker, tinker_cookbook; print('Tinker dependencies available')"
cd ..
```

Install frontend dependencies:

```powershell
cd frontend
pnpm install
cd ..
```

## Configuration

Set backend environment variables:

```powershell
$env:TINKER_API_KEY="your_tinker_api_key"
$env:ALLOW_ANON="false"
$env:DATABASE_URL="sqlite:///tinker_platform.db"
```

If the cookbook is outside the repository:

```powershell
$env:COOKBOOK_PATH="C:\path\to\tinker-cookbook"
```

Optional Hugging Face configuration:

```powershell
$env:HF_TOKEN="hf_your_token"
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
$env:ENCRYPTION_KEY="generated_key"
```

Configure `frontend\.env.local`:

```dotenv
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_TINKER_API_KEY=your_tinker_api_key
```

Do not commit API keys, Hugging Face tokens, or encryption keys.

## Run Locally

Start the backend:

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

In a second terminal, start the frontend:

```powershell
cd frontend
pnpm dev --hostname 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000`.

Check backend health:

```powershell
curl.exe http://127.0.0.1:8000/health
```

## Typical Workflow

1. Create a project.
2. Register a Hugging Face dataset or upload JSONL examples.
3. Choose a model and training recipe.
4. Review or override the recommended hyperparameters.
5. Start the run and inspect logs, metrics, and progress.
6. Review generated checkpoints and test model behavior.
7. Register or deploy the selected model artifact.

## Dataset Formats

Alpaca-style JSONL:

```json
{"instruction":"Translate to Italian","input":"Good morning","output":"Buongiorno"}
```

Chat messages:

```json
{"messages":[{"role":"user","content":"Say hello"},{"role":"assistant","content":"Hello"}]}
```

Uploaded rows are persisted under `artifacts/datasets/`.

## Testing

Run backend lint and tests from the repository root:

```powershell
cd backend
ruff check .
cd ..
python -m pytest backend
```

Current verified result:

```text
218 passed, 7 skipped
```

The skipped tests require a running backend:

```powershell
$env:RUN_LIVE_E2E="1"
python -m pytest backend\tests\test_hyperparam_e2e.py backend\tests\test_hyperparam_manual.py
```

Run frontend checks:

```powershell
cd frontend
pnpm lint
pnpm exec tsc --noEmit
pnpm build
```

The automated backend tests use test doubles and do not start paid Tinker
training jobs.

## Docker

Start PostgreSQL and Redis for local development:

```powershell
docker compose -f docker-compose.dev.yml up -d
```

The complete stack requires `TINKER_API_KEY`:

```powershell
docker compose up --build
```

## Project Structure

```text
tuner-ui/
  backend/
    main.py                 FastAPI application and routes
    job_runner.py           Training orchestration
    models.py               SQLAlchemy models
    schemas.py              API schemas
    recipes/                Recipe configuration
    tinker_cookbook/        Cookbook integration code
    utils/                  Environment, execution, and security helpers
    tests/                  Backend test suite
  frontend/
    app/                    Next.js routes
    components/             Screens and reusable UI
    lib/api.ts              API client
  artifacts/
    datasets/               Uploaded local datasets
  docker-compose.yml
  docker-compose.dev.yml
```

## Troubleshooting

### Tinker imports fail

Confirm the SDK and cookbook are installed in the Python environment used to
run FastAPI:

```powershell
cd backend
python -c "import tinker, tinker_cookbook"
```

If `TINKER_API_KEY` is configured but these imports fail, the backend stops
instead of silently substituting a different training path.

### Requests return 401

Confirm `ALLOW_ANON=false`, the backend has `TINKER_API_KEY`, and the frontend
uses the matching `NEXT_PUBLIC_TINKER_API_KEY`.

### Frontend cannot reach the API

Check `NEXT_PUBLIC_API_BASE_URL`, then verify:

```powershell
curl.exe http://127.0.0.1:8000/health
```

### Hugging Face operations fail

Set a token with the required repository and dataset permissions. Set a stable
`ENCRYPTION_KEY` before saving tokens through the UI.

## License

See [LICENSE](LICENSE).
