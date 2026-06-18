# Tuner UI

Tuner UI is a full-stack interface for interactive model tuning workflows. It
combines a Next.js workspace with a FastAPI backend for dataset management,
training run orchestration, checkpoint tracking, chat testing, model registry
workflows, and Hugging Face deployment.

The project is being prepared as supporting material for the Thinking Machines
Lab Interactivity Research Grants application:

- Application instructions: https://thinkingmachines.ai/news/interactivity-research-grants/apply/
- Program terms: https://thinkingmachines.ai/legal/interactivity-research-grants-terms/

## Research Grant Fit

The grant program evaluates proposals for relevance to multimodal or real-time
interactivity, feasibility, construct validity, and simplicity/generalizability.
This repo is positioned as an applied research platform for human-in-the-loop
model tuning:

- Real-time training feedback: run status, logs, metrics, checkpoints, and
  model registry state are exposed through an operator-facing UI.
- Interactive iteration loop: upload or register datasets, launch tuning runs,
  inspect progress, chat with base or registered models, and deploy artifacts.
- Reproducible workflows: API schemas and tests make the system reviewable
  without private credentials.
- Real Tinker behavior: when `TINKER_API_KEY` is configured, the backend
  requires the real Tinker/cookbook stack and refuses to silently simulate.

For the grant proposal, this repository can support a 6-month agenda around
interactive fine-tuning UX, live evaluation signals, dataset quality loops,
checkpoint comparison, and operator-in-the-loop model improvement.

## Current Status

This codebase has been audited and repaired so it can run locally without a
Tinker API key while preserving the real Tinker path for later use.

Verified locally:

- Backend tests: `217 passed, 7 skipped`
- Frontend production build: passed
- Frontend TypeScript check: passed
- Frontend lint: passed with warnings
- No-Tinker smoke flow: health, project creation, inline JSONL dataset upload,
  run creation, and simulated progress all pass

The 7 skipped backend tests are live-server E2E checks. Run them only after
starting the backend and setting `RUN_LIVE_E2E=1`.

## Features

### Training Workflows

- Supported recipes: SFT, DPO, RL, PPO, GRPO, Distillation, Chat SL,
  Preference, Tool Use, Multiplayer RL, Math RL, Evaluation, and Sampling
- LoRA-oriented hyperparameter support
- Automatic hyperparameter recommendations
- Run status, progress, logs, metrics, and checkpoint tracking
- Optional no-key preview mode for local UI review
- Real Tinker/cookbook execution when dependencies and `TINKER_API_KEY` exist
- Hard failure instead of simulation if a real key is configured but Tinker is
  not installed correctly

### Dataset Management

- Register Hugging Face datasets
- Upload browser-provided JSONL rows
- Persist inline JSONL data to `artifacts/datasets/*.jsonl`
- Preview and validate datasets
- Supports Alpaca-style and chat-message-style training rows

### Model Testing

- Chat with base models
- Chat with registered/fine-tuned models
- Model selector distinguishes base models from registered models
- Sampling/chat endpoints remain callable in local simulation mode

### Model Registry and Deployment

- Register custom or trained models
- Track model metadata, project links, and run links
- Store Hugging Face tokens encrypted when `ENCRYPTION_KEY` is configured
- Deploy checkpoints to Hugging Face when real artifacts are available

## Requirements

- Python 3.11+
- Node.js 18+
- pnpm

Recommended:

```powershell
npm install -g pnpm
```

## Installation

### Backend

```powershell
cd C:\Users\ASUS\Desktop\tuner-ui\backend
python -m pip install -r requirements.txt
python -m pip install -r tests\requirements-test.txt
```

### Frontend

```powershell
cd C:\Users\ASUS\Desktop\tuner-ui\frontend
pnpm.cmd install
```

## Running Locally

Use two terminals.

Terminal 1, backend:

```powershell
cd C:\Users\ASUS\Desktop\tuner-ui\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Terminal 2, frontend:

```powershell
cd C:\Users\ASUS\Desktop\tuner-ui\frontend
pnpm.cmd dev --hostname 127.0.0.1 --port 3000
```

Open:

```text
http://127.0.0.1:3000
```

## Configuration

### No-Key Local Preview Mode

No Tinker key is required only for local UI review and backend smoke testing.
By default:

```powershell
$env:ALLOW_ANON="true"
```

The backend will start and run local preview/smoke flows. This mode is not the
real training path.

### Real Tinker Mode

Install the real Tinker/cookbook stack, then set:

```powershell
$env:TINKER_API_KEY="your_real_tinker_api_key"
$env:ALLOW_ANON="false"
```

Optional frontend environment:

```powershell
$env:NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8000"
$env:NEXT_PUBLIC_TINKER_API_KEY="your_real_tinker_api_key"
```

When `TINKER_API_KEY` is configured, the backend requires the real Tinker
package and cookbook stack. If those imports fail, startup/training fails
loudly instead of falling back to simulation.

### Hugging Face Token Encryption

Set `ENCRYPTION_KEY` before storing Hugging Face tokens:

```powershell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
$env:ENCRYPTION_KEY="generated_key_here"
```

## Testing

Backend:

```powershell
cd C:\Users\ASUS\Desktop\tuner-ui
python -m pytest backend\tests -q
```

Live backend E2E tests:

```powershell
cd C:\Users\ASUS\Desktop\tuner-ui
$env:RUN_LIVE_E2E="1"
python -m pytest backend\tests\test_hyperparam_e2e.py backend\tests\test_hyperparam_manual.py -q
```

Frontend:

```powershell
cd C:\Users\ASUS\Desktop\tuner-ui\frontend
.\node_modules\.bin\tsc.CMD --noEmit
pnpm.cmd lint
pnpm.cmd build
```

## Project Structure

```text
tuner-ui/
  backend/
    main.py                  FastAPI app and API routes
    models.py                SQLAlchemy models
    schemas.py               Pydantic request/response schemas
    job_runner.py            Training runner and no-key local preview fallback
    utils/                   Helpers for metrics, files, encryption, recipes
    tests/                   Backend regression tests
  frontend/
    app/                     Next.js app routes
    components/              UI components and screens
    lib/api.ts               Shared API client
    lib/utils.ts             UI utility helpers
  artifacts/
    datasets/                Local JSONL datasets generated from uploads
  docker-compose.yml
  docker-compose.dev.yml
```

## Notes for the Grant Proposal

The application materials should include a 1-3 page project summary, a 1-page
budget, PI/contributor CVs, organization/location details, tax/admin details if
applicable, and a primary contact email.

The published deadline is June 19, 2026 at 11:59 PM PDT. The proposal timeline
can cover up to 6 months. The program terms state that accepted participants are
eligible for a USD 100,000 grant, with institutional indirect costs capped at
10% of the cash grant amount for applicable organizations.

The proposal should not include confidential third-party information. The terms
state that proposal materials and work product are not treated as confidential,
and publication of work product that uses or references company materials
requires prior written approval.

## Troubleshooting

### Backend starts in no-key preview mode

This is expected if Tinker packages are not installed or no API key is set.
This is only for local review. If `TINKER_API_KEY` is set, the backend must use
real Tinker dependencies or fail.

### Frontend cannot reach backend

Confirm the backend is running:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Confirm the frontend API base URL points to the backend:

```powershell
$env:NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8000"
```

### Token storage warning

Set `ENCRYPTION_KEY` before saving Hugging Face tokens. Without it, the backend
generates a temporary key for the current process only.

### Live E2E tests fail with connection refused

Start the backend first, then set `RUN_LIVE_E2E=1`.

## License

See [LICENSE](LICENSE).
