<div align="center">
  <img src="frontend/public/favicon.png" alt="Tinker Platform Logo" width="80" height="80" style="border-radius: 16px; box-shadow: 0 8px 16px rgba(0,0,0,0.15); border: 2px solid #e5e7eb;">
  <p><strong>Tuner UI</strong></p>
  <p><em>A full-stack platform for fine-tuning and training AI models, featuring a modern web UI and powerful backend API.</em></p>

  ![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
  ![Node](https://img.shields.io/badge/node-18+-green.svg)
  ![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)
  ![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)
  ![Tests](https://img.shields.io/badge/tests-229%20total-brightgreen.svg)
</div>

## 🚨 Vibe Code Alert
This project was 99% vibe coded as a fun Saturday hack to explore the Tinker Cookbook and see how quickly a full-featured training platform could be built. The result? A functional web UI that makes fine-tuning LLMs as easy as clicking a few buttons. No overthinking, just pure flow state coding.

## 📋 Table of Contents
- [Demo](#demo-video)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Demo Video
Watch the complete demo: https://www.youtube.com/watch?v=qdnSWMPZri8

## UI Preview
![Tinker Platform Dashboard](frontend/public/dashboard.png)

## Features

### 🎯 Model Training
- **Multi-Model Support**: Llama, Qwen, DeepSeek architectures
- **Training Recipes**: SFT, DPO, RL, Distillation, Chat SL, Math RL, On-Policy Distillation
- **LoRA Fine-tuning**: Efficient parameter-efficient training
- **Real-time Monitoring**: Live progress tracking with metrics
- **Auto Hyperparameters**: Intelligent parameter suggestions based on model size

### 📊 Dataset Management
- **JSONL Upload**: Direct dataset file upload with validation
- **HuggingFace Integration**: Seamless dataset importing
- **Data Preview**: Interactive dataset exploration
- **Format Conversion**: Support for Alpaca and multi-turn conversation formats
- **Format Detection**: Automatic dataset format identification

### 💬 Model Testing & Chat
- **Interactive Chat**: Test models with real-time conversations
- **Model Comparison**: Side-by-side evaluation tools
- **Inference API**: Direct model querying capabilities
- **Checkpoint Downloads**: Export trained model weights
- **Evaluation Suite**: Comprehensive model testing with custom prompts

### 🚀 HuggingFace Deployment
- **One-Click Deploy**: Deploy trained models to HuggingFace Hub with a single click
- **Secure Token Management**: Encrypted storage of HuggingFace API tokens
- **Auto Model Cards**: Automatically generated model cards with training details
- **Public/Private Repos**: Choose repository visibility
- **LoRA Weight Merging**: Option to merge LoRA weights with base model
- **Deployment Dashboard**: Track all deployments with status monitoring
- **Direct Links**: Quick access to your models on HuggingFace Hub

### 🗂️ Project Organization
- **Workspace Management**: Project-based organization
- **Run History**: Complete training run tracking
- **Model Registry**: Versioned model catalog
- **Metrics & Logs**: Detailed training metrics and logs
- **Cost Estimation**: Training cost calculations

## Prerequisites

**For Local Development:**
- **Python** 3.11+ - [Download](https://python.org/)
- **Node.js** 18+ - [Download](https://nodejs.org/)
- **pnpm** - Install with `npm install -g pnpm`

**For Docker Deployment:**
- **Docker** 24+ & Docker Compose V2 - [Download](https://docker.com/)

## Quick Start

### Option 1: Local Development (Recommended for Development)

#### 1. Clone & Configure
```bash
git clone https://github.com/klei30/tuner-ui.git
cd tuner-ui

# Backend config
cp backend/.env.example backend/.env
# Edit backend/.env and add your TINKER_API_KEY

# Frontend config
cp frontend/.env.example frontend/.env.local
```

#### 2. Setup Backend
```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

#### 3. Setup Frontend
```bash
cd ../frontend
pnpm install
```

#### 4. Run (2 terminals)
```bash
# Terminal 1 - Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
pnpm dev
```

**Access:** http://localhost:3000

---

### Option 2: Docker (Recommended for Production)

#### Quick Start with Docker
```bash
git clone https://github.com/klei30/tuner-ui.git
cd tuner-ui

# Configure environment
cp backend/.env.example backend/.env
# Edit backend/.env with your TINKER_API_KEY

# Start infrastructure (PostgreSQL + Redis)
docker-compose -f docker-compose.dev.yml up -d

# Run backend locally (faster iteration)
cd backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Run frontend locally (separate terminal)
cd frontend && pnpm install && pnpm dev
```

#### Full Docker Deployment
```bash
# Create root .env file for Docker Compose
cat > .env << 'EOF'
POSTGRES_PASSWORD=your_secure_password
SECRET_KEY=your_32_char_secret_key_here_min
ENCRYPTION_KEY=your_fernet_key_here
TINKER_API_KEY=your_tinker_api_key
EOF

# Build and start all services
docker-compose up -d --build

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

**Access:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Configuration

### Environment Files

| File | Purpose |
|------|---------|
| `backend/.env` | Backend configuration (create from `.env.example`) |
| `frontend/.env.local` | Frontend configuration (create from `.env.example`) |
| `.env` (root) | Docker Compose variables (only for Docker deployment) |

### Required Variables

**Backend (`backend/.env`):**
```bash
TINKER_API_KEY=your_tinker_api_key    # Required - Get from Tinker platform
DATABASE_URL=sqlite:///./tuner_ui.db  # SQLite for dev, PostgreSQL for prod
ALLOW_ANON=true                        # Set to false in production
ENCRYPTION_KEY=your_fernet_key         # Generate with command below
```

**Frontend (`frontend/.env.local`):**
```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_TINKER_API_KEY=your_tinker_api_key
```

### Generate Encryption Key
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

---

## Project Structure

```
tuner-ui/
├── backend/                 # FastAPI backend
│   ├── main.py             # API application
│   ├── models.py           # Database models
│   ├── celery_app.py       # Background task queue
│   ├── alembic/            # Database migrations
│   ├── Dockerfile          # Backend container
│   └── .env.example        # Config template
├── frontend/               # Next.js frontend
│   ├── app/                # Next.js pages
│   ├── lib/api.ts          # API client
│   ├── Dockerfile          # Frontend container
│   └── .env.example        # Config template
├── nginx/                  # Reverse proxy config
├── docker-compose.yml      # Full stack deployment
├── docker-compose.dev.yml  # Dev infrastructure only
└── docs/                   # Documentation
```

## HuggingFace Deployment Setup

### Quick Start
1. **Generate encryption key:**
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

2. **Add to backend/.env:**
   ```bash
   ENCRYPTION_KEY=your-generated-key-here
   ```

3. **Get HuggingFace Token:**
   - Visit https://huggingface.co/settings/tokens
   - Create a new token with **write** permissions
   - Copy the token (starts with `hf_`)

4. **Connect in UI:**
   - Navigate to Settings page (http://localhost:3000/settings)
   - Paste your HuggingFace token
   - Click "Connect HuggingFace"

5. **Deploy Models:**
   - Complete a training run
   - Click "Deploy to HuggingFace" on any checkpoint
   - Configure repository settings
   - Click "Deploy" - your model will be live on HuggingFace Hub!

For more details, see the HuggingFace documentation at https://huggingface.co/docs

## Testing

This project includes comprehensive test suites for both backend and frontend.

### Backend Tests (191+ tests)

The backend includes extensive test coverage across:
- **API Endpoints** (31 tests): All FastAPI endpoints
- **Training Workflows** (19 tests): SFT, DPO, RL, and all recipe types
- **Dataset Processing** (29 tests): Format detection and validation
- **Checkpoint Management** (27 tests): Lifecycle and storage
- **Model Evaluation** (27 tests): Evaluation and metrics
- **Utility Functions** (38 tests): Text processing and helpers
- **Job Runner** (20+ tests): Background job execution

#### Running Backend Tests
```bash
cd backend

# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m e2e          # End-to-end tests only

# Run specific test file
pytest tests/test_api_endpoints.py

# Run with verbose output
pytest -v
```

For more details, see [backend/tests/README.md](backend/tests/README.md)

### Frontend Tests (5 tests)

Frontend tests using Vitest and React Testing Library:
- **Hyperparameter Calculator** (5 tests): Component rendering and interactions

#### Running Frontend Tests
```bash
cd frontend/tests

# Install dependencies (first time only)
pnpm install

# Run all tests
pnpm test:full

# Run tests in watch mode
pnpm test

# Run with UI
pnpm test:ui

# Run specific test file
pnpm test:run simple_tests.test.ts
```

### Test Statistics
- **Total Tests**: 229
- **Backend Tests**: 191+ (Unit, Integration, E2E)
- **Frontend Tests**: 5
- **Test Coverage**: Comprehensive coverage of core functionality
- **Success Rate**: ~82% (with known fixture issues being addressed)

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
If port 8000 or 3000 is already in use:
- For backend: Change the port in the uvicorn command: `--port 8001`
- For frontend: It will automatically use the next available port (e.g., 3001)
- Update the `NEXT_PUBLIC_API_BASE_URL` in `frontend/.env.local` to match the backend port

#### 2. Frontend Fails to Fetch Data
If the frontend shows "Failed to fetch" or no models load:
- Ensure the backend is running on the correct host/port
- Check that `NEXT_PUBLIC_API_BASE_URL` matches the backend URL
- Verify the API key in both .env files
- Try restarting both services

#### 3. Backend Import Errors
If you see import errors in the backend:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the virtual environment
- Some optional dependencies may not be available (like tinker, llm)

#### 4. Frontend Compilation Errors
If the frontend fails to compile:
- Ensure all dependencies are installed: `pnpm install`
- Clear the Next.js cache: `rm -rf .next` (or `rd /s /q .next` on Windows)
- Restart the dev server

#### 5. Training Logs Not Displaying
If training starts but logs don't appear in the UI:
- Ensure the backend can write to the `artifacts/` directory
- Check the browser console for any fetch errors
- Verify the backend is running and accessible

#### 6. IPv4/IPv6 Resolution Issues
If you experience connection issues:
- Use `127.0.0.1` instead of `localhost` for `API_BASE_URL`
- Ensure backend is bound to `127.0.0.1` not `0.0.0.0`

#### 7. Test Failures
If tests fail:
- Ensure all test dependencies are installed
- Check environment variables are set correctly
- See test documentation for specific requirements
- Some tests may require TINKER_API_KEY environment variable

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest` (backend) and `pnpm test:full` (frontend)
5. Commit with descriptive messages
6. Push to your fork
7. Create a Pull Request

## Documentation

- **API Documentation**: http://localhost:8000/docs (Swagger UI when running)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[backend/tests/README.md](backend/tests/README.md)** - Backend testing guide

## Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: SQL toolkit and ORM
- **Pydantic**: Data validation using Python type hints
- **Pytest**: Testing framework with extensive fixtures
- **Ruff**: Fast Python linter and formatter

### Frontend
- **Next.js 16**: React framework with App Router
- **React 19**: Latest React with server components
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Radix UI**: Accessible component primitives
- **Vitest**: Fast unit testing framework

### ML/Training
- **Tinker Cookbook**: Training recipes and utilities
- **HuggingFace**: Dataset and model integration
- **LoRA**: Parameter-efficient fine-tuning

## Roadmap

- [ ] on progess


## Acknowledgments

- Built with the [Tinker Cookbook](https://github.com/thinkingmachines/tinker)
- Inspired by modern ML training platforms
- Community contributions and feedback

---

<div align="center">
  <p>Made with ❤️ by the community</p>
  <p>
    <a href="https://github.com/YOUR_USERNAME/tuner-ui/issues">Report Bug</a>
    ·
    <a href="https://github.com/YOUR_USERNAME/tuner-ui/issues">Request Feature</a>
  </p>
</div>
