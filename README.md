<div align="center">
  <img src="frontend/public/favicon.png" alt="Tinker Platform Logo" width="80" height="80" style="border-radius: 16px; box-shadow: 0 8px 16px rgba(0,0,0,0.15); border: 2px solid #e5e7eb;">
  <p><strong>Tinker UI</strong></p>
  <p><em>A full-stack platform for fine-tuning and training AI models, featuring a modern web UI and powerful backend API.</em></p>

  ![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
  ![Node](https://img.shields.io/badge/node-18+-green.svg)
  ![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)
  ![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)
  ![Tests](https://img.shields.io/badge/tests-229%20total-brightgreen.svg)
  ![License](https://img.shields.io/badge/license-MIT-blue.svg)
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
Watch the complete demo: https://www.youtube.com/watch?v=f2bYUzlbcAY

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
Before getting started, ensure you have the following installed:
- **Node.js** (version 18 or higher) - [Download](https://nodejs.org/)
- **Python** (version 3.11 or higher) - [Download](https://python.org/)
- **pnpm** (package manager) - Install with `npm install -g pnpm`
- **Git** - [Download](https://git-scm.com/)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/tinker-ui.git
cd tinker-ui
```

### 2. Backend Setup
Navigate to the backend directory and set up the Python environment:
```bash
cd backend
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Frontend Setup
Navigate to the frontend directory and install dependencies:
```bash
cd ../frontend
pnpm install
```

## Configuration

### Backend Configuration
Create a `.env` file in the backend directory:
```bash
# backend/.env
TINKER_API_KEY=your_tinker_api_key_here
DATABASE_URL=sqlite:///./tinker_platform.db
ALLOW_ANON=true

# Generate encryption key with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
ENCRYPTION_KEY=your_encryption_key_here
```

### Frontend Configuration
Create a `.env.local` file in the frontend directory:
```bash
# frontend/.env.local
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_TINKER_API_KEY=your_tinker_api_key_here
```

## Running the Application

### 1. Start the Backend
In the backend directory:
```bash
cd backend
uvicorn main:app --reload
```
The backend will be available at http://127.0.0.1:8000

### 2. Start the Frontend
In a new terminal, navigate to the frontend directory:
```bash
cd frontend
pnpm dev
```
The frontend will be available at http://localhost:3000

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

For detailed instructions, see [docs/HUGGINGFACE_DEPLOYMENT.md](docs/HUGGINGFACE_DEPLOYMENT.md)

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
- See [docs/PROGRESS_BAR_FIX.md](docs/PROGRESS_BAR_FIX.md) for details on progress tracking

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

## Project Structure

```
tinker-ui/
├── backend/                # FastAPI backend
│   ├── main.py            # Main API application
│   ├── job_runner.py      # Background job execution
│   ├── models.py          # SQLAlchemy models
│   ├── tinker_cookbook/   # Training recipes
│   ├── tests/             # Backend test suite
│   └── requirements.txt   # Python dependencies
├── frontend/              # Next.js frontend
│   ├── app/              # Next.js 16 app directory
│   ├── components/       # React components
│   ├── lib/              # Utility functions
│   ├── tests/            # Frontend test suite
│   └── package.json      # Node dependencies
├── docs/                 # Documentation
├── TESTING_SUMMARY.md    # Testing guide
├── TEST_RESULTS.md       # Test results
└── README.md            # This file
```

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

- **[docs/HUGGINGFACE_DEPLOYMENT.md](docs/HUGGINGFACE_DEPLOYMENT.md)** - HuggingFace deployment guide
- **[TESTING_SUMMARY.md](TESTING_SUMMARY.md)** - Complete testing documentation
- **[TEST_RESULTS.md](TEST_RESULTS.md)** - Detailed test results and analysis
- **[docs/PROGRESS_BAR_FIX.md](docs/PROGRESS_BAR_FIX.md)** - Progress tracking implementation details
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
    <a href="https://github.com/YOUR_USERNAME/tinker-ui/issues">Report Bug</a>
    ·
    <a href="https://github.com/YOUR_USERNAME/tinker-ui/issues">Request Feature</a>
  </p>
</div>
