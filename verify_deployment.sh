#!/bin/bash
# Tuner-UI Production Deployment Verification Script

echo "=========================================="
echo "TUNER-UI PRODUCTION DEPLOYMENT VERIFICATION"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

check_step() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[PASS]${NC} $1"
        ((PASSED++))
    else
        echo -e "${RED}[FAIL]${NC} $1"
        ((FAILED++))
    fi
}

# 1. Check file structure
echo "1. Checking file structure..."
[ -f "docker-compose.yml" ] && echo "  - docker-compose.yml exists"
check_step "Docker Compose configuration"

[ -f "backend/Dockerfile" ] && echo "  - backend/Dockerfile exists"
check_step "Backend Dockerfile"

[ -f "frontend/Dockerfile" ] && echo "  - frontend/Dockerfile exists"
check_step "Frontend Dockerfile"

[ -f "backend/celery_app.py" ] && echo "  - celery_app.py exists"
check_step "Celery configuration"

[ -f "backend/tasks.py" ] && echo "  - tasks.py exists"
check_step "Background tasks"

echo ""

# 2. Check Python dependencies
echo "2. Checking Python environment..."
cd backend
python -c "import fastapi, uvicorn, sqlalchemy, alembic, celery" 2>/dev/null
check_step "Core Python dependencies"

python -c "import loguru, sentry_sdk" 2>/dev/null
check_step "Monitoring dependencies"

echo ""

# 3. Check database
echo "3. Checking database..."
python -c "from database import engine; from sqlalchemy import inspect; insp = inspect(engine); assert 'runs' in insp.get_table_names()" 2>/dev/null
check_step "Database tables exist"

python -c "from database import engine; from sqlalchemy import inspect; insp = inspect(engine); cols = [c['name'] for c in insp.get_columns('runs')]; assert 'celery_task_id' in cols" 2>/dev/null
check_step "Database schema updated (celery_task_id)"

echo ""

# 4. Check migrations
echo "4. Checking migrations..."
[ -d "alembic/versions" ] && echo "  - Migration directory exists"
check_step "Alembic migrations directory"

python -m alembic current 2>/dev/null | grep -q "002"
check_step "Database at latest migration"

echo ""

# 5. Check configuration
echo "5. Checking configuration..."
[ -f ".env.example" ] && echo "  - .env.example exists"
check_step "Environment example file"

python -c "from config import settings; assert settings.database_url is not None" 2>/dev/null
check_step "Configuration loading"

cd ..

echo ""

# 6. Check frontend
echo "6. Checking frontend..."
cd frontend
[ -f "lib/api.ts" ] && echo "  - API client exists"
check_step "Frontend API client"

[ -f "lib/types.ts" ] && echo "  - Type definitions exist"
check_step "Frontend types"

pnpm build >/dev/null 2>&1
check_step "Frontend builds successfully"

cd ..

echo ""

# 7. Check documentation
echo "7. Checking documentation..."
[ -f "docs/DEPLOYMENT_GUIDE.md" ] && echo "  - Deployment guide exists"
check_step "Deployment documentation"

[ -f "PRODUCTION_DEPLOYMENT_COMPLETE.md" ] && echo "  - Completion summary exists"
check_step "Implementation summary"

echo ""

# Summary
echo "=========================================="
echo "VERIFICATION SUMMARY"
echo "=========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
else
    echo -e "${GREEN}Failed: 0${NC}"
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS]${NC} All checks passed! Ready for deployment."
    echo ""
    echo "Next steps:"
    echo "  1. Configure environment: cp backend/.env.example backend/.env"
    echo "  2. Set production secrets in backend/.env"
    echo "  3. Deploy: docker-compose up -d"
    echo "  4. Run migrations: docker-compose exec backend alembic upgrade head"
    echo "  5. Access: http://localhost:3000"
else
    echo -e "${YELLOW}[WARNING]${NC} Some checks failed. Review above for details."
fi

exit $FAILED
