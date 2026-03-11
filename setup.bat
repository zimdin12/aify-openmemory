@echo off
REM =============================================================================
REM Aify Container - Initial Setup (Windows)
REM =============================================================================

echo Aify Container Setup
echo ========================

if not exist .env (
    copy .env.example .env
    echo Created .env from template
) else (
    echo .env already exists, skipping
)

if not exist config\service.json (
    copy config\service.example.json config\service.json
    echo Created config/service.json from template
) else (
    echo config/service.json already exists, skipping
)

if not exist docker-compose.override.yml (
    copy docker-compose.override.yml.example docker-compose.override.yml
    echo Created docker-compose.override.yml from template
) else (
    echo docker-compose.override.yml already exists, skipping
)

REM Initialize git submodules if any
if exist .gitmodules (
    git submodule update --init --recursive
    echo Git submodules initialized
)

echo.
echo Setup complete! Next steps:
echo   1. Edit .env to configure your service
echo   2. Edit config\service.json for service-specific settings
echo   3. Run: docker compose up -d --build
echo   4. Test: curl http://localhost:8800/health
