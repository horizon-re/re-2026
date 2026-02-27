@echo off
REM Train context window ablation: n=1, n=2, n=3
REM All with full features

echo ========================================
echo Context Window Ablation Training
echo Training n=1, n=2, n=3 with full features
echo ========================================
echo.

REM ============================================================
REM n=1: Minimal context (1 sentence before/after)
REM ============================================================
echo [1/3] Training: n=1 (minimal context)
echo ----------------------------------------
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py ^
    --n-before 1 ^
    --n-after 1 ^
    --features full ^
    --epochs 50 ^
    --patience 5 ^
    --lr 1e-4 ^
    --batch-train 256

if errorlevel 1 (
    echo ERROR: n=1 training failed
    exit /b 1
)

echo.
echo √ n=1 training complete
echo.

REM ============================================================
REM n=2: Default context (2 sentences before/after)
REM ============================================================
echo [2/3] Training: n=2 (default context)
echo ----------------------------------------
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py ^
    --n-before 2 ^
    --n-after 2 ^
    --features full ^
    --epochs 50 ^
    --patience 5 ^
    --lr 1e-4 ^
    --batch-train 256

if errorlevel 1 (
    echo ERROR: n=2 training failed
    exit /b 1
)

echo.
echo √ n=2 training complete
echo.

REM ============================================================
REM n=3: Extended context (3 sentences before/after)
REM ============================================================
echo [3/3] Training: n=3 (extended context)
echo ----------------------------------------
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py ^
    --n-before 3 ^
    --n-after 3 ^
    --features full ^
    --epochs 50 ^
    --patience 5 ^
    --lr 1e-4 ^
    --batch-train 256

if errorlevel 1 (
    echo ERROR: n=3 training failed
    exit /b 1
)

echo.
echo √ n=3 training complete
echo.

echo ========================================
echo All context window training complete!
echo ========================================
echo.
echo Models saved in: classifier\models\mpnet_phase1_5_ablation\
echo.
echo Next step: Run evaluation
echo   python scripts\classifier\eval\eval_mpnet_ablation_comprehensive.py
echo.

pause