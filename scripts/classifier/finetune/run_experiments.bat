@echo off
REM ============================================================
REM Batch Experiment Runner for Multi-Task DeBERTa Training
REM ============================================================
REM
REM This file should be in: scripts/classifier/finetune/
REM It will run multiple experiments to find the best config
REM
REM ============================================================

echo ==================================================
echo Multi-Task DeBERTa Training - Batch Experiments
echo ==================================================
echo.

REM Save current directory and navigate to project root
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%..\..\..\"

echo Current directory: %CD%
echo.

REM Check if we're in the right place
if not exist "classifier\outputs\splits\holdout_60.jsonl" (
    echo ERROR: Cannot find data files!
    echo Expected to find: classifier\outputs\splits\holdout_60.jsonl
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

set HF_HUB_DISABLE_SYMLINKS_WARNING=1
REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
)

REM Configuration
set PYTHON=python
set SCRIPT=scripts\classifier\finetune\train_multitask_deberta.py
set COMMON_ARGS=--use-frnfr --use-synthetic

REM Check if training script exists
if not exist "%SCRIPT%" (
    echo ERROR: Cannot find training script!
    echo Expected to find: %SCRIPT%
    echo Current directory: %CD%
    echo.
    echo Make sure train_multitask_deberta.py is in scripts\classifier\finetune\
    echo.
    pause
    exit /b 1
)

echo Found training script: %SCRIPT%
echo.

REM Create log directory
set LOG_DIR=scripts\classifier\finetune\experiment_logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Get timestamp (Windows format)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

echo Logs will be saved to: %LOG_DIR%
echo.

echo ================================================
echo EXPERIMENT SET 1: Model Comparison
echo ================================================
echo.

echo Running: DeBERTa Baseline...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model deberta --n-context 1 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp1_deberta_baseline.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DeBERTa Baseline
) else (
    echo [FAILED] DeBERTa Baseline - Check log file
)
echo.

echo Running: MiniLM Baseline...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model minilm --n-context 1 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp1_minilm_baseline.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] MiniLM Baseline
) else (
    echo [FAILED] MiniLM Baseline - Check log file
)
echo.

echo Running: MPNet Baseline...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model mpnet --n-context 1 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp1_mpnet_baseline.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] MPNet Baseline
) else (
    echo [FAILED] MPNet Baseline - Check log file
)
echo.

echo Running: Paraphrase-MPNet Baseline...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model paraphrase-mpnet --n-context 1 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp1_paraphrase_mpnet_baseline.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Paraphrase-MPNet Baseline
) else (
    echo [FAILED] Paraphrase-MPNet Baseline - Check log file
)
echo.

echo.
echo ================================================
echo EXPERIMENT SET 2: LoRA Rank Tuning (DeBERTa)
echo ================================================
echo.

echo Running: DeBERTa LoRA-8...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model deberta --lora-r 8 --lora-alpha 16 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp2_deberta_lora8.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DeBERTa LoRA-8
) else (
    echo [FAILED] DeBERTa LoRA-8 - Check log file
)
echo.

echo Running: DeBERTa LoRA-16...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model deberta --lora-r 16 --lora-alpha 32 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp2_deberta_lora16.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DeBERTa LoRA-16
) else (
    echo [FAILED] DeBERTa LoRA-16 - Check log file
)
echo.

echo Running: DeBERTa LoRA-32...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model deberta --lora-r 32 --lora-alpha 64 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp2_deberta_lora32.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DeBERTa LoRA-32
) else (
    echo [FAILED] DeBERTa LoRA-32 - Check log file
)
echo.

echo.
echo ================================================
echo EXPERIMENT SET 3: Learning Rate Tuning (DeBERTa)
echo ================================================
echo.

echo Running: DeBERTa LR 1e-5...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model deberta --lr 1e-5 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp3_deberta_lr1e5.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DeBERTa LR 1e-5
) else (
    echo [FAILED] DeBERTa LR 1e-5 - Check log file
)
echo.

echo Running: DeBERTa LR 2e-5...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model deberta --lr 2e-5 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp3_deberta_lr2e5.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DeBERTa LR 2e-5
) else (
    echo [FAILED] DeBERTa LR 2e-5 - Check log file
)
echo.

echo Running: DeBERTa LR 3e-5...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model deberta --lr 3e-5 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp3_deberta_lr3e5.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DeBERTa LR 3e-5
) else (
    echo [FAILED] DeBERTa LR 3e-5 - Check log file
)
echo.

echo Running: DeBERTa LR 5e-5...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model deberta --lr 5e-5 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp3_deberta_lr5e5.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DeBERTa LR 5e-5
) else (
    echo [FAILED] DeBERTa LR 5e-5 - Check log file
)
echo.

echo.
echo ================================================
echo EXPERIMENT SET 4: Context Window Size (DeBERTa)
echo ================================================
echo.

echo Running: DeBERTa Context=1...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model deberta --n-context 1 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp4_deberta_context1.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DeBERTa Context=1
) else (
    echo [FAILED] DeBERTa Context=1 - Check log file
)
echo.

echo Running: DeBERTa Context=2...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model deberta --n-context 2 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp4_deberta_context2.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DeBERTa Context=2
) else (
    echo [FAILED] DeBERTa Context=2 - Check log file
)
echo.

echo Running: DeBERTa Context=3...
%PYTHON% %SCRIPT% %COMMON_ARGS% --model deberta --n-context 3 --seed 42 > "%LOG_DIR%\%TIMESTAMP%_exp4_deberta_context3.log" 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DeBERTa Context=3
) else (
    echo [FAILED] DeBERTa Context=3 - Check log file
)
echo.

echo.
echo ==================================================
echo All Experiments Completed!
echo ==================================================
echo.
echo Logs saved in: %LOG_DIR%
echo.
echo To analyze results, check the logs and compare:
echo   - Primary metric: req_f1 (Task 1 F1-score)
echo   - Secondary metrics: req_acc, func_f1, amb_f1
echo.
echo Next steps:
echo   1. Review logs to find best configuration
echo   2. Train final model with best config + multiple seeds
echo   3. Ensemble predictions from multiple seeds
echo   4. Evaluate on test set
echo.

REM Return to original directory
cd /d "%SCRIPT_DIR%"

pause