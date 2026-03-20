@echo off
echo ======================================================================
echo SEIZURE DATASET COMPREHENSIVE VALIDATION
echo ======================================================================
echo.

echo [1/2] Validating Seizure Test Dataset...
echo ----------------------------------------------------------------------
python scripts/validate_seizure_dataset.py
echo.
echo.

echo [2/2] Analyzing Seizure Train/Val/Test Splits...
echo ----------------------------------------------------------------------
python scripts/analyze_seizure_splits.py
echo.
echo.

echo ======================================================================
echo VALIDATION COMPLETE
echo ======================================================================
echo.
echo Results saved to:
echo   - seizure_results/data_validation/
echo   - seizure_results/splits_analysis/
echo.
pause

