:: Renommer le fichier run.ba en run.bat

@echo off

:: Run test.py to train-finetune, get predictions on test set and save my ML pipeline for the API

python test.py

:: Start the API
echo Starting FastAPI API...
start cmd /k "uvicorn app:app --host localhost --port 4000"

:: Start the Streamlit web app
echo Starting Streamlit Web Application...
start cmd /k "streamlit run streamlit_nba.py"

:: Get feedback
echo API and Streamlit are now running.
