@echo off 
virtualenv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
python auto_keras_magic.py
call venv\Scripts\deactivate.bat
