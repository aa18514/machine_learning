@echo off 
::set up virtual environment folder
virtualenv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
python auto_keras_magic.py
call venv\Scripts\deactivate.bat
::remove virtual environment folder
rm -rf venv
