@ECHO OFF

SET "VENV_NAME=venv"

IF EXIST "%VENV_NAME%\Scripts\activate.bat" (
  ECHO venv already exists, running...
) ELSE (
  ECHO Creating venv...
  python.exe -m venv "%VENV_NAME%"
  ECHO venv created, installing requirements...
)

CALL "%VENV_NAME%\Scripts\activate"
pip install -r requirements.txt

python.exe imageeditor.py

PAUSE
