@ECHO OFF

SET "VENV_NAME=venv_python310"
SET "USER_NAME=%USERNAME%"
SET "PYTHON_PATH=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe"

IF EXIST "%VENV_NAME%\Scripts\activate.bat" (
  ECHO venv already exists, running...
) ELSE (
  ECHO Creating venv...
  "%PYTHON_PATH%" -m venv "%VENV_NAME%"
  ECHO venv created, installing requirements...
)

CALL "%VENV_NAME%\Scripts\activate"
pip install -r requirements.txt
pip install -r requirements_exe.txt
pyinstaller --onefile --name=CoskImageEditor --path=./"%VENV_NAME%"\Lib\site-packages --add-data=./"%VENV_NAME%"\Lib\site-packages\tkinterdnd2;tkinterdnd2 imageeditor.py
xcopy /s /i dll dist\dll
xcopy /s /i gui_images dist\gui_images

PAUSE
