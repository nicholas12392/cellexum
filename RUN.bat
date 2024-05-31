@ECHO OFF

CD > tmpFile
SET /p working_dir= < tmpFile
DEL tmpFile
echo Temporarily added ENVIRONMENT VARIABLE: %working_dir%

SET PATH=%PATH%;%working_dir%

python.exe image_processing.py

pause
