cd /d %~dp0
call conda activate occt
python app_split_box.py

SET /P INPUTSTR="Input Y:Restart / N:Finish: "

echo %INPUTSTR%
if "%INPUTSTR%"=="" (
    pause
) else if "%INPUTSTR%"=="Y" (
    call %~0
) else if "%INPUTSTR%"=="y" (
    call %~0
) else if "%INPUTSTR%"=="cmd" (
    exit /B 0
) else (
    pause
)
