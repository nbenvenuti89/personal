@echo on
REM BI KPIs Alert - Back Propagation for Spinners_HAU

for /f "tokens=1-8 delims=::./ " %%A in ('echo %DATE% %TIME%') do set  FileDateTime=%%B-%%C-%%D_%%E-%%F
set logfile="C:\Users\niccolo\Desktop\BI_Alert_System\New_Release_v4\BackPropagation\Log\Back_Propagation_Spinners_HAU_%FileDateTime%_log.txt"
set root=C:\Users\niccolo\miniconda3


call %root%\Scripts\activate.bat %root%
call "python" "C:\Users\niccolo\Desktop\BI_Alert_System\New_Release_v4\BackPropagation\Script\MagnumBI_Alert_v2.0.py" > %logfile% %Spinners_HAU %0

