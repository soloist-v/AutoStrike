if "%~1"=="goto" (^
 if "%~2"==":uacsuccess" (goto :uacsuccess))

mshta vbscript:createobject("shell.application").shellexecute("%~s0","goto :uacsuccess ","","runas",1)(window.close) &goto :end

:uacsuccess
echo :uacsuccess

goto :end
:end
python .\auto_strike.py