@echo off
REM Remove all collections from House_1_online_AD to House_30_online_AD using curl.exe

REM Loop from 1 to 30
for /l %%i in (1,1,30) do (
    setlocal enabledelayedexpansion
    set "house_num=%%i"
    if !house_num! lss 10 (
        set "house_num=0!house_num!"
    )
    curl.exe -X DELETE "http://localhost:6333/collections/House_!house_num!_online"
    curl.exe -X DELETE "http://localhost:6333/collections/House_!house_num!_online_AD"
    endlocal
)