@echo off

for /l %%i in (1,1,30) do (
    setlocal enabledelayedexpansion
    set "house_num=%%i"
    if !house_num! lss 10 (
        set "house_num=0!house_num!"
    )
    curl.exe -X POST "http://localhost:6333//collections/House_!house_num!_online_AD"/snapshots
    endlocal
)