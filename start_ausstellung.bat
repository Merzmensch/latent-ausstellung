@echo off
title Latent Ausstellung
echo.
echo  ██████████████████████████████████████████████████
echo  █  LATENT AUSSTELLUNG                            █
echo  █  http://localhost:5000/ausstellung             █
echo  ██████████████████████████████████████████████████
echo.

call C:\Users\vladi\miniconda3\Scripts\activate.bat stylegan

cd /d C:\Users\vladi\stylegan2-explorer

start "" http://localhost:5000/ausstellung

python server.py --pkl models\mem.pkl

pause
