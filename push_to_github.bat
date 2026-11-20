@echo off
title Pushing to GitHub...
echo ========================================================
echo   Pushing Image Deblurring Project to GitHub
echo   Repository: https://github.com/MALLIK-JI/FOCUS---Network-Aware-Image-Deblurring
echo ========================================================
echo.

cd /d "C:\Users\TANISHK RAJ\OneDrive\Desktop\kanix\Image_Deblurring_Project"

echo Running: git push -u origin main
echo (If a browser window or login prompt appears, please sign in to authorize)
echo.

git push -u origin main

echo.
echo ========================================================
echo Push complete! Press any key to close this window.
echo ========================================================
pause
