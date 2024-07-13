@echo off

REM Initialize Python virtual environment
python -m venv blind-eyes

REM Activate the virtual environment
call blind-eyes\Scripts\activate
cd blind-eyes

REM Upgrade pip to the latest version
python -m pip install --upgrade pip


pip install ultralytics tensorflow opencv-python paddlepaddle paddleocr lapx 
pip cache purge
git clone https://github.com/MBUYt0n/blind-eye-dealers.git
git clone https://github.com/davisking/dlib.git
cd dlib 
pip install git+https://github.com/davisking/dlib.git
cd ..
rd /s /q dlib

cd blind-eye-dealers
mkdir models
cd models

curl -L -O https://raw.githubusercontent.com/ChiragSaini/facial-emotion-detector/master/emotion_little_vgg_2.h5
curl -L -O https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5

deactivate

echo Virtual environment setup and repository clone completed!
pause
exit