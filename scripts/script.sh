python3 -m venv blind-eyes
source blind-eyes/bin/activate
cd blind-eyes
git clone https://github.com/davisking/dlib.git
cd dlib
pip install git+https://github.com/davisking/dlib.git
cd ..
rm -rf dlib
git clone https://github.com/MBUYt0n/blind-eye-dealers.git
cd blind-eye-dealers
mkdir models
cd models
wget https://raw.githubusercontent.com/ChiragSaini/facial-emotion-detector/master/emotion_little_vgg_2.h5
wget https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5
cd ../..
pip install ultralytics tensorflow==2.15.0 opencv-python lapx paddlepaddle paddleocr
pip cache purge
deactivate