# HRNet
Implementation of HRNet.
This code was based on the Simple Baselines for Human Pose Estimation and Tracking. 

Trained model [COCO data,70 epoch] (https://drive.google.com/file/d/1H9dElsNDvA--ybbRANaaiLkajN6O0tHF/view?usp=sharing)

## Main Results 
### Results on COCO 2017 val (using 70 epoch trained model)
| Arch | Input size | AP | AP .5 | AP.75 | AP(M) | AP(L) | AR | AR .5 |
|---|---|---|---|---|---|---|---|---|---|
|HRNet_w32 | 256x192 | 39.8 | 73.5 | 38.5 | 37.6 | 45.1 | 54.6 | 84.4 |


Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
