# NEU-CLS Dataset

Download the NEU Metal Surface Defect Dataset from Kaggle:
https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data

After downloading, extract to this directory so the structure is:

```
data/NEU-CLS/
    Crazing/
    Inclusion/
    Patches/
    Pitted_surface/
    Rolled-in_scale/
    Scratches/
```

Each class folder should contain ~300 grayscale images (.bmp or .jpg, 200×200 pixels).

Then run:
    python train.py --data_dir data/NEU-CLS --epochs 30
