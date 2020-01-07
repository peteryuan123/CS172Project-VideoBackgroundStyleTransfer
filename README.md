**Background style transfer**

Use

```
pip install -r requirements.txt
```

to install the required environment



Use

```
./make.sh
```

to set up environment



Use 

```
python3 seg.py --resume model/SiamMask_DAVIS.pth --video YourVideo
```

for frame extraction and segmentation



Use

```
cd fast-neural-style && cd neural_style
python3 neural_style.py
```

for image style transferring



Use 

```
cd ../..
python3 combine.py
```

to combine original object with transfered background