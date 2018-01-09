# NeuralFDR
Software accompanying "NeuralFDR: Learning Discovery Thresholds from Hypothesis Features"


## Dependencies 
You will have to install PyTorch to run the code, follow the instructions from http://pytorch.org


## Download the data
Download the data used in the paper from this [dropbox folder](https://www.dropbox.com/sh/wtp58wd60980d6b/AAA4wA60ykP-fDfS5BNsNkiGa?dl=0).

## Train a NeuralFDR model

```
python train.py --data data/data_airway.csv --dim 1 --out airway
```

The report will be available in airway folder

