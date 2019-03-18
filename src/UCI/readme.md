http://archive.ics.uci.edu/ml/datasets.html
- adult - convert categorical to integer, 2 classification, big dataset
    - convert categorical to cat codes, drop missing values (final instances== 45222), binary classification (0 if <=50K else 1). Predict 1 (33% data instances)

- waveform - preprocess to only use 2 out of 3 classes.
    - used real data without noise and dropped class label 2, no missing values. Total instances=3304, (50-50% data instances, 0 and 1 label), binary classification

- Statlog (Shuttle) - , preprocess to only use 2 classes.
    - dropped all classes except 1, 4, no missing values. Now 4 is class label 1, and 1 is 0. Total instances=54,489 (20% relevant class), binary classification

- Statlog (Landsat Satellite) - use only 2 classes
    - dropped all classes except 1, 4, no missing values. Now 4 is class label 1, and 1 is 0. Total instances=2150 (30% relevant class), binary classification

- penbased http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
    - dropped all classes except 2, 4, no missing values. Now 4 is class label 1, and 1 is 0. Total instances=2300 (50% relevant class), binary classification

- optdigits - http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    - dropped all classes except 1, 3, no missing values. Now 3 is class label 1, and 1 is 0. Total instances=1143 (50% relevant class), binary classification

- covertype - http://archive.ics.uci.edu/ml/datasets/Covertype
    - dropped all classes except 5, 4, no missing values. Now 4 is class label 1, and 5 is 0. Total instances=12,240 (34% relevant class), binary classification

- breast cancer - http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    - no missing values. Converted labels M, B to 1,0. Total instances=569 (37% relevant class), binary classification

- diabetic - http://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
    - no missing values. 1 is diabetic, 0 non-diabetic. Total instances=1151 (53% relevant), binary classification


