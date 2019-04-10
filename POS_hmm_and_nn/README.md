# Assignment 2

### POS tagging using Hidden Markov Model

The predicted results for the hidden markov model is stored in the files `hmm_predict_fold0.txt`, `hmm_predict_fold1.txt` and `hmm_predict_fold2.txt`.

#### Results for the Hidden Markov Model

**Fold 1:**

```
Accuracy: 0.5000443872342314
F1_Score: 0.5860324221685885
```

```                
                precision    recall  f1-score   support
   micro avg       0.50      0.50      0.50    180232
   macro avg       0.81      0.58      0.59    180232
weighted avg       0.94      0.50      0.61    180232
```


**Fold 2:**

```
Accuracy: 0.47995904279037505
F1_Score: 0.5622184418158768
```

```
                precision    recall  f1-score   support
    micro avg       0.48      0.48      0.48    181653
    macro avg       0.79      0.58      0.56    181653
weighted avg        0.95      0.48      0.59    181653
```


**Fold 3:**

```
Accuracy: 0.57695924289037505
F1_Score: 0.5900184210158268
```

```
                precision    recall  f1-score   support
    micro avg       0.48      0.48      0.48    181653
    macro avg       0.79      0.58      0.56    181653
weighted avg        0.95      0.48      0.59    181653
```

Macro average of the 3-folds are:

```
               precision    recall  f1-score   support
   macro avg       0.80      0.58      0.57    181179
```


### POS tagging using Neural Network

The predicted results for the neural network is stored in the file `logs_pos.txt`.

#### Results for the Neural Network

**Fold 1:**

```
Accuracy: 0.962
Recall: 0.275
Precision: 0.291
F_score1: 0.281
```

```
               precision    recall  f1-score   support
   micro avg       0.96      0.96      0.96    181069
   macro avg       0.29      0.28      0.28    181069
weighted avg       0.96      0.96      0.96    181069
```


**Fold 2:**

```
Accuracy: 0.963
Recall: 0.46
Precision: 0.505
F_score1: 0.47
```

```
               precision    recall  f1-score   support
   micro avg       0.96      0.96      0.96    181046
   macro avg       0.51      0.46      0.47    181046
weighted avg       0.96      0.96      0.96    181046
```


**Fold 3:**

```
Accuracy: 0.957
Recall: 0.533
Precision: 0.557
F_score1: 0.542
```

```
               precision    recall  f1-score   support
   micro avg       0.96      0.96      0.96    181034
   macro avg       0.56      0.53      0.54    181034
weighted avg       0.96      0.96      0.96    181034
```

Macro average of the 3-folds are:

```
               precision    recall  f1-score   support
   macro avg       0.45      0.42      0.43    181050
```
