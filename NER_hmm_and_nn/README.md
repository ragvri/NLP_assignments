# Assignment 3

### NER using Hidden Markov Model

### NER using Neural Network

Two models were created for the neural network `nn_coarse.h5` and `nn_fine.h5`. Following are the results for the same:

#### Coarse model results

**Fold 1:**

```
Accuracy: 0.95
Recall: 0.627
Precision: 0.671
F_score1: 0.647
```

```
               precision    recall  f1-score   support
   micro avg       0.95      0.95      0.95      3878
   macro avg       0.67      0.63      0.65      3878
weighted avg       0.95      0.95      0.95      3878
```

**Fold 2:**

```
Accuracy: 0.957
Recall: 0.661
Precision: 0.735
F_score1: 0.688
```

```
               precision    recall  f1-score   support
   micro avg       0.96      0.96      0.96      3877
   macro avg       0.73      0.66      0.69      3877
weighted avg       0.95      0.96      0.95      3877
```

**Fold 3:**

```
Accuracy: 0.956
Recall: 0.703
Precision: 0.71
F_score1: 0.706
```

```
               precision    recall  f1-score   support
   micro avg       0.96      0.96      0.96      3876
   macro avg       0.71      0.70      0.71      3876
weighted avg       0.96      0.96      0.96      3876
```

Macro average of the 3-folds are:

```
               precision    recall  f1-score   support
   macro avg       0.70      0.66      0.68      3877
```

#### Fine model results

**Fold 1:**

```
Accuracy: 0.95
Recall: 0.243
Precision: 0.367
F_score1: 0.282
```

```
               precision    recall  f1-score   support
   micro avg       0.95      0.95      0.95      6467
   macro avg       0.37      0.24      0.28      6467
weighted avg       0.94      0.95      0.94      6467
```

**Fold 2:**

```
Accuracy: 0.956
Recall: 0.229
Precision: 0.368
F_score1: 0.269
```

```
               precision    recall  f1-score   support
   micro avg       0.96      0.96      0.96      6459
   macro avg       0.37      0.23      0.27      6459
weighted avg       0.94      0.96      0.95      6459
```

**Fold 3:**

```
Accuracy: 0.951
Recall: 0.22
Precision: 0.326
F_score1: 0.249
```

```
               precision    recall  f1-score   support
   micro avg       0.95      0.95      0.95      6456
   macro avg       0.33      0.22      0.25      6456
weighted avg       0.94      0.95      0.94      6456
```

Macro average of the 3-folds are:

```
               precision    recall  f1-score   support
   macro avg       0.36      0.23      0.27      6461
```
