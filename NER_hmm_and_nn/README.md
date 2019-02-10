# Assignment 3

### NER using Hidden Markov Model

The predicted results for the coarse model are stored in three files `hmm_coarse0.txt`, `hmm_coarse1.txt` and `hmm_coarse2.txt`.

Similarly for the fine model, the predicted results are stored in files `hmm_fine0.txt`, `hmm_fine1.txt` and `hmm_fine2.txt`.

The results after running the perl script `connlleval.pl` are as follows:

#### Coarse model results

```
processed 6461 tokens with 66 phrases; found: 226 phrases; correct: 31.
accuracy:  94.72%; precision:  13.72%; recall:  46.97%; FB1:  21.23
                 : precision:  13.72%; recall:  46.97%; FB1:  21.23  226
                 : precision:  13.72%; recall:  46.97%; FB1:  21.23  226
```

```
processed 6230 tokens with 75 phrases; found: 227 phrases; correct: 44.
accuracy:  94.96%; precision:  19.38%; recall:  58.67%; FB1:  29.14
                 : precision:  19.38%; recall:  58.67%; FB1:  29.14  227
                 : precision:  19.38%; recall:  58.67%; FB1:  29.14  227
```

```
processed 6691 tokens with 65 phrases; found: 198 phrases; correct: 33.
accuracy:  95.43%; precision:  16.67%; recall:  50.77%; FB1:  25.10
                 : precision:  16.67%; recall:  50.77%; FB1:  25.10  198
                 : precision:  16.67%; recall:  50.77%; FB1:  25.10  198
```

#### Fine model results

```
processed 6300 tokens with 62 phrases; found: 210 phrases; correct: 31.
accuracy:  94.94%; precision:  14.76%; recall:  50.00%; FB1:  22.79
          company: precision:  43.48%; recall:  71.43%; FB1:  54.05  23
         facility: precision:   7.14%; recall:  33.33%; FB1:  11.76  14
          geo-loc: precision:  17.78%; recall:  66.67%; FB1:  28.07  45
            movie: precision:  20.00%; recall:  50.00%; FB1:  28.57  5
      musicartist: precision:   0.00%; recall:   0.00%; FB1:   0.00  7
            other: precision:  10.00%; recall:  40.00%; FB1:  16.00  40
           person: precision:  12.73%; recall:  50.00%; FB1:  20.29  55
          product: precision:   0.00%; recall:   0.00%; FB1:   0.00  11
       sportsteam: precision:   0.00%; recall:   0.00%; FB1:   0.00  4
           tvshow: precision:   0.00%; recall:   0.00%; FB1:   0.00  6
```

```
processed 6539 tokens with 62 phrases; found: 224 phrases; correct: 33.
accuracy:  94.51%; precision:  14.73%; recall:  53.23%; FB1:  23.08
          company: precision:  30.77%; recall:  61.54%; FB1:  41.03  26
         facility: precision:  17.65%; recall:  42.86%; FB1:  25.00  17
          geo-loc: precision:   8.11%; recall:  37.50%; FB1:  13.33  37
            movie: precision:  20.00%; recall:  50.00%; FB1:  28.57  5
      musicartist: precision:   0.00%; recall:   0.00%; FB1:   0.00  11
            other: precision:   3.12%; recall:  16.67%; FB1:   5.26  32
           person: precision:  23.53%; recall:  84.21%; FB1:  36.78  68
          product: precision:   5.56%; recall:  25.00%; FB1:   9.09  18
       sportsteam: precision:   0.00%; recall:   0.00%; FB1:   0.00  5
           tvshow: precision:   0.00%; recall:   0.00%; FB1:   0.00  5
```

```
processed 6543 tokens with 72 phrases; found: 217 phrases; correct: 26.
accuracy:  94.53%; precision:  11.98%; recall:  36.11%; FB1:  17.99
          company: precision:  17.24%; recall:  71.43%; FB1:  27.78  29
         facility: precision:   5.88%; recall:  14.29%; FB1:   8.33  17
          geo-loc: precision:  10.64%; recall:  35.71%; FB1:  16.39  47
            movie: precision:   0.00%; recall:   0.00%; FB1:   0.00  3
      musicartist: precision:   0.00%; recall:   0.00%; FB1:   0.00  5
            other: precision:  11.11%; recall:  20.00%; FB1:  14.29  36
           person: precision:  15.87%; recall:  55.56%; FB1:  24.69  63
          product: precision:  12.50%; recall:  25.00%; FB1:  16.67  8
       sportsteam: precision:   0.00%; recall:   0.00%; FB1:   0.00  8
           tvshow: precision:   0.00%; recall:   0.00%; FB1:   0.00  1
```

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
