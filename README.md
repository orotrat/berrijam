#### Image Classifier Experiment for Berrijam Competition

Attempt at classification with only 5 positive/negative training examples.

Currently 2 casses present.

Cats vs Dogs and ripe bananas vs unripe bananas.

### Cats vs Dogs

Confusion Matrix:

|              | Predicted Cats | Predicted Dogs |
|--------------|----------------|----------------|
| Actual Cats  | 924            | 76             |
| Actual Dogs  | 24             | 976            |

Classification Report:

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| Cats      | 0.97      | 0.92   | 0.95     | 1000    |
| Dogs      | 0.93      | 0.98   | 0.95     | 1000    |

Accuracy: 95.00%

### Ripe Bananas vs Unripe Bananas

Confusion Matrix:

|                | Predicted Ripe | Predicted Unripe |
|----------------|----------------|------------------|
| Actual Ripe    | 216            | 22               |
| Actual Unripe  | 74             | 164              |

Classification Report:

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| Ripe      | 0.74      | 0.91   | 0.82     | 238     |
| Unripe    | 0.88      | 0.69   | 0.77     | 238     |

Accuracy: 79.83%

Requires python 3.11.0 for tensorflow