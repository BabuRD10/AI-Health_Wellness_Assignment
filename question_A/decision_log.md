# Decision Log - Question A

## 1. Dataset Selection

I selected the Pima Indians Diabetes Dataset for the health-risk
prediction task.

The dataset contains 768 rows and 8 input features.

## 2. Data Inspection

The dataset was inspected before training.

There were no missing values and no duplicate rows.

Zero values were observed in some medical-related columns. These
were not automatically treated as missing values because the dataset
does not establish that every zero value represents missing data.

## 3. Personal Seed

Personal seed S = 3012 was used for the train/test split and model
random states where applicable.

## 4. Train/Test Split

An 80/20 train/test split was used.

Stratification was used so that the class distribution was maintained
between the training and testing sets.

## 5. Level 1 Model Selection

Two classification models were selected:

1. Logistic Regression
2. Random Forest

Random Forest performed better than Logistic Regression on the test
data.

Logistic Regression:
Accuracy: 74.68%
Precision: 64.71%
Recall: 61.11%

Random Forest:
Accuracy: 91.56%
Precision: 88.68%
Recall: 87.04%

## 6. Level 2 From-Scratch Model

Logistic Regression was implemented from scratch using NumPy.

The implementation included:

- Sigmoid function
- Logistic loss function
- Gradient descent
- Weight and bias updates
- Probability prediction
- Classification using a 0.5 threshold
- Confusion matrix calculated without a library

The same train/test split used for the sklearn Logistic Regression
was used so that the accuracy comparison would be fair.

From-scratch Logistic Regression Accuracy: 75.32%

Sklearn Logistic Regression Accuracy: 74.68%

Difference: 0.65 percentage points

Top 3 feature weights:

1. Glucose: 0.8589
2. Pregnancies: 0.6017
3. SkinThickness: 0.4496

## 7. Level 3 Threshold Decision

The classification threshold was lowered to increase recall.

The highest threshold that achieved at least 90% recall on the training
data was selected.

Selected threshold: 0.21

Training recall: 90.19%

The selected threshold was then tested on the unseen test data.

Test accuracy: 68.18%
Test precision: 52.75%
Test recall: 88.89%

## 8. Threshold Observation

Lowering the classification threshold increased recall but reduced
precision and accuracy.

This shows why accuracy alone is not sufficient when evaluating a
screening-oriented classification model.

For this assignment experiment, 0.21 was selected as the candidate
screening threshold because it achieved the required high recall on
the training data.

This threshold is only an experimental model threshold and must not
be considered a clinical or medical recommendation. Real-world use
would require appropriate clinical validation.

## 9. Limitations

The experiment uses a public dataset and is intended for academic
demonstration.

The model has not been clinically validated and should not be used
to make medical decisions.

Model performance can change with different datasets, populations,
features, and train/test splits.

## 10. AI Usage

AI assistance was used for learning concepts, understanding the
assignment requirements, structuring the implementation, and
debugging code.

The code was run and tested locally, and the reported results are
from the actual local execution.