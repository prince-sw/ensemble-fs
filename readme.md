## Introduction
In this project, we have performed performance analysis of fifteen feature selection methods by comparing 'accuracy' performance metric of each method over five classification algorithms.
We have used ten publicly available datasets for this purpose.

### Feature Selection Methods Used:
1. Pair-wise Correlation
3. Regularized Self Representation
4. Variance Threshold
5. Logistic Regression based selection
6. Random Forest (Gini importance)
7. Boruta Algorithm
8. LASSO Algorithm
9. Extra Tree Classifier
10. Mutual Information Classifier
11. Chi-Square Test
12. Recursive Feature Elimination with RF
13. Correlation
14. Cosine Similarity and Standard deviation with Exponent
15. Laplacian Score
16. Iterative Laplacian Score

### Classification Algorithms Used:
1. Decision Trees
2. Logistic Regression
3. Random Forest
4. KNN
5. Naive Bayes

### Datasets Used:
1. Iris
2. Breast Cancer
3. Pima Indians Diabetes
4. Cirrhosis Prediction
5. Parkinson's Disease
6. Heart Disease
7. Sonar
8. Stroke Prediction
9. Wine Quality
10. Abalone

## Results:
Two screenshots of the obtained results are given below.

![12](https://user-images.githubusercontent.com/108113078/211331043-95bd275a-a2bf-4658-966a-c25df14eaeb7.png)

![Screenshot 2023-01-09 194959](https://user-images.githubusercontent.com/108113078/211331070-8726dd49-6bd4-49ff-810f-fbfc9624dc95.png)

K is the number of best features taken. k=2 implies 2 best features given by each feature selection methods have been used to perform classification, based on which accuracy was calculated.

Accuracy = (TP + TN)/(TP + TN + FP + FN): 
where T is True, F is False, P is Positive and N is Negative.
