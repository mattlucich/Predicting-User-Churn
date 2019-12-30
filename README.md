# Predicting-User-Churn

Using classification models to identify music streaming app users at risk of churning.

## Libraries Used

pyspark<br/>
numpy<br/>
pandas<br/>
matplotlib<br/>
seaborn<br/>
scipy<br/>
datetime<br/>
re<br/>

```bash
pip install requirements.txt
```

## Motivation 

Lowering customer churn is a goal all companies strive for. One of the most effective ways to do so is by being able to identify customers who are at risk of churning and taking action to keep them onboard. Thanks to the advent of machine learning, this task has become a bit easier over the years.

## Summary 

* Four classification models were tested: logistic regression, random forest, gradient-boosted tree, and Naive Bayes. 

* For each model, different sets of hyper parameters were compared in order to derive the best model. 

* Cross-validation was employed.

* Models were evaluated on f1 score via validation dataset.

* The best model is Naive Bayes with an F1 score of 71.1%.


## Files
Sparkify.ipynb - The jupyter notebook containing the analysis. Including preprocessing, data exploration, modeling, and evaluation.

mini_sparkify_event_data.json (not included) - The user activity/analytics dataset which comes from Udacity.
