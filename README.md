kaggle-AllState
====================

All State Competition

Model:

preprocessing - filtered the training data to better match the test set (not as a full history for each customer)

logistic regression model to assign a probability that a customer changed at least one option


gbm - interactions 2

features - most important: last picked option, state, probability mind changed, cost, age

all features are flattened out timewise into a cross section

model trained on each option as well as all pairwise combination of options (28 total models)

-this was done because there is a not insignificant codependence amonst the options

model combination - last picked option model was very strong, there are only select cases the other models performed better and ths they were only used in those cases
