# Breast-Cancer-Prediction
Prediction if a tumor will be benign or malignant given different attributes of the tumor.

The goal of this 1st version was to use voting classifiers in addition to classic classification models such as RandomForestTree and Support Vector Machines. Accuracy was used as the metrics.
Accuracy achieved was very high for all of the 3 options, between 95% and 98%, but since it is a critical medical decision, we need to absloutely avoid false negatives.
Goal on following versions will be to use other metrics such as precision and recall. Since it is a medical issue, we would prefer to have lower false negatives for the patient mental health. Cross validation will also be applied to verify the solidity of the model and other classifications models will be used to make the voting classifier more robust.
Depending on the model selected, some parameter tuning could be applied.
Also we could check in the preprocessing steps if removing outliers past 3std or 2std changes the model. Again, since it is a medical issue we need to be careful about changing data set since one datapoint corresponds to the life diagnosis of one person.

