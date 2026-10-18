**SPOTTING STUDENTS AT RISK EARLY**

A machine learning project that predicts, after four weeks, which online students will fail or withdraw from a course.

Notebook: https://colab.research.google.com/drive/1BLT7CGejndqQfdL_bhX1hviP_xnQ0MHL?usp=sharing
Report: https://docs.google.com/document/d/1CoMJBYDcmCHd5xVYmTanjBtd_C_XyRoUlpRMe5EdJZI/edit?usp=sharing
Google Drive (data and saved models): https://drive.google.com/drive/folders/1FA4lc0D8QMjtWFIrF8a5-wxsPXQ4j30C?usp=sharing

THE PROBLEM
Many students who study online do not finish. In the data we used, 44.1% of students who were still enrolled at week 4 later failed or withdrew. If a school can see who is in trouble early, it can offer help in time.

Question: Can we tell, using only the first four weeks of data, which students will fail or withdraw?

THE DATA
Dataset: Open University Learning Analytics Dataset (OULAD)
Source used: Kaggle, anlgrbz/student-demographics-online-education-dataoulad
Original source: https://analyse.kmi.open.ac.uk/open_dataset
License: CC BY 4.0
Size: 32,593 students, about 10.6 million rows of website clicks, 7 courses
Citation: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. Scientific Data, 4, 170171. https://doi.org/10.1038/sdata.2017.171

The raw data is NOT stored in this repo. The notebook downloads it from Kaggle (see “How to run”).

THE TARGET
A student is “at risk” if the final result is Withdrawn or Fail. A student is “not at risk” if the result is Pass or Distinction.

We predict on day 28 (end of week 4). We removed the 5,055 students who had already unregistered by day 28. That left 27,538 students.

WHAT WE BUILT
Features (28 columns, all from day 28 or earlier): website clicks (total, per week, by activity type), active days, early assignment scores and counts, and student background.
Traditional models (scikit-learn): logistic regression, random forest, and gradient boosting. The last two were tuned with randomized search and cross-validation grouped by student.
Deep learning model (Keras Functional API): a small neural network, tuned over 12 settings.
Experiments: how early can we predict (1, 2, 4, 8, 12 weeks), and how the alert cutoff changes the results.
Analysis: feature importance and error analysis, including results by group.

We never used the unregistration date or the final result as an input, because they would give away the answer.

RESULTS (test set, 5,576 students, cutoff 0.50)
Model Accuracy Precision Recall F1 ROC-AUC
Logistic regression (baseline) 0.724 0.743 0.600 0.664 0.789
Random forest (tuned) 0.726 0.752 0.594 0.663 0.790
Gradient boosting (tuned) 0.727 0.744 0.610 0.670 0.797
Neural network (Keras Functional API) 0.719 0.738 0.593 0.657 0.793

Key findings:

- All four models score about the same. The information in the first weeks limits the result, not the model.
- Waiting longer helps: ROC-AUC rises from 0.734 (1 week) to 0.861 (12 weeks).
- Lowering the alert cutoff from 0.50 to 0.30 raises recall from 0.610 to 0.844, and cuts missed at-risk students from 991 to 397. The cost is more false alarms (531 to 1,442).
- The strongest signals are the average score on early assignments, the course, assignments handed in, and active days.
- The at-risk students the model misses look strong at week 4 (more active days and higher scores than the average student who is not at risk).

HOW TO RUN
- Open the notebook in Google Colab (or Jupyter).
- Install the packages:
   pip install -r requirements.txt
- Run the cells from top to bottom. The first cells download the data from Kaggle with kagglehub. This needs an internet connection.
- Random seed is 42 everywhere, so results should match the tables above.
  
FILES
Notebook: https://colab.research.google.com/drive/1BLT7CGejndqQfdL_bhX1hviP_xnQ0MHL?usp=sharing
The full notebook: data, features, models, experiments

requirements.txt
Python packages and versions

results/all_model_results.csv
Test scores for the four models

results/cutoff_experiment.csv
Scores at 1, 2, 4, 8 and 12 weeks

results/nn_tuning_results.csv
Neural network tuning results

results/feature_importance.csv
Permutation importance

results/error_by_group.csv
Recall and flag rate by group

Large files (the cleaned dataset and the saved models) are in the Google Drive folder:

- oulad_features.csv (the cleaned feature table)
- gradient_boosting_pipeline.joblib and .pkl (best traditional model, with its preprocessing)
- early_warning_nn.h5, early_warning_nn.keras and early_warning_nn_savedmodel/ (neural network)
- nn_preprocessor.joblib (preprocessing needed before the neural network)

LIMITS
- One school and one time period (2013 and 2014).
- One train/test split. Small gaps between models are not proof that one model is better.
- Failing and withdrawing are treated as one outcome.
- At the 0.30 cutoff, about 4 in 10 flagged students were not truly at risk. The tool suits low-cost support, such as a check-in message.
- The tool uses background facts and flags some groups more often. A school should check it for fairness before using it.
