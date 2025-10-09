[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UcP9Py08)

# Assumptions
Right now, we are working on some assumptions to simplify things:
- Codes 77 (Don't know/Not sure) and 99 (Refused) are treated as NaNs
  - Limitation: they could mean something. For example, respondent refuses to answer do to the severity of his illness
- NaNs in categorical features are also one-hot encoded (treated as one more category)
  - Exception: BPMEDS (taking medicine for high pressure), where NaNs are mainly because BPHIGH4 (have high pressure) is different from yes. We assumed that if respondent doesn't have high pressure, then does not take medicine. NaNs are encoded with 2 (not taking medicine).
  - Limitation: We are assuming that NaNs are informative signal. We need one more column
  - Alternative: Use all zeros to encode NaNs. Should modify `data_cleaning.one_hot_encode_column`

# [for 09/10] (if we can, if not, we work on it Thursday)
- everyone makes a model (pick whatever function) with all their features and % of dataset
- then remove some features and see how the accuracy changes
- data division
-   Lluc: columns 1-106
-   Siscu: 107-213
-   Ola: 213 - end

# [for 02/10]
- Ola: compile the array of info about features and select ~ 20 correlated features with not a lot of nulls for first pass function
- Lluc: description of each of the features
- Siscu - split training set into train and test
- Ola + Lluc + Siscu: each one applies a function to the dataset
  - Ola: mean_sgd
  - Lluc:mean_gd
  - Siscu: least_squares

#  [for 6/10]
- Ola: Implement the evaluation step
- Lluc: Implement the data cleaning for the threshold approach
- Siscu: Implement the data split & main implementation
- Ola + Lluc + Sicu: bring all functions that are covered on the first four labs and compare them (make sure that the provided tests are passed)
