import copy
import math

from sklearn.metrics import confusion_matrix
import numpy as np

def get_counts(clf, x_train, y_train, x_test, y_test, test_df, biased_col, metric='aod'):
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    cnf_matrix_valid = confusion_matrix(y_test, y_pred)

    #print(cnf_matrix_valid)

    TP = cnf_matrix_valid[0][0]
    FP = cnf_matrix_valid[0][1]
    FN = cnf_matrix_valid[1][0]
    TN = cnf_matrix_valid[1][1]


    test_df['current_pred_' + biased_col] = y_pred.tolist()

    test_df['TP_' + biased_col + "_1"] = np.where((test_df['Probability'] == 1) &
                                           (test_df['current_pred_' + biased_col] == 1) &
                                           (test_df[biased_col] == 1), 1, 0)

    test_df['TN_' + biased_col + "_1"] = np.where((test_df['Probability'] == 0) &
                                                  (test_df['current_pred_' + biased_col] == 0) &
                                                  (test_df[biased_col] == 1), 1, 0)

    test_df['FN_' + biased_col + "_1"] = np.where((test_df['Probability'] == 1) &
                                                  (test_df['current_pred_' + biased_col] == 0) &
                                                  (test_df[biased_col] == 1), 1, 0)

    test_df['FP_' + biased_col + "_1"] = np.where((test_df['Probability'] == 0) &
                                                  (test_df['current_pred_' + biased_col] == 1) &
                                                  (test_df[biased_col] == 1), 1, 0)

    test_df['TP_' + biased_col + "_0"] = np.where((test_df['Probability'] == 1) &
                                                  (test_df['current_pred_' + biased_col] == 1) &
                                                  (test_df[biased_col] == 0), 1, 0)

    test_df['TN_' + biased_col + "_0"] = np.where((test_df['Probability'] == 0) &
                                                  (test_df['current_pred_' + biased_col] == 0) &
                                                  (test_df[biased_col] == 0), 1, 0)

    test_df['FN_' + biased_col + "_0"] = np.where((test_df['Probability'] == 1) &
                                                  (test_df['current_pred_' + biased_col] == 0) &
                                                  (test_df[biased_col] == 0), 1, 0)

    test_df['FP_' + biased_col + "_0"] = np.where((test_df['Probability'] == 0) &
                                                  (test_df['current_pred_' + biased_col] == 1) &
                                                  (test_df[biased_col] == 0), 1, 0)

    a = test_df['TP_' + biased_col + "_1"].sum()
    b = test_df['TN_' + biased_col + "_1"].sum()
    c = test_df['FN_' + biased_col + "_1"].sum()
    d = test_df['FP_' + biased_col + "_1"].sum()
    e = test_df['TP_' + biased_col + "_0"].sum()
    f = test_df['TN_' + biased_col + "_0"].sum()
    g = test_df['FN_' + biased_col + "_0"].sum()
    h = test_df['FP_' + biased_col + "_0"].sum()

    if metric=='aod':
        return  calculate_average_odds_difference(a, b, c, d, e, f, g, h)
    elif metric=='eod':
        return calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h)
    elif metric=='d2h':
        return d2h(TP,FP,FN,TN)


def calculate_average_odds_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
	TPR_male = TP_male/(TP_male+FN_male)
	TPR_female = TP_female/(TP_female+FN_female)
	FPR_male = FP_male/(FP_male+TN_male)
	FPR_female = FP_female/(FP_female+TN_female)
	average_odds_difference = abs(abs(TPR_male - TPR_female) - abs(FPR_male - FPR_female))/2
	print("average_odds_difference",average_odds_difference)
	return average_odds_difference


def calculate_equal_opportunity_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
	TPR_male = TP_male/(TP_male+FN_male)
	TPR_female = TP_female/(TP_female+FN_female)
	# print(TPR_male,TPR_female)
	equal_opportunity_difference = abs(TPR_male - TPR_female)
	print("equal_opportunity_difference:",equal_opportunity_difference)
	return equal_opportunity_difference

## Calculate d2h
def d2h(TP,FP,FN,TN):
   if (FP + TN) != 0:
       far = FP/(FP+TN)
   if (TP + FN) != 0:
       recall = TP/(TP + FN)
   dist2heaven = math.sqrt((1 - (recall)**2)/2)
   print("dist2heaven:",dist2heaven)
   return dist2heaven


def measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, biased_col, metric):
    df = copy.deepcopy(test_df)
    get_counts(clf, X_train, y_train, X_test, y_test, df, biased_col, metric=metric)