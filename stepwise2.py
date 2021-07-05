import pandas as pd
import statsmodels.api as sm


with open('All_data.csv', 'r') as f:
    allData = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\All_data.csv', skiprows=0, delimiter=',')

#2000 Data
x = allData[['Population_00', 'White_00', 'Black_00', 'Asian_00', 'Hispanic_00', 'Republican_00',
'Democratic_00', 'Independence_00', 'Conservative_00', 'Liberal_00', 'Green_00', 'Working_Families_00', 'AvgHousePrice_00']]


y = allData['Response_Rate_00'] #2000

def forward_regression(x, y,
                       threshold_in,
                       verbose=False):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    return included

def backward_regression(x, y,
                           threshold_out,
                           verbose=False):
    included=list(x.columns)
    while True:
        changed=False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
