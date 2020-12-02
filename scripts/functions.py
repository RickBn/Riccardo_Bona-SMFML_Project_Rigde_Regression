import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def processData(df, rmOutliers = False, rmCappedHouseValues = False, imputeModel = 'median', scaler = None, intercept = True):
    df_features = df.columns

    ocean_proximity = df['ocean_proximity'].copy()
    X = df.drop(['ocean_proximity'], axis=1)
    X = X.reset_index()
    X = X.drop(['index'], axis=1)

    features = X.columns

    validImputers = {'median', 'mean'}

    if imputeModel in validImputers:
        if imputeModel == 'median':
            imp = SimpleImputer(missing_values=np.nan, strategy='median')
            imp.fit(X)
            X = imp.transform(X)
        elif imputeModel == 'mean':
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(X)
            X = imp.transform(X)
    else:
        raise ValueError("results: status must be one of %r." % validImputers)

    X = pd.DataFrame(X, columns=features)
    X = pd.concat((X, ocean_proximity), axis=1)

    if(rmOutliers):
        for f in df_features[:-1]:
            q1 = np.percentile(X[f], 25, interpolation='midpoint')
            q3 = np.percentile(X[f], 75, interpolation='midpoint')
            iqr = q3 - q1
            X = X.loc[(X[f] > (q1 - (1.5 * iqr))) & (X[f] < (q3 + (1.5 * iqr)))]

    if(rmCappedHouseValues):
        X = X.loc[X['median_house_value'] < 500001]

    y = X['median_house_value']
    X = X.drop(['median_house_value'], axis=1)
    X = X.reset_index()
    X = X.drop(['index'], axis=1)
    y = y.reset_index()
    y = y.drop(['index'], axis=1)

    o_p = pd.get_dummies(data=X['ocean_proximity'], prefix='op')
    # o_p = pd.DataFrame(LabelEncoder().fit_transform(X['ocean_proximity']))
    # o_p.columns = ['ocean_proximity']
    X = X.drop(['ocean_proximity'], axis=1)
    X = X.reset_index()
    X = X.drop(['index'], axis=1)

    x_features = X.columns

    if scaler:
            X = scaler.fit_transform(X)
            X = pd.DataFrame(X, columns=x_features)

    X = pd.concat((X, o_p), axis=1)

    if intercept:
        X.insert(0, 'intercept', np.ones(len(X)))
        
    x_features = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X, y, x_features, X_train, X_test, y_train, y_test

def plot_ocean_proximity(df):
    plt.plot(df.loc[df['ocean_proximity'] == 'INLAND']['longitude'],
             df.loc[df['ocean_proximity'] == 'INLAND']['latitude'], marker='.', linestyle='None')
    plt.plot(df.loc[df['ocean_proximity'] == '<1H OCEAN']['longitude'],
             df.loc[df['ocean_proximity'] == '<1H OCEAN']['latitude'], marker='.', linestyle='None')
    plt.plot(df.loc[df['ocean_proximity'] == 'NEAR BAY']['longitude'],
             df.loc[df['ocean_proximity'] == 'NEAR BAY']['latitude'], marker='.', linestyle='None')
    plt.plot(df.loc[df['ocean_proximity'] == 'NEAR OCEAN']['longitude'],
             df.loc[df['ocean_proximity'] == 'NEAR OCEAN']['latitude'], marker='.', linestyle='None')
    plt.plot(df.loc[df['ocean_proximity'] == 'ISLAND']['longitude'],
             df.loc[df['ocean_proximity'] == 'ISLAND']['latitude'], marker='.', linestyle='None')
    plt.legend(labels=['INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND'])
    plt.title('ocean_proximity by coordinate')

def remove_by_vif(X, vif = 5):
    for i in range(len(X.columns)):
        l = [variance_inflation_factor(X.values, j) for j in range(X.shape[1])]
        s = pd.Series(index=X.columns, data=l).sort_values(ascending=False)
        if s.iloc[0] > vif:
            X.drop(s.index[0], axis=1, inplace=True)
            print('Removed: ', s.index[0], ', VIF: ', s.iloc[0])
        else:
            break
    return X

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def adjusted_r2(y_true, y_pred, n, k):
    RSS = np.sum(np.subtract(y_true, y_pred) ** 2)
    TSS = np.sum(np.subtract(y_true, y_pred.mean()) ** 2)
    adjusted_r2 = 1 - ((n - 1) / (n - k - 1) * (RSS/TSS))
    return adjusted_r2

def chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def ridge_w(X_train, y_train, alpha):
    I = np.identity(X_train.shape[1])
    xTx = np.dot(X_train.T, X_train)
    xTy = np.dot(X_train.T, y_train)
    return np.dot(np.linalg.pinv(np.add(xTx, alpha * I)), xTy)

def run_cv(X, y, num_splits, alphas, nested = False):
    xy = pd.concat((X, y), axis=1)
    xys = shuffle(xy)

    if not nested:
        cv_train_error, cv_test_error, cv_train_r2, cv_test_r2 = k_fold_cv(xys, num_splits, alphas)
    else:
        cv_train_error, cv_test_error, cv_train_r2, cv_test_r2 = nested_cv(xys, num_splits, alphas)

    return cv_train_error, cv_test_error, cv_train_r2, cv_test_r2

def k_fold_cv(data, num_splits, alphas):
    cv_train = np.zeros(len(alphas))
    cv_test = np.zeros(len(alphas))
    cv_r2_train = np.zeros(len(alphas))
    cv_r2_test = np.zeros(len(alphas))

    xyc = chunk(data, num_splits)

    for i, a in enumerate(alphas):
        cv_train_error = np.zeros(num_splits)
        cv_test_error = np.zeros(num_splits)
        cv_r2_train_score = np.zeros(num_splits)
        cv_r2_test_score = np.zeros(num_splits)

        for j, c in enumerate(xyc):
            y_test = c.iloc[:, -1]
            X_test = c.iloc[:, :-1]

            ts = xyc[:j] + xyc[j + 1:]
            ts = pd.concat(ts[:])

            y_train = ts.iloc[:, -1]
            X_train = ts.iloc[:, :-1]

            w = ridge_w(X_train, y_train, a)

            y_p_train = np.dot(X_train, w)
            cv_train_error[j] = mean_squared_error(y_train, y_p_train)
            cv_r2_train_score[j] = adjusted_r2(y_train, y_p_train, X_train.shape[0], X_train.shape[1])

            y_p_test = np.dot(X_test, w)
            cv_test_error[j] = mean_squared_error(y_test, y_p_test)
            cv_r2_test_score[j] = adjusted_r2(y_test, y_p_test, X_test.shape[0], X_test.shape[1])

        cv_train[i] = np.mean(cv_train_error)
        cv_test[i] = np.mean(cv_test_error)
        cv_r2_train[i] = np.mean(cv_r2_train_score)
        cv_r2_test[i] = np.mean(cv_r2_test_score)

    return [cv_train, cv_test, cv_r2_train, cv_r2_test]

def nested_cv(data, num_splits, alphas, verbose=False, num_internal=5):
    cv_test_error = np.zeros(num_splits)
    cv_train_error = np.zeros(num_splits)
    cv_r2_train_score = np.zeros(num_splits)
    cv_r2_test_score = np.zeros(num_splits)

    xyc = chunk(data, num_splits)

    for i, c in enumerate(xyc):
        y_test = c.iloc[:, -1]
        X_test = c.iloc[:, :-1]

        ts = xyc[:i] + xyc[i + 1:]
        ts = pd.concat(ts)
        ts = chunk(ts, num_internal)

        cv_val_error = {}

        for j, f in enumerate(ts):
            y_val = f.iloc[:, -1]
            X_val = f.iloc[:, :-1]

            inner_train_set = ts[:j] + ts[j + 1:]
            inner_train_set = pd.concat(inner_train_set[:])

            sub_y_train = inner_train_set.iloc[:, -1]
            sub_X_train = inner_train_set.iloc[:, :-1]

            for alpha in alphas:
                cv_val_error[alpha] = []

                w = ridge_w(sub_X_train, sub_y_train, alpha)
                y_p_val = np.dot(X_val, w)

                cv_val_error[alpha].append(mean_squared_error(y_val, y_p_val))

        mean_cv_val_error = {e: np.average(cv_val_error[e]) for e in cv_val_error}

        chosen_alpha = min(mean_cv_val_error, key=mean_cv_val_error.get)
        chosen_a_val_error = mean_cv_val_error[chosen_alpha]

        if verbose:
            print("Inner loop n°", i, ": chosen alpha = ", chosen_alpha)
            print("Inner loop n°", i, ": val error = ", mean_cv_val_error)
            print("Inner loop n°", i, ": val error = ", chosen_a_val_error)

        ts = pd.concat(ts[:])

        y_train = ts.iloc[:, -1]
        X_train = ts.iloc[:, :-1]

        w = ridge_w(X_train, y_train, chosen_alpha)

        y_p_train = np.dot(X_train, w)
        cv_train_error[i] = mean_squared_error(y_train, y_p_train)
        cv_r2_train_score[i] = adjusted_r2(y_train, y_p_train, X_train.shape[0], X_train.shape[1])

        y_p_test = np.dot(X_test, w)
        cv_test_error[i] = mean_squared_error(y_test, y_p_test)
        cv_r2_test_score[i] = adjusted_r2(y_test, y_p_test, X_test.shape[0], X_test.shape[1])

        if verbose:
            print("Fold", i, "Train MSE = ", cv_test_error[i], ", Train r2 score = ", cv_r2_test_score[i])
            print("Fold", i, "Test MSE = ", cv_test_error[i], ", Test r2 score = ", cv_r2_test_score[i])

    return [np.mean(cv_train_error), np.mean(cv_test_error), np.mean(cv_r2_train_score), np.mean(cv_r2_test_score)]

def n_shuffle_cv(X, y, num_splits, params, num_shuffles, nested=False):
    cv_train, cv_test = [[] for i in range(num_shuffles)], [[] for i in range(num_shuffles)]
    cv_r2_train, cv_r2_test = [[] for i in range(num_shuffles)], [[] for i in range(num_shuffles)]

    for i in range(num_shuffles):
        train, test, r2_train, r2_test = run_cv(X, y, num_splits, params, nested)
        cv_train[i] = train
        cv_test[i] = test
        cv_r2_train[i] = r2_train
        cv_r2_test[i] = r2_test

    if not nested:
        train_df = pd.DataFrame(cv_train)
        test_df = pd.DataFrame(cv_test)
        r2_train_df = pd.DataFrame(cv_r2_train)
        r2_test_df = pd.DataFrame(cv_r2_test)
        final_train = train_df.mean()
        final_test = test_df.mean()
        final_r2_train = r2_train_df.mean()
        final_r2_test = r2_test_df.mean()

    else:
        final_train = np.mean(cv_train)
        final_test = np.mean(cv_test)
        final_r2_train = np.mean(cv_r2_train)
        final_r2_test = np.mean(cv_r2_test)


    return [final_train, final_test, final_r2_train, final_r2_test]

def nested_pca(X, y, params, num_splits=5, plot=False, plot_title=""):
    pca_train_errors = []
    pca_test_errors = []

    for i in range(1, X.shape[1] + 1, 1):
        pca = PCA(n_components=i)
        X_pca = pd.DataFrame(pca.fit_transform(X))
        X_pca.insert(0, 'intercept', np.ones(len(X_pca)))

        cv_train, cv_test, cv_r2_train, cv_r2_test = run_cv(X_pca, y, num_splits, params, True)
        print('N° of components:', i, 'Train results: MSE:', cv_train, 'RMSE:', np.sqrt(cv_train),' | r2:', cv_r2_train, ' |''Test results: MSE:', cv_test, 'RMSE:', np.sqrt(cv_test), ' | r2:', cv_r2_test, ' |')
        pca_train_errors.append(cv_train)
        pca_test_errors.append(cv_test)

    if(plot):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(plot_title)
        ax.scatter(range(1, X.shape[1] + 1, 1), pca_test_errors)
        ax.plot(range(1, X.shape[1] + 1, 1), pca_test_errors)
        ax.plot(np.argmin(pca_test_errors) + 1, pca_test_errors[np.argmin(pca_test_errors)], marker='x', color='red', markersize=12)
        ax.set_xlabel("N° of components")
        ax.set_ylabel("MSE")
    print(pca_test_errors[np.argmin(pca_test_errors)])

    return [pca_train_errors, pca_test_errors]