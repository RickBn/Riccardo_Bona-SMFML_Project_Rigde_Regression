from scripts.functions import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

df = pd.read_csv('data/cal-housing.csv')

num_splits = 5
num_shuffles = 10

params = [0, 0.0001, 0.001, 0.01, 0.05, 0.1,
          0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0,
          4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 200, 300, 400, 500]

# For all datasets: intercept, dummies for ocean_proximity, median imputer, standard scaler

# Case (a): no data removed
X, y, x_features, X_train, X_test, y_train, y_test = processData(df, scaler=StandardScaler())

cv_train, cv_test, cv_r2_train, cv_r2_test = n_shuffle_cv(X, y, num_splits, params, num_shuffles, nested=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(params, cv_train)
ax.plot(params, cv_test)
ax.set_xlabel('Alpha')
ax.set_ylabel('MSE')
ax.legend(labels=['training error', 'test error'])

print('NO VALUES REMOVED:')
print('AVERAGE:')
print('TRAIN:', ' | MSE:', np.mean(cv_train), ' | RMSE:', np.sqrt(np.mean(cv_train)),
      ' | R2:', np.mean(cv_r2_train), ' | ')
print('TEST:', ' | MSE:', np.mean(cv_test), '| RMSE:', np.sqrt(np.mean(cv_test)),
      ' | R2:', np.mean(cv_r2_test), ' | ')

print('BEST:')
print('TRAIN:', '| Best alpha:', params[np.argmin(cv_train)], ' | MSE:', cv_train[np.argmin(cv_train)],
      ' | RMSE:', np.sqrt(cv_train[np.argmin(cv_train)]), ' | R2:', cv_r2_train[np.argmin(cv_train)], ' | ')
print('TEST:', ' | Best alpha:', params[np.argmin(cv_test)], ' | MSE:', cv_test[np.argmin(cv_test)],
      ' | RMSE:', np.sqrt(cv_test[np.argmin(cv_test)]), ' | R2:', cv_r2_test[np.argmin(cv_test)], ' | ')

n_cv_train, n_cv_test, n_cv_r2_train, n_cv_r2_test = n_shuffle_cv(X, y, num_splits, params, num_shuffles, nested=True)
print("| MSE= ", "Train:", n_cv_train, " - Test:", n_cv_test, " | R2 Score= ", "Train:", n_cv_r2_train, " - Test:", n_cv_r2_test, " |")

# Case (b): Only capped median_house_values removed:
X, y, x_features, X_train, X_test, y_train, y_test = processData(df, rmCappedHouseValues=True, scaler=StandardScaler())

cv_train, cv_test, cv_r2_train, cv_r2_test = n_shuffle_cv(X, y, num_splits, params, num_shuffles, nested=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(params, cv_train)
ax.plot(params, cv_test)
ax.set_xlabel('Alpha')
ax.set_ylabel('MSE')
ax.legend(labels=['training error', 'test error'])

print('medianHouseValue CAPPED VALUES REMOVED:')
print('AVERAGE:')
print('TRAIN:', ' | MSE:', np.mean(cv_train), ' | RMSE:', np.sqrt(np.mean(cv_train)),
      ' | R2:', np.mean(cv_r2_train), ' | ')
print('TEST:', ' | MSE:', np.mean(cv_test), '| RMSE:', np.sqrt(np.mean(cv_test)),
      ' | R2:', np.mean(cv_r2_test), ' | ')

print('BEST:')
print('TRAIN:', '| Best alpha:', params[np.argmin(cv_train)], ' | MSE:', cv_train[np.argmin(cv_train)],
      ' | RMSE:', np.sqrt(cv_train[np.argmin(cv_train)]), ' | R2:', cv_r2_train[np.argmin(cv_train)], ' | ')
print('TEST:', ' | Best alpha:', params[np.argmin(cv_test)], ' | MSE:', cv_test[np.argmin(cv_test)],
      ' | RMSE:', np.sqrt(cv_test[np.argmin(cv_test)]), ' | R2:', cv_r2_test[np.argmin(cv_test)], ' | ')

n_cv_train, n_cv_test, n_cv_r2_train, n_cv_r2_test = n_shuffle_cv(X, y, num_splits, params, num_shuffles, nested=True)
print("| MSE= ", "Train:", n_cv_train, " - Test:", n_cv_test, " | R2 Score= ", "Train:", n_cv_r2_train, " - Test:", n_cv_r2_test, " |")

# Case (c): All outliers removed:
X, y, x_features, X_train, X_test, y_train, y_test = processData(df, rmOutliers=True, scaler=StandardScaler())

cv_train, cv_test, cv_r2_train, cv_r2_test = n_shuffle_cv(X, y, num_splits, params, num_shuffles, nested=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(params, cv_train)
ax.plot(params, cv_test)
ax.set_xlabel('Alpha')
ax.set_ylabel('MSE')
ax.legend(labels=['training error', 'test error'])

print('ALL OUTLIERS REMOVED:')
print('AVERAGE:')
print('TRAIN:', ' | MSE:', np.mean(cv_train), ' | RMSE:', np.sqrt(np.mean(cv_train)),
      ' | R2:', np.mean(cv_r2_train), ' | ')
print('TEST:', ' | MSE:', np.mean(cv_test), '| RMSE:', np.sqrt(np.mean(cv_test)),
      ' | R2:', np.mean(cv_r2_test), ' | ')

print('BEST:')
print('TRAIN:', '| Best alpha:', params[np.argmin(cv_train)], ' | MSE:', cv_train[np.argmin(cv_train)],
      ' | RMSE:', np.sqrt(cv_train[np.argmin(cv_train)]), ' | R2:', cv_r2_train[np.argmin(cv_train)], ' | ')
print('TEST:', ' | Best alpha:', params[np.argmin(cv_test)], ' | MSE:', cv_test[np.argmin(cv_test)],
      ' | RMSE:', np.sqrt(cv_test[np.argmin(cv_test)]), ' | R2:', cv_r2_test[np.argmin(cv_test)], ' | ')

n_cv_train, n_cv_test, n_cv_r2_train, n_cv_r2_test = n_shuffle_cv(X, y, num_splits, params, num_shuffles, nested=True)
print("| MSE= ", "Train:", n_cv_train, " - ", np.sqrt(n_cv_train), " - Test:", n_cv_test, " - ",  np.sqrt(n_cv_test),
      " | R2 Score= ", "Train:", n_cv_r2_train, " - Test:", n_cv_r2_test, " |")

#No collinear features
X, y, x_features, X_train, X_test, y_train, y_test = processData(df, scaler=StandardScaler(), intercept=False)
vif = pd.DataFrame([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], X.columns)
X = remove_by_vif(X)
X.insert(0, 'intercept', np.ones(len(X)))

cv_train, cv_test, cv_r2_train, cv_r2_test = n_shuffle_cv(X, y, num_splits, params, num_shuffles, nested=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(params, cv_train)
ax.plot(params, cv_test)
ax.set_xlabel('Alpha')
ax.set_ylabel('MSE')
ax.legend(labels=['training error', 'test error'])

print('NO COLLINEAR FEATURES:')
print('AVERAGE:')
print('TRAIN:', ' | MSE:', np.mean(cv_train), ' | RMSE:', np.sqrt(np.mean(cv_train)),
      ' | R2:', np.mean(cv_r2_train), ' | ')
print('TEST:', ' | MSE:', np.mean(cv_test), '| RMSE:', np.sqrt(np.mean(cv_test)),
      ' | R2:', np.mean(cv_r2_test), ' | ')

print('BEST:')
print('TRAIN:', '| Best alpha:', params[np.argmin(cv_train)], ' | MSE:', cv_train[np.argmin(cv_train)],
      ' | RMSE:', np.sqrt(cv_train[np.argmin(cv_train)]), ' | R2:', cv_r2_train[np.argmin(cv_train)], ' | ')
print('TEST:', ' | Best alpha:', params[np.argmin(cv_test)], ' | MSE:', cv_test[np.argmin(cv_test)],
      ' | RMSE:', np.sqrt(cv_test[np.argmin(cv_test)]), ' | R2:', cv_r2_test[np.argmin(cv_test)], ' | ')

n_cv_train, n_cv_test, n_cv_r2_train, n_cv_r2_test = n_shuffle_cv(X, y, num_splits, params, num_shuffles, nested=True)
print("| MSE= ", "Train:", n_cv_train, " - ", np.sqrt(n_cv_train), " - Test:", n_cv_test, " - ",  np.sqrt(n_cv_test),
      " | R2 Score= ", "Train:", n_cv_r2_train, " - Test:", n_cv_r2_test, " |")

#Principal component analysis, dataset standardized and no intercept added before performing PCA

# Case (a):
X, y, x_features, X_train, X_test, y_train, y_test = processData(df, scaler=StandardScaler(), intercept=False)

pca = PCA(whiten=True)
pca.fit(X)
variance = pd.DataFrame(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))

pca_train_a, pca_test_a = nested_pca(X, y, params, num_splits=5)

# Case (b):
X, y, x_features, X_train, X_test, y_train, y_test = processData(df, rmCappedHouseValues=True, scaler=StandardScaler(), intercept=False)

pca = PCA(whiten=True)
pca.fit(X)
variance = pd.DataFrame(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))

pca_train_b, pca_test_b = nested_pca(X, y, params, num_splits=5)

# Case (c):
X, y, x_features, X_train, X_test, y_train, y_test = processData(df, rmOutliers=True, scaler=StandardScaler(), intercept=False)

pca = PCA(whiten=True)
pca.fit(X)
variance = pd.DataFrame(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))

pca_train_c, pca_test_c = nested_pca(X, y, params, num_splits=5)

# Plot pca test performance
mum_components = X.shape[1] + 1
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(range(1, mum_components, 1), pca_test_a)
ax.plot(range(1, mum_components, 1), pca_test_a)
ax.scatter(range(1, mum_components, 1), pca_test_b)
ax.plot(range(1, mum_components, 1), pca_test_b)
ax.scatter(range(1, mum_components, 1), pca_test_c)
ax.plot(range(1, mum_components, 1), pca_test_c)
ax.legend(labels=['No values removed', 'Capped values removed', 'Outliers removed'], markerscale=0)
ax.plot(np.argmin(pca_test_a) + 1, pca_test_a[np.argmin(pca_test_a)], marker='o', color='red', markersize=12)
ax.plot(np.argmin(pca_test_b) + 1, pca_test_b[np.argmin(pca_test_b)], marker='o', color='red', markersize=12)
ax.plot(np.argmin(pca_test_c) + 1, pca_test_c[np.argmin(pca_test_c)], marker='o', color='red', markersize=12)
ax.set_xlabel("N° of components")
ax.set_ylabel("MSE")