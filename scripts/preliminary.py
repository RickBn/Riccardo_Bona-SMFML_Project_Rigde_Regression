from scripts.functions import *
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

df = pd.read_csv('data/cal-housing.csv')

#General histogram of the dataframe
df.hist(bins=100, figsize=(15, 10))

#Plot ocean proximity by coordinate and barplot
plot_ocean_proximity(df)
df['ocean_proximity'].value_counts().plot(kind='bar', color=['orange', 'dodgerblue', 'red', 'green', 'violet'], title='ocean_proximity')

ocean_proximity = df['ocean_proximity'].copy()
df = df.drop(['ocean_proximity'], axis=1)
df = df.reset_index()
df = df.drop(['index'], axis=1)

features = df.columns

plt.hist(df['total_bedrooms'], bins=100)
plt.title('total_bedrooms')

imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(df)
df = imp.transform(df)

df = pd.DataFrame(df, columns=features)
df = pd.concat((df, ocean_proximity), axis=1)

#Boxplots for each column and outliers count
fig, axes = plt.subplots(1, len(df.columns[:-1]))
for i, col in enumerate(df.columns[:-1]):
    ax = sns.boxplot(y=df[col], ax=axes.flatten()[i])
    ax.set_ylim(df[col].min(), df[col].max())
    ax.set_xlabel(col)
    ax.set_ylabel(None)
    ax.tick_params(axis='y', rotation=90)
plt.show()

# Number of outliers per column
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
iqr = q3 - q1
((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).sum()

# Correlation map
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(df.corr(), mask=mask, cmap=cmap, center=0, robust=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

# VIF computation and collinear features removal
X, y, x_features, X_train, X_test, y_train, y_test = processData(df, scaler=StandardScaler(), intercept=False)
vif = pd.DataFrame([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], X.columns)
X = remove_by_vif(X)

mask = np.triu(np.ones_like(X.corr(), dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(X.corr(), mask=mask, cmap=cmap, center=0, robust=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

vif = pd.DataFrame([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], X.columns)
