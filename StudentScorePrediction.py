# %%
import pandas as pd

# %%
import numpy as np

# %%
import pandas as pd

data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Score": [35, 40, 45, 50, 55, 60, 70, 75, 85, 95]
}

df = pd.DataFrame(data)

print(df)
print("Shape:", df.shape)

# %%
from sklearn.model_selection import train_test_split

# %%
from sklearn.linear_model import LinearRegression

# %%

X=df[["Hours_Studied"]].to_numpy()

# %%
y=df["Score"].to_numpy()

# %%
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=20)

# %%
model=LinearRegression()

# %%
model.fit(X_train, y_train)

# %%
y_pred=model.predict(X_test)

# %%
print("R2 score:", model.score(X_test, y_test))

# %%
print("slope:", model.coef_)

# %%
print("Intercept:", model.intercept_)

# %%
print(model.predict([[5]]))

# %%
import matplotlib.pyplot as plt

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.show()



