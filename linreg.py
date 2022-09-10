import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import numpy as np
from enum import Enum
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Columns(Enum):
    DATE = "Date"
    USDNOK = "TWI NKSP Index  (R1)"
    OIL = "CO1 Comdty  (L1)"
    STOCKS = "MXWO Index  (R2)"

df = pd.read_excel ("div data.xlsx", sheet_name="2000-2019")


# date_column = df[Columns.DATE.value]

# first order diff
df = df.diff()
df = df.dropna()

# Undo date differencing
# df[Columns.DATE.value] = date_column

# Show
# df.plot.scatter(x=Columns.STOCKS.value, y=Columns.USDEUR.value)
# plt.sh-

# Remove EUR correlation
print(df)

X = df[[Columns.OIL.value, Columns.STOCKS.value]]
Y = df[Columns.USDNOK.value]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,random_state=101)

lm = LinearRegression()
lm.fit(X_train,Y_train)

print(lm.coef_)

prediction = lm.predict(X_test)
plt.scatter(Y_test,prediction)
plt.show()



