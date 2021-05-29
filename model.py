
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Dump a ML model, therefore, we use a pickle format for this
import pickle
import statsmodels.api as sm

df_result = pd.read_csv('products.csv')
#print(df.head())

mod = sm.OLS(exog=df_result[['Positive Tweets', 'Negative Tweets']], endog=df_result['Sales USD'])
res = mod.fit()
#print(res.summary())

X = df_result[['Positive Tweets','Negative Tweets']]
y = df_result['Sales USD']
pred_y = res.predict(X)

# print(y)
# print(pred_y)
# print(res.predict([122,4]))

pickle.dump( res , open('model.pkl','wb') )
sales_model = pickle.load(open('model.pkl','rb'))

#print(sales_model.predict([122,4]))