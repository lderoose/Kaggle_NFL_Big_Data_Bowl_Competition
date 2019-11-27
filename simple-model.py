import pandas as pd
import numpy as np
from scipy.stats import norm
from kaggle.competitions import nflrush

env = nflrush.make_env()

def normSmooth(pred):
    window, std = 30, 4.1
    prediction = np.zeros(199)
    prediction[int(pred)+99:] = 1
    depart = int((int(pred)+99)-(window/2))
    fin = int((int(pred)+99)+(window/2))
    quantile = np.arange(start = int(pred)-(window/2), stop=int(pred)+(window/2), step=1) 
    array = norm.cdf(quantile, loc=int(pred), scale=std)
    prediction[depart:fin] = array
    return prediction.reshape(1,199)

yard = 3
prediction = pd.DataFrame(normSmooth(pred = np.array(yard)), columns=['Yards'+str(i) for i in range(-99,100)])
i=0
for (test_df, sample_sub) in env.iter_test():
    i+=1
    print('Shape', test_df.shape)
    print(sample_sub.shape)
    print(prediction.shape)
    prediction = pd.DataFrame(normSmooth(pred = np.array(yard)), columns=['Yards'+str(i) for i in range(-99,100)])
    prediction.index = [test_df['PlayId'].values[0]]
    prediction.index.name='PlayId'
    env.predict(prediction)
env.write_submission_file()

