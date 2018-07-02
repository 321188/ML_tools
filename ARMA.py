#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


date_period = 500
date_index = pd.date_range(start='2000-01-01', periods=date_period, freq='D')
print(date_index)

data_index = np.arange(start=1, stop = date_period +1)
print(data_index)


### AR(2) -- Y1
Y1_lag = 0
Y1_lag2 = 0
Y1_err = 0
Y1_err2 = 0
Y1 = np.zeros(date_period)

for i in data_index:
    error = np.random.normal()
    y1 = 0.1*Y1_lag + 0.72*Y1_lag2 + error
    Y1[i-1]= y1
    Y1_err2 = Y1_err
    Y1_err = error
    Y1_lag2 = Y1_lag
    Y1_lag = y1
print(Y1)


### MA(2) -- Y2
Y2_err = 0
Y2_err2 = 0
Y2 = np.zeros(date_period)

for i in data_index:
    error = np.random.normal()
    y2 = error - 0.1*Y2_err - 0.72*Y2_err2
    Y2[i-1]= y2
    Y2_err2 = Y2_err
    Y2_err = error
print(Y2)


### ARMA(1,1) -- Y3
Y3_lag = 0
Y3_err = 0
Y3 = np.zeros(date_period)

for i in data_index:
    error = np.random.normal()
    y3 = 0.9*Y3_lag + error + 0.8*Y3_err
    Y3[i-1] = y3
    Y3_lag = y3
    Y3_err = error
print(Y3)


### merge Y1, Y2, Y3
ts_raw_df = pd.DataFrame(
    index=date_index,
    data={
        't':data_index,
        'Y1': Y1,
        'Y2': Y2,
        'Y3':Y3
    }
)
ts_demo_df = ts_raw_df[ (ts_raw_df['t']>100) & (ts_raw_df['t'] < (date_period-4))]


Y1_ts = ts_demo_df.Y1
Y2_ts = ts_demo_df.Y2
Y3_ts = ts_demo_df.Y3


### plot ACF PACF
plt.rcParams["font.size"] = 10
fig, ax = plt.subplots(3, 3, figsize=(20, 10))
axes = ax.flatten()
diff_color = "#800000"

axes[0].plot(Y1_ts)
axes[0].set_title("Y1_AR(2)")
fig = sm.graphics.tsa.plot_acf(Y1_ts, lags=40, ax=axes[1])
fig = sm.graphics.tsa.plot_pacf(Y1_ts, lags=40, ax=axes[2])

axes[3].plot(Y2_ts, color=diff_color, alpha=1)
axes[3].set_title("Y2_MA(2)")
fig = sm.graphics.tsa.plot_acf(Y2_ts, lags=40, ax=axes[4], color=diff_color)
fig = sm.graphics.tsa.plot_pacf(Y2_ts, lags=40, ax=axes[5], color=diff_color)

axes[6].plot(Y3_ts)
axes[6].set_title("Y3_ARMA(1,1)")
fig = sm.graphics.tsa.plot_acf(Y3_ts, lags=40, ax=axes[7] )
fig = sm.graphics.tsa.plot_pacf(Y3_ts, lags=40, ax=axes[8])


### estimate time series model and predict
pred_ts = ts_raw_df[ts_raw_df['t']> (date_period-5)].index.date
print(pred_ts)


fig, ax = plt.subplots(figsize=(8, 4))
Y1_model = sm.tsa.ARMA(Y1_ts,order=(2,0)).fit(disp=False)
print(Y1_model.summary())
Y1_predict = Y1_model.forecast(steps=5)[0]
print(Y1_predict)
ax.plot(pred_ts, Y1_predict)
Y1_real = ts_raw_df[ts_raw_df['t']> (date_period-5)].Y1
ax.plot(pred_ts, Y1_real)
ax.legend(["predict","real"], loc="best")
fig.suptitle('Y1 5 steps prediction')
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
Y2_model = sm.tsa.ARMA(Y2_ts,order=(0,2)).fit(disp=False)
print(Y2_model.summary())
Y2_predict = Y2_model.forecast(steps=5)[0]
print(Y2_predict)
ax.plot(pred_ts, Y2_predict)
Y2_real = ts_raw_df[ts_raw_df['t']> (date_period-5)].Y2
ax.plot(pred_ts, Y2_real)
ax.legend(["predict","real"], loc="best")
fig.suptitle('Y2 5 steps prediction')
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
Y3_model = sm.tsa.ARMA(Y3_ts,order=(1,1)).fit(disp=False)
print(Y3_model.summary())
Y3_predict = Y3_model.forecast(steps=5)[0]
print(Y3_predict)
ax.plot(pred_ts, Y3_predict)
Y3_real = ts_raw_df[ts_raw_df['t']> (date_period-5)].Y3
ax.plot(pred_ts, Y3_real)
ax.legend(["predict","real"], loc="best")
fig.suptitle('Y3 5 steps prediction')
plt.show()



