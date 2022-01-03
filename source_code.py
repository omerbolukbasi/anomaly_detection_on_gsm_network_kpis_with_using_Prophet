#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot


# https://facebook.github.io/prophet/docs/trend_changepoints.html
# 
# https://facebook.github.io/prophet/docs/saturating_forecasts.html
# 

# In[50]:


# Read the data into a pandas data frame.
df = pd.read_csv("volte_interconnect.csv",sep=";")
df["DT"] = pd.to_datetime(df.DT,format="%d.%m.%Y")


# In[51]:


df.head()


# In[52]:


df.set_index("DT").plot()


# In[53]:


df.info()


# In[54]:


# Replace the column name of date with "ds" and actual values with "y".
df.columns = ["ds","y"]

# define the model
model = Prophet(changepoint_prior_scale=0.5)

# fit the model
model.fit(df)


# In[55]:


# Future values can be predicted with the following code.
future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)
fig = model.plot(forecast)


# In[56]:


forecast


# In[57]:


# Plot the predicted trend change points.
fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast)


# In[58]:


# Read real GSM network data.
df2 = pd.read_csv("S1_MME_S1_MME_20211228094935.csv")
df2["Time"] = df2["Time"].str.split("\t").str[1]
df2["Time"] = pd.to_datetime(df2["Time"])
df2.columns


# In[59]:


# Implementation of trend detection with real network KPI's.
mme = "ANHVUSN03"
kpi = "\tS1-MME Attach Success Rate(%)"
df3 = df2.pivot(index="Time",columns="MME",values=kpi)


# In[60]:


df3[[mme]].plot(figsize=(17,7))


# In[61]:


df_anomally = df3[[mme]]
df_anomally = df_anomally.reset_index()

# Change column names.
df_anomally.columns = ["ds","y"]
# define the model
model = Prophet(changepoint_prior_scale=0.5,weekly_seasonality=False,daily_seasonality=True,yearly_seasonality=False)
# fit the model
model.fit(df_anomally)


# In[62]:


# Extract the upper and lower bond in order to detect anomalies.
future = model.make_future_dataframe(periods=0)
forecast = model.predict(future)
fig = model.plot(forecast)


# In[63]:


# Detect the pattern changes. 
fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast,threshold = 0.7)


# In[64]:


# Implementation of trend detection with real network KPI's.

mme = "AVHVUSN02"
kpi = "\tS1-MME Attach Success Rate(%)"
df3 = df2.pivot(index="Time",columns="MME",values=kpi)
df_anomally = df3[[mme]]
df_anomally = df_anomally.reset_index()

df3[[mme]].plot(figsize=(17,7))


# In[65]:


# Cahnge the column names.
df_anomally.columns = ["ds","y"]

# Define the model
model = Prophet(changepoint_prior_scale=0.5,weekly_seasonality=False,daily_seasonality=True,yearly_seasonality=False)

# Fit the model
model.fit(df_anomally)
future = model.make_future_dataframe(periods=0)
forecast = model.predict(future)

#Detect the change points.
fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast,threshold = 0.6)


# In[66]:


# Implementation of trend detection with real network KPI's.
mme = "AVHVUSN02"
kpi = "\tS1-MME Combined EPS/IMSI Attach Delay(ms)"
df3 = df2.pivot(index="Time",columns="MME",values=kpi)
df_anomally = df3[[mme]]
df_anomally = df_anomally.reset_index()

df3[[mme]].plot(figsize=(17,7))


# In[67]:


# Detect the trend changes. 
df_anomally.columns = ["ds","y"]

# define the model
model = Prophet(changepoint_prior_scale=0.5,weekly_seasonality=False,daily_seasonality=True,yearly_seasonality=False)

# fit the model
model.fit(df_anomally)
future = model.make_future_dataframe(periods=0)
forecast = model.predict(future)
fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast,threshold = 0.6)


# In[68]:


# Change points:
a

