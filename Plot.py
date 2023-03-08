# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 21:43:50 2023

@author: abdwah
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
pd.set_option("display.max_rows", None)
import seaborn as sns


# Load data 
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1,22)] 
col_names = index_names + setting_names + sensor_names
print(col_names)
print(setting_names)
print(sensor_names)




# read data
train = pd.read_csv("C:\\Users\\abdwah\\Predictive_Maintenance\\New_Paper\\A_B\\dataset\\FD001\\train_FD001.txt",sep='\s+', header=None, names=col_names)
test = pd.read_csv("C:\\Users\\abdwah\\Predictive_Maintenance\\New_Paper\\A_B\\dataset\\FD001\\test_FD001.txt",sep='\s+', header=None, names=col_names)
y_test = pd.read_csv("C:\\Users\\abdwah\\Predictive_Maintenance\\New_Paper\\A_B\\dataset\\FD001\\RUL_FD001.txt", sep='\s+', header=None, names=['RUL'])
print(y_test)

# Train data contains all features (Unit Number + setting parameters & sensor parameters)
# Test data contains all features (Unit Number + setting parameters & sensor parameters)
# Y_test contains RUL for the test data.

print(train.head(5))

test_df = test.groupby('unit_nr').agg({'time_cycles':'max'})

print(test_df)

y_test['time_cycles'] = test_df['time_cycles']
train_df= train.groupby('unit_nr').agg({'time_cycles':'max'})


# calculate and add RUL(Remaining Useful Life) to the train dataset.
def add_remaining_useful_life(df, max_df):
    # Get the total number of cycles for each unit
    max_cycle= max_df['time_cycles']
    
    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    
    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame
  
train = add_remaining_useful_life(train,train_df)
test=add_remaining_useful_life(test,test_df)
test[index_names+['RUL']].head()

data=train.copy()
test_data=test.copy()
#print(test_data)


test_data['increasing'] = test_data['s_2']+ test_data['s_3']+ test_data['s_4']+test_data['s_8']+test_data['s_13']+ test_data['s_15']+test_data['s_17']
data['increasing'] = data['s_2']+ data['s_3']+ data['s_4']+data['s_8']+data['s_13']+ data['s_15']+data['s_17']

data['decreasing'] = data['s_7']+data['s_12'] + data['s_20'] + data['s_21']
test_data['decreasing']=test_data['s_7']+test_data['s_12'] + test_data['s_20'] + test_data['s_21']


def plot_sensor(sensor_name,X):
    plt.figure(figsize=(13,5))
    for i in X['unit_nr'].unique():
        if (i % 10 == 0):  # only plot every engine
            plt.plot('RUL', sensor_name, 
                     data=X[X['unit_nr']==i])
            #plt.axvline(60, color='red', linestyle='dashed', linewidth=4)
    plt.xlim(250, 0)  # reverse the x-axis so RUL counts down to zero
    plt.xticks(np.arange(0, 275, 25))
    plt.ylabel(sensor_name)
    plt.xlabel('Remaining Use fulLife')
    plt.show()
    
      
plot_sensor('increasing',data)
plot_sensor('increasing',test_data)
plot_sensor('decreasing',data)
plot_sensor('decreasing',test_data)




from scipy.signal import savgol_filter

#Check correlation for the increasing trend sensors first

def get_correlation_plots(component,a,train_PCA,x=250,y=25):
    plt.figure(figsize=(10,4))
    for unit_nr in train_PCA.unit_nr.unique():
        if unit_nr%10==0:
            data= train_PCA[train_PCA['unit_nr']==unit_nr]
            y1=data[component].ewm(com=0.1).mean()
            y1= savgol_filter(y1, a, 3)
            #plt.figure(figsize=(10,4))
            plt.plot(data['RUL'],y1)
            #plt.plot(random['RUL'], random['ssensor_2'])
            plt.xlim(x, 0)  # reverse the x-axis so RUL counts down to zero
            plt.xticks(np.arange(0, x, y))
            plt.ylabel('Exponential Weighted Moving Average')
            plt.xlabel('Remaining Use fulLife')
            plt.grid(True)
    plt.show()
get_correlation_plots('increasing',21,data)
get_correlation_plots('increasing',21,test_data)
get_correlation_plots('decreasing',11,data)
get_correlation_plots('decreasing',21,test_data)
get_correlation_plots('increasing',15,data[data['RUL']<21],21,1)




corr_train=[]

for unit_nr in data.unit_nr.unique():
    X= data[data['unit_nr']==unit_nr]
    x1=X['increasing'].ewm(com=0.5).mean()
    x1= savgol_filter(x1, 27, 3)
    corr_train.append(x1)



corr_test=[]

for unit_nr in test_data.unit_nr.unique():
    data1= test_data[test_data['unit_nr']==unit_nr]
    y1=data1['increasing'].ewm(com=0.5).mean()
    y1= savgol_filter(y1, 27, 3)
    corr_test.append(y1)


def get_flat_list(comp):
    flat_list = [item for sublist in comp for item in sublist]
    return flat_list

corr_train=get_flat_list(corr_train)

corr_test=get_flat_list(corr_test)


data['EWM']= corr_train

test_data['EWM']=corr_test

data['week']= (data['RUL']/7).astype(int)

test_data['week']= (data['RUL']/7).astype(int)

plt.figure(figsize=(25,6))

sns.boxplot('week','EWM',data=data)
plt.xlim(48,0)
plt.axhline(8830.59,color='black', linestyle='dashed', linewidth=1.5)

#plt.axhline(1139.63,color='black', linestyle='dashed', linewidth=1.5)
plt.title('RUL for risky EWMA per week')


plt.figure(figsize=(25,6))

sns.boxplot('RUL','EWM',data=data[data['RUL']<21])
plt.xlim(21,0)
plt.axhline(8850,color='black', linestyle='dashed', linewidth=1.5)

#plt.axhline(1139.63,color='black', linestyle='dashed', linewidth=1.5)
plt.title('RUL for risky EWMA per week')


plt.figure(figsize=(25,6))

sns.boxplot('week','EWM',data=test_data)
plt.xlim(40,0)
plt.axhline(8830.59,color='black', linestyle='dashed', linewidth=1.5)

#plt.axhline(1139.63,color='black', linestyle='dashed', linewidth=1.5)
plt.title('RUL for risky EWMA per week')


data['Health']='Healthy'

test_data['Health']='Healthy'

def get_health(data):
    for i in range(len(data)):
        if data.loc[i,'RUL']<30:
            if data.loc[i,'EWM']>8830:
                data.loc[i,'Health']='Unhealthy'
            else:
                data.loc[i,'Health']= 'Health cannot be determined'
                time_cycles= data.loc[i,'time_cycles']
                unit_nr= data.loc[i,'unit_nr']
                print(f'Cannot determine health for unit_nr time_cycles Check manually for Engine {unit_nr}: Cycle {time_cycles}')
    #data.dropna(inplace= True)
    return data


data= get_health(data)


plt.figure(figsize=(10,10))

# Data to plot
labels= 'Healthy','Unhealthy','Undetermined'
sizes = data['Health'].value_counts()
colors = ['indianred','gold','cyan']
explode = (0.1, 0,0.2)  # explode 3rd slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.title('Distribution of Health Cycles')
plt.show()






















# plt.figure(figsize=(10,10))

# # Data to plot
# states =['Healthy','Unhealthy','Indefinite']
# labels= list(states)
# print(labels)
# sizes = data['Health'].value_counts()
# colors = ['red','gold','darkred']
# explode = (0.1, 0,0.2)  # explode 3rd slice

# # Plot
# plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

# plt.axis('equal')

# plt.title('Distribution of Health Cycles')
# plt.show()





plt.figure(figsize=(25,18))
sns.heatmap(train.corr(),annot=True ,cmap='PiYG')
plt.show()










# # total number of engines
# train['unit_nr'].unique()
# y_test.shape

# train.describe()

# # drop setting_3 as it does not change
# train=train.drop('setting_3',axis=1)


# # calculate and add RUL(Remaining Useful Life) to the train dataset.
# def add_remaining_useful_life(df):
#     # Get the total number of cycles for each unit
#     grouped_by_unit = df.groupby(by="unit_nr")
#     max_cycle = grouped_by_unit["time_cycles"].max()
    
#     # Merge the max cycle back into the original frame
#     result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    
#     # Calculate remaining useful life for each row
#     remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
#     result_frame["RUL"] = remaining_useful_life
    
#     # drop max_cycle as it's no longer needed
#     result_frame = result_frame.drop("max_cycle", axis=1)
#     return result_frame
# train = add_remaining_useful_life(train)
# train[sensor_names+['RUL']].head()


# # plot RUL 
# df_max_rul = train[['unit_nr', 'RUL']].groupby('unit_nr').max().reset_index()
# df_max_rul['RUL'].hist(bins=20, color = 'skyblue', ec='red', alpha=0.7, rwidth=0.85, figsize=(15,7))
# plt.xlabel('RUL')
# plt.ylabel('frequency')
# plt.show()


# # plot sensors with RUL realtions

# def plot_sensor(sensor_name):
#     plt.figure(figsize=(13,5))
#     for i in train['unit_nr'].unique():
#         if (i % 10 == 0):  # only plot every 10th unit_nr
#             plt.plot('RUL', sensor_name, 
#                      data=train[train['unit_nr']==i])
#     plt.xlim(250, 0)  # reverse the x-axis so RUL counts down to zero
#     plt.xticks(np.arange(0, 275, 25))
#     plt.ylabel(sensor_name)
#     plt.xlabel('Remaining Use fulLife')
#     plt.show()
# for sensor_name in sensor_names:
#     plot_sensor(sensor_name)

# extract the important features which have strong corelation


















