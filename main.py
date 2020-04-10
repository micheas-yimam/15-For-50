import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import xlsxwriter 
##%matplotlib inline


def linear_reg_predict(x_train,y_train,x_test,y_test):
    regressor = LinearRegression() 
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    diff = y_pred - y_test
    print(y_pred)

def approval_dataset(train_data,test_data):
    y_train = pd.DataFrame(Train_data, columns= ['SignedMargin'])
    y_test = pd.DataFrame(Test_data, columns= ['SignedMargin'])
    states = pd.DataFrame(Test_data, columns= ['state'])
    x_train = pd.DataFrame(Train_data, columns= ['Net Approval Elasticity'])
    x_test = pd.DataFrame(Test_data, columns= ['Net Approval Elasticity'])
    return x_train,y_train,x_test,y_test,states

# def transpose_results(y_pred):
#     with pd.ExcelWriter('data.xlsx') as writer:  
#         for item in y_pred:
#             y_pred[item].to_excel(writer, sheet_name='results')



if __name__ == "__main__":
    

    Train_data = pd.read_excel('data.xlsx', sheet_name = 'Train')
    Test_data = pd.read_excel('data.xlsx', sheet_name = 'Test')
    approval_dataset(Train_data,Test_data)
    x_train,y_train,x_test,y_test,states = approval_dataset(Train_data,Test_data)
    pred_2020 = linear_reg_predict(x_train,y_train,x_test,y_test)
    print(pred_2020)
    #transpose_results(pred_2020)