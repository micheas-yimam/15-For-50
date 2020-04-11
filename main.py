import pandas as pd  
import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import xlsxwriter 
import geoplotlib
from geoplotlib.utils import BoundingBox
from geoplotlib.colors import ColorMap
import geopandas as gpd
##%matplotlib inline


def linear_reg_predict(x_train,y_train,x_test,y_test):
    regressor = LinearRegression() 
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    print(y_pred)
    with open('results.txt', 'w') as filehandle:
        for listitem in y_pred:
            filehandle.write('%s\n' % listitem)

def approval_dataset(train_data,test_data):
    y_train = pd.DataFrame(Train_data, columns= ['SignedMargin'])
    y_test = pd.DataFrame(Test_data, columns= ['SignedMargin'])
    states = pd.DataFrame(Test_data, columns= ['state'])
    x_train = pd.DataFrame(Train_data, columns= ['Net Approval Elasticity','National State Difference'])
    x_test = pd.DataFrame(Test_data, columns= ['Net Approval Elasticity','National State Difference'])
    return x_train,y_train,x_test,y_test,states

    ### I need to figure out a way to update excel files
    ### Take out the manual process of some of these steps

def map(final_results):
    final_results = pd.read_excel('data.xlsx', sheet_name = 'Map')
    final_results = pd.DataFrame(final_results, columns=['State','Map','lat','long'])
    print(final_results)
    fp = "tl_2019_us_state.shx"
    map_df = gpd.read_file(fp)
    print(map_df)
    merged = map_df.set_index('NAME').join(final_results.set_index('State'))
    merged.head()
    print(merged)
    # set a variable that will call whatever column we want to visualise on the map
    variable = 'Map'
    # set the range for the choropleth
    vmin, vmax = 120, 220
    # create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(10, 6))
    # create map
    merged.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')


    # geoplotlib.dot(final_results)
    # geoplotlib.show()
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
    final_results = pd.read_excel('data.xlsx', sheet_name = 'Map')
    map(final_results)
    #transpose_results(pred_2020)