import pandas as pd  
import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import geoplotlib
from geoplotlib.utils import BoundingBox
from geoplotlib.colors import ColorMap
from shapely.geometry import Point, Polygon
import geopandas as gpd
##%matplotlib inline


def linear_reg_predict(x_train,y_train,x_test,y_test):
    """ produces an array from the linear regression function
    Args: 
        four arrays; two training sets and two testing sets.
    Returns:
        model predictions for specfied training/testing sets 
    """
    regressor = LinearRegression() 
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    print(y_pred)
    with open('results.txt', 'w') as filehandle:
        for listitem in y_pred:
            filehandle.write('%s\n' % listitem)

def model_v1_dataset(train_data,test_data):
    """ four arrays input and output variables for the regression model
    Args: 
        a training sheet from excel and a testing sheet
    Returns:
        input and output training and testing arrays 
    """
    y_train = pd.DataFrame(Train_data, columns= ['SignedMargin'])
    y_test = pd.DataFrame(Test_data, columns= ['SignedMargin'])
    states = pd.DataFrame(Test_data, columns= ['state'])
    x_train = pd.DataFrame(Train_data, columns= ['Net Approval Elasticity','National State Difference'])
    x_test = pd.DataFrame(Test_data, columns= ['Net Approval Elasticity','National State Difference'])
    return x_train,y_train,x_test,y_test,states

    ### I need to figure out a way to update excel files
def write_result(data,y_pred):
    writer = pd.ExcelWriter(data, engine = 'xlsxwriter')
    print(y_pred)
    y_pred = pd.DataFrame(y_pred)
    print(y_pred)
    y_pred.to_excel(writer, sheet_name = 'python output')
    ### Take out the manual process of some of these steps

def map(final_results):
    """ A visual map utilizing the geopandas library. 
    Args: 
        the predictions for the 12 swing states (testing set y results)
    Returns:
        a visual map of the results
    """
    final_results = pd.read_excel('data.xlsx', sheet_name = 'Map')
    final_results = pd.DataFrame(final_results, columns=['State','Map','lat','long'])
    print(final_results)
    fp = "tl_2019_us_state.shp"
    map_df = gpd.read_file(fp)
    map_df.head()
    print(map_df.head(1))
    merged = map_df.set_index('state name').join(final_results.set_index('State'))
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
    model_v1_dataset(Train_data,Test_data)
    x_train,y_train,x_test,y_test,states = model_v1_dataset(Train_data,Test_data)
    pred_2020 = linear_reg_predict(x_train,y_train,x_test,y_test)
    print(pred_2020)
    final_results = pd.read_excel('data.xlsx', sheet_name = 'Map')
    path_final_result='data.xlsx'
    write_result(path_final_result,pred_2020)
    # Doesnt work yet: map(final_results)
    #transpose_results(pred_2020)