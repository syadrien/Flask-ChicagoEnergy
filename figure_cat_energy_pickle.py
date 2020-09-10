# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:36:17 2020

@author: Adrien SY
"""

import pandas as pd
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import numpy as np
import os 
from bokeh.io import curdoc, show    
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, VBar
from bokeh.embed import components
import pickle


os.chdir("/app/Processed_data")
# os.chdir("C:/Users/elira/Desktop/DataIncubator/Capstone/Flask-Capstone/Processed_data")

    
# load the dataframe from pickle

file = open("all_data.obj",'rb')
all_data = pickle.load(file)
file.close()

GroupYear=all_data.groupby(['Year Built']).count()

YearlyData=all_data.groupby(['Data Year']).mean()

Energy=YearlyData[['Electricity Use (kBtu)',
 'Natural Gas Use (kBtu)',
 'District Steam Use (kBtu)',
 'District Chilled Water Use (kBtu)',
 'All Other Fuel Use (kBtu)']]


# file = open("all_data_filtered1.obj",'rb')
# all_data_filtered1 = pickle.load(file)
# file.close()


file = open('measuredValue', 'rb')
measuredValue = pickle.load(file)
file.close()



file = open('model_result', 'rb')
model_result = pickle.load(file)
file.close()


rf = pickle.load(open('rf_model','rb'))

def count_building():
    
    ###################################################
    #This is the first figure
    p = figure(title="Count of building reporting energy use over time")
    
    p.title.text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    
    p.vbar(GroupYear.index,bottom=0, top=GroupYear.ID.values, width=0.5,color='blue')
    p.xgrid.grid_line_color = None
    p.yaxis.axis_label='Number of building'
    p.xaxis.axis_label='Year'
    # show(p)
    
    return p


def figure_cat_energy():
    
    ###################################################
    #This is the first figure
    p2 = figure(title="Energy Use for years 2014-2016 for Chicago buildings")
    p2.title.text_font_size = '12pt'
    p2.xaxis.axis_label_text_font_size = "12pt"
    p2.yaxis.axis_label_text_font_size = "12pt"
    
    p2.vbar(Energy.index,bottom=0, top=Energy['Electricity Use (kBtu)'].values, width=0.5,color='orange',legend_label="Electricity")
    p2.vbar(Energy.index,bottom=Energy['Electricity Use (kBtu)'].values, top=Energy['Electricity Use (kBtu)'].values+Energy['Natural Gas Use (kBtu)'].values, width=0.5,fill_color="blue",legend_label="Natural Gas")
    p2.vbar(Energy.index,bottom=Energy['Electricity Use (kBtu)'].values+Energy['Natural Gas Use (kBtu)'].values, top=Energy['Electricity Use (kBtu)'].values+Energy['Natural Gas Use (kBtu)'].values+Energy['District Steam Use (kBtu)'].values, width=0.5,fill_color="green",legend_label="District Steam")
    p2.vbar(Energy.index,bottom=Energy['Electricity Use (kBtu)'].values+Energy['Natural Gas Use (kBtu)'].values+Energy['District Steam Use (kBtu)'].values, top=Energy['Electricity Use (kBtu)'].values+Energy['Natural Gas Use (kBtu)'].values+Energy['District Steam Use (kBtu)'].values+Energy['District Chilled Water Use (kBtu)'].values, width=0.5,fill_color="red",legend_label="District Chilled Water")
    p2.vbar(Energy.index,bottom=Energy['Electricity Use (kBtu)'].values+Energy['Natural Gas Use (kBtu)'].values+Energy['District Steam Use (kBtu)'].values+Energy['District Chilled Water Use (kBtu)'].values, top=Energy['Electricity Use (kBtu)'].values+Energy['Natural Gas Use (kBtu)'].values+Energy['District Steam Use (kBtu)'].values+Energy['District Chilled Water Use (kBtu)'].values+Energy['All Other Fuel Use (kBtu)'].values, width=0.5,fill_color="black",legend_label="All Other Fuel")
    p2.xgrid.grid_line_color = None
    p2.legend.location = "top_right"
    p2.legend.orientation = "vertical"
    p2.yaxis.axis_label='Energy use in kBtu'
    p2.xaxis.axis_label='Year'
    # show(p2)
    
    return p2


def figure_scatter():
    
    ###################################################
    from bokeh.palettes import brewer
    colors = brewer["Spectral"][len(all_data["Data Year"].unique())]
    colormap = {i: colors[i-2018] for i in all_data["Data Year"].unique()}
    colors = [colormap[x] for x in all_data["Data Year"].values]
    all_data['colors'] = colors
    
    
    
    #This is the first figure
    p3 = figure(title="Surface area, Greenhouse gas emission and date")
    p3.circle('Gross Floor Area - Buildings (sq ft)','Total GHG Emissions (Metric Tons CO2e)',legend_label="Data Year", color='colors',source=all_data)
    p3.xgrid.grid_line_color = None
    p3.yaxis.axis_label='Total GHG Emissions (Metric Tons CO2e)'
    p3.xaxis.axis_label='Gross floor area - Buildings (sq ft)'
    p3.title.text_font_size = '12pt'
    p3.xaxis.axis_label_text_font_size = "12pt"
    p3.yaxis.axis_label_text_font_size = "12pt"
    # show(p3)
    
    return p3


def figure_temp():
    
    
    # from sklearn.linear_model import LinearRegression
    from bokeh.models import Label

    #Derived from https://www.weather.gov/lot/Annual_Temperature_Rankings_Chicago
    Year=YearlyData.index
    yearStr=[str(i) for i in list(Year)]
    Temperature=[47.5,50.1,52.3,52.6,50.9]
    Temperature=np.asanyarray(Temperature)
    GHG=np.asfarray(YearlyData['GHG Intensity (kg CO2e/sq ft)'],float)
    # X, Y = Temperature.reshape(-1,1), GHG.reshape(-1,1)
    
    # reggression score
    # reg=LinearRegression().fit(X, Y)
    
    # coef = -1.08234697 / intercept = 66.52943534  / r2 = 0.87
        
    X_line = np.linspace(46,54,10)
    Y_line = np.linspace(46,54,10) * -1.08234697 + 66.52943534
    
    #This is the first figure
    p4 = figure(title='Average GHG Intensity for 2014-2016 and annual temperature | R2=0.87',y_axis_type="log",x_axis_type="log")
    p4.circle(Temperature,GHG,fill_color='blue',radius=0.1)
    p4.line(X_line, Y_line,line_width=5,line_alpha=0.7,color='gray')
    p4.text(Temperature,GHG,text=yearStr)
    p4.yaxis.axis_label='Temperature in °F'
    p4.xaxis.axis_label='GHG Intensity (kg CO2e/sq ft)'
    p4.title.text_font_size = '12pt'
    p4.xaxis.axis_label_text_font_size = "12pt"
    p4.yaxis.axis_label_text_font_size = "12pt"
    # show(p4)
    
    return p4



def perf_RF():
    
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor



    # list_building_type = ['Adult Education',
    # 'Ambulatory Surgical Center',
    #  'Automobile Dealership',
    #  'Bank Branch',
    #  'College/University',
    #  'Convention Center',
    #  'Courthouse',
    #  'Distribution Center',
    #  'Enclosed Mall',
    #  'Financial Office',
    #  'Fitness Center/Health Club/Gym',
    #  'Hospital (General Medical & Surgical)',
    #  'Hotel',
    #  'Ice/Curling Rink',
    #  'Indoor Arena',
    #  'K-12 School',
    #  'Laboratory',
    #  'Library',
    #  'Lifestyle Center',
    #  'Medical Office',
    #  'Mixed Use Property',
    #  'Movie Theater',
    #  'Multifamily Housing',
    #  'Museum',
    #  'Office',
    #  'Other',
    #  'Other - Education',
    #  'Other - Entertainment/Public Assembly',
    #  'Other - Lodging/Residential',
    #  'Other - Mall',
    #  'Other - Public Services',
    #  'Other - Recreation',
    #  'Other - Services',
    #  'Other - Specialty Hospital',
    #  'Outpatient Rehabilitation/Physical Therapy',
    #  'Performing Arts',
    #  'Pre-school/Daycare',
    #  'Prison/Incarceration',
    #  'Repair Services (Vehicle, Shoe, Locksmith, etc.)',
    #  'Residence Hall/Dormitory',
    #  'Residential Care Facility',
    #  'Retail Store',
    #  'Senior Care Community',
    #  'Social/Meeting Hall',
    #  'Stadium (Open)',
    #  'Strip Mall',
    #  'Supermarket/Grocery Store',
    #  'Urgent Care/Clinic/Other Outpatient',
    #  'Wholesale Club/Supercenter',
    #  'Worship Facility']    
    
    
    # # filter outlier for Source EUI (kBtu/sq ft) , then apply a log transformation
    # all_data_X=all_data_filtered1[['log GFA','Year Built','# of Buildings','ZipCodeBIS'] + list_building_type]
    
    # all_data_Y=all_data_filtered1['log Source EUI (kBtu/sq ft)']
    
    
    # X_train, X_test, y_train, y_test = train_test_split(all_data_X, all_data_Y, \
    #                                                     test_size=0.33, random_state=42)
        
    # rf = RandomForestRegressor(n_estimators=1000, min_samples_split=2,max_features='sqrt',max_depth=110, bootstrap=True,random_state = 42)# Train the model on training data
    # rf.fit(X_train, y_train)
    
    # # Loading model to compare the results
    # # see pickle loading at beginning
    
    
    
    # # predict on all the dataset :
    # log_model_result = rf.predict(all_data_X)# Calculate the absolute errors
    
    # # apply exp transf to result to get Source EUI (kBtu/sq ft)
    # model_result = np.exp(log_model_result)
    
    #all_data_filtered1['model_result']=model_result
                      
    A = np.linspace(0,800,100)

    p5 = figure(title='Perfomance of the RF model')
    
    p5.circle(measuredValue,model_result,legend_label='R²: 0.76 | Accuracy: 97.05% | n=9,790')
    p5.line(A, A, color='black', legend_label='Perfect fit')
    p5.xaxis.axis_label = 'Measured source EUI (kBtu/sq ft)'
    p5.yaxis.axis_label ='Modelled source EUI (kBtu/sq ft)'
    p5.title.text = 'Modelling of Source Energy Use Intensity (EUI) using Random Forest'
    p5.xgrid.grid_line_color = 'gray'	
    p5.title.text_font_size = '12pt'
    p5.xaxis.axis_label_text_font_size = "12pt"
    p5.yaxis.axis_label_text_font_size = "12pt"
    # show(p5)
    
    return p5


 

# #prediction of log Source EUI (kBtu/sq ft)
# def prediction(GFA,Year,BuilNumb,Zip,StringBuildingType):
    
    
#     listPara = ['log GFA',
#      'Year Built',
#      '# of Buildings',
#      'ZipCodeBIS',
#      'Adult Education',
#      'Ambulatory Surgical Center',
#      'Automobile Dealership',
#      'Bank Branch',
#      'College/University',
#      'Convention Center',
#      'Courthouse',
#      'Distribution Center',
#      'Enclosed Mall',
#      'Financial Office',
#      'Fitness Center/Health Club/Gym',
#      'Hospital (General Medical & Surgical)',
#      'Hotel',
#      'Ice/Curling Rink',
#      'Indoor Arena',
#      'K-12 School',
#      'Laboratory',
#      'Library',
#      'Lifestyle Center',
#      'Medical Office',
#      'Mixed Use Property',
#      'Movie Theater',
#      'Multifamily Housing',
#      'Museum',
#      'Office',
#      'Other',
#      'Other - Education',
#      'Other - Entertainment/Public Assembly',
#      'Other - Lodging/Residential',
#      'Other - Mall',
#      'Other - Public Services',
#      'Other - Recreation',
#      'Other - Services',
#      'Other - Specialty Hospital',
#      'Outpatient Rehabilitation/Physical Therapy',
#      'Performing Arts',
#      'Pre-school/Daycare',
#      'Prison/Incarceration',
#      'Repair Services (Vehicle, Shoe, Locksmith, etc.)',
#      'Residence Hall/Dormitory',
#      'Residential Care Facility',
#      'Retail Store',
#      'Senior Care Community',
#      'Social/Meeting Hall',
#      'Stadium (Open)',
#      'Strip Mall',
#      'Supermarket/Grocery Store',
#      'Urgent Care/Clinic/Other Outpatient',
#      'Wholesale Club/Supercenter',
#      'Worship Facility']                
    
#     userInput = pd.DataFrame(np.nan,index=[0],columns=listPara)
#     userInput.loc[:,'log GFA']=np.log(GFA)
#     userInput['Year Built']=Year
#     userInput['# of Buildings']=BuilNumb
#     userInput['ZipCodeBIS']=Zip
    
#     # all catgeries set a null
#     for cat in listPara[4:]:
#         userInput[cat]=0

#     userInput[StringBuildingType]=1
    
#     pred_userInput=rf.predict(userInput)
    
#     return pred_userInput
    
