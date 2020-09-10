# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:47:04 2020

@author: Adrien SY
"""

from flask import Flask, render_template, request
from bokeh.embed import components
from bokeh.layouts import column
from bokeh.layouts import gridplot

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])

    
def homepage():

    if request.method == 'POST':

        # GFA,Year,BuilNumb,Zip,StringBuildingType = request.form['GFA','Year','BuilNumb','Zip','StringBuildingType']
        
        
        # create the 1st plot
        from figure_cat_energy_pickle import figure_cat_energy
        p1 = figure_cat_energy()
        
        # create the second plot 
        from figure_cat_energy_pickle import count_building
        p2 = count_building()
        
        
        # create the third plot 
        from figure_cat_energy_pickle import figure_scatter
        p3 = figure_scatter()
        
        
        # create the 4th plot 
        from figure_cat_energy_pickle import figure_temp
        p4 = figure_temp()
        
        
        # create the 4th plot 
        from figure_cat_energy_pickle import perf_RF
        p5 = perf_RF()
        
        
        # from figure_cat_energy_pickle import prediction
        # p6 = prediction(GFA,Year,BuilNumb,Zip,StringBuildingType)
        
            
        # script1, div1 = components(gridplot([[p2, p1], [p4, p3]], plot_width=650, plot_height=650))
        script, div = components(gridplot([[p2, p1], [p4, p3]], plot_width=650, plot_height=650))
        script5, div5 = components(p5)
    
    
    
        return render_template('index.html', div=div, script=script,div5=div5, script5=script5)
    
    else:
        
        
        # GFA,Year,BuilNumb,Zip,StringBuildingType = request.form['GFA','Year','BuilNumb','Zip','StringBuildingType']
        
        
        # create the 1st plot
        from figure_cat_energy_pickle import figure_cat_energy
        p1 = figure_cat_energy()
        
        # create the second plot 
        from figure_cat_energy_pickle import count_building
        p2 = count_building()
        
        
        # create the third plot 
        from figure_cat_energy_pickle import figure_scatter
        p3 = figure_scatter()
        
        
        # create the 4th plot 
        from figure_cat_energy_pickle import figure_temp
        p4 = figure_temp()
        
        
        # create the 4th plot 
        from figure_cat_energy_pickle import perf_RF
        p5 = perf_RF()
        
        
        # from figure_cat_energy_pickle import prediction
        # p6 = prediction(GFA,Year,BuilNumb,Zip,StringBuildingType)
        
            
        # script1, div1 = components(gridplot([[p2, p1], [p4, p3]], plot_width=650, plot_height=650))
        script, div = components(gridplot([[p2, p1], [p4, p3]], plot_width=650, plot_height=650))
        script5, div5 = components(p5)
        
        return render_template('index.html', div=div, script=script,div5=div5, script5=script5)



if __name__ == '__main__':
    app.run(port=33507)
