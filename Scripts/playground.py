# script that is a playground for exploring dynamic stock modeling

# import libraries
from dynamic_stock_model import DynamicStockModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

data = pd.read_csv('/Users/josepharehart/PycharmProjects/Bldg_Stock/InputData/dummy_data.csv')

# Implement a stock-driven model
Years = data.Year
TotalBldgStock = data.Floor_Area
BldgLife_mean = data.AverageLifetime_yrs

# Build the DSM data class
US_Bldg_DSM = DynamicStockModel(t = Years,
                                s = TotalBldgStock,
                                lt={'Type': 'Normal', 'Mean': np.array(BldgLife_mean),
                                    'StdDev': 0.3 * np.array(BldgLife_mean)})
CheckStr, ExitFlag = US_Bldg_DSM.dimension_check()
print(CheckStr)


S_C, O_C, I, ExitFlag = US_Bldg_DSM.compute_stock_driven_model()
# S_C: Stock by cohort
# O_C: Outflow by cohort
# I: inflow (construction of buildings)

O, ExitFlag   = US_Bldg_DSM.compute_outflow_total() # Total outflow
DS, ExitFlag  = US_Bldg_DSM.compute_stock_change()  # Stock change
Bal, ExitFlag = US_Bldg_DSM.check_stock_balance()   # Vehicle balance

print(np.abs(Bal).sum()) # show sum absolute of all mass balance mismatches.

plt2, = plt.plot(US_Bldg_DSM.t, US_Bldg_DSM.s)
plt4, = plt.plot([2020,2020],[0,8e4], color = 'k', LineStyle = '--')
plt.legend([plt2], ['Stock'], loc = 2)
plt.xlabel('Year')
plt.ylabel('Floor Area')
plt.title('Floor Area Stock')
plt.show();

plt1, = plt.plot(US_Bldg_DSM.t, US_Bldg_DSM.i)
plt3, = plt.plot(US_Bldg_DSM.t, US_Bldg_DSM.o)
plt4, = plt.plot([2020,2020],[0,1e4], color = 'k', LineStyle = '--')
plt.xlabel('Year')
plt.ylabel('Floor Area per year')
plt.title('Floor Area flows')
plt.legend([plt1,plt3], ['Inflow','Outflow'], loc = 2)
plt.show();

data_pop = pd.read_csv('/Users/josepharehart/PycharmProjects/Bldg_Stock/InputData/USA_pop_forecast.csv')

# Create interpolation
f_median = interp1d(data_pop.Year, data_pop.Median)
f_upper_95 = interp1d(data_pop.Year, data_pop.Upper_95)
f_lower_95 = interp1d(data_pop.Year, data_pop.Lower_95)
f_upper_80 = interp1d(data_pop.Year, data_pop.Upper_80)
f_lower_80 = interp1d(data_pop.Year, data_pop.Lower_80)

# List of years
years = np.linspace(1990, 2100, num=111, endpoint=True)
US_pop = f_median(years)

# Plot of population forecasts
plt1, = plt.plot(years, f_upper_95(years))
plt2, = plt.plot(years, f_upper_80(years))
plt3, = plt.plot(years, f_median(years))
plt4, = plt.plot(years, f_lower_80(years))
plt5, = plt.plot(years, f_lower_95(years))
plt6, = plt.plot([2020,2020],[2.4e8,4.5e8], color = 'k', LineStyle = '--')
plt.legend([plt1,plt2,plt3,plt4,plt5],
           ['Upper 95th','Upper 80th','Median','Lower 80th','Lower 95th'],
           loc=2)
plt.xlabel('Year')
plt.ylabel('US Population')
plt.title('Historical and Forecast of Population in the US')
plt.show();





