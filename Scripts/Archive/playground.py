# script that is a playground for exploring dynamic stock modeling

# import libraries
# from dynamic_stock_model import DynamicStockModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# import odym.modules.dynamic_stock_model as dsm
# import odym.modules.ODYM_Functions as msf
# import odym.modules.ODYM_Classes as msc


from odym import dynamic_stock_model as dsm

# Load in datasets
RECS_Area = pd.read_csv('/Users/josepharehart/PycharmProjects/Bldg_Stock/InputData/RECS_Area.csv')
RECS_Weights = pd.read_csv('/Users/josepharehart/PycharmProjects/Bldg_Stock/InputData/RECS_Area_weights.csv')
CBECS_Area = pd.read_csv('/Users/josepharehart/PycharmProjects/Bldg_Stock/InputData/CBECS_Area.csv')
CBECS_Weights = pd.read_csv('/Users/josepharehart/PycharmProjects/Bldg_Stock/InputData/RECS_Area_weights.csv')

data_pop = pd.read_csv('/Users/josepharehart/PycharmProjects/Bldg_Stock/InputData/USA_pop_forecast.csv')

# Create interpolation
f_median = interp1d(data_pop.Year, data_pop.Median,     kind='cubic')
f_upper_95 = interp1d(data_pop.Year, data_pop.Upper_95, kind='cubic')
f_lower_95 = interp1d(data_pop.Year, data_pop.Lower_95, kind='cubic')
f_upper_80 = interp1d(data_pop.Year, data_pop.Upper_80, kind='cubic')
f_lower_80 = interp1d(data_pop.Year, data_pop.Lower_80, kind='cubic')

# Study Period
year1 = 1900
year2 = 2100
years = np.linspace(year1, year2, num=(year2-year1+1), endpoint=True)

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

# Floor Area elasticity (residential and commercial)
FA_elasticity_res = 200
FA_elasticity_res_vec = np.array([FA_elasticity_res] * len(years))
FA_elasticity_com = 140
FA_elasticity_com_vec = np.array([FA_elasticity_com] * len(years))

# Population forecast
US_pop = f_median(years)
# FA_stock = np.multiply(US_pop, FA_elasticity_res)
FA_stock = US_pop * FA_elasticity_res

# Residential input data
res_input_data = pd.DataFrame({'Year':years,
                               'US_pop':US_pop,
                               'FA_elasticity':FA_elasticity_res_vec,
                               'FA_stock':FA_stock},)

# Building lifespan distributions
# Normal
BldgLife_mean = 80     # years
BldgLife_mean_vec = [BldgLife_mean] * len(years)     # years
lifetime_NormalLT = {'Type': 'Normal', 'Mean': np.array(BldgLife_mean_vec), 'StdDev': 0.3*np.array(BldgLife_mean_vec)}

# Weibull
lifetime_WeibullLT = {'Type': 'Weibull', 'Shape': np.array([5.5]), 'Scale': np.array([85.5])}
# lifetime_WeibullLT = {'Type': 'Weibull', 'Shape': np.array([1.47]), 'Scale': np.array([37.64])}

# Gamma (currently not working)
# lifetime_GammaLT = {'Type': 'Gamma', 'Scale': np.array([2.7]), 'Shape': np.array([32.9])}

# Lognormal (currently not working)
# lifetime_LogNormalLT = {'Type': 'LogNormal', 'Mean': np.array([4]), 'StdDev': np.array([80])}

myLT = lifetime_WeibullLT

# Residential floor area  age-cohort in base year: 2015
# S_0_res_2015 = [100] * len(years)
S_0_res_2015 = np.flipud(RECS_Weights.Res_Weight * US_pop[115]*FA_elasticity_res)      # square meters of res by age in 2015

# Implement a stock-driven model
US_Bldg_DSM = dsm.DynamicStockModel(t=years, s=FA_stock, lt=myLT)

CheckStr = US_Bldg_DSM.dimension_check()
print(CheckStr)

S_C = US_Bldg_DSM.compute_evolution_initialstock(InitialStock=S_0_res_2015, SwitchTime=116)
S_C, O_C, I = US_Bldg_DSM.compute_stock_driven_model()
O   = US_Bldg_DSM.compute_outflow_total() # Total outflow
DS  = US_Bldg_DSM.compute_stock_change()  # Stock change
Bal = US_Bldg_DSM.check_stock_balance()   # Vehicle balance
print(np.abs(Bal).sum()) # show sum absolute of all mass balance mismatches.





# S_C, O_C, I = US_Bldg_DSM.compute_stock_driven_model_initialstock(InitialStock=S_0_res_2015[0:115],
#                                                     SwitchTime=116,
#                                                     NegativeInflowCorrect=True)
# O   = US_Bldg_DSM.compute_outflow_total() # Total outflow
# DS  = US_Bldg_DSM.compute_stock_change()  # Stock change
# Bal = US_Bldg_DSM.check_stock_balance()   # Vehicle balance
# print(np.abs(Bal).sum()) # show sum absolute of all mass balance mismatches.

## Old code
# S_C = US_Bldg_DSM.compute_evolution_initialstock(InitialStock=S_0_res,SwitchTime=2015)
# S_C, O_C, I, ExitFlag = US_Bldg_DSM.compute_stock_driven_model()
# S_C: Stock by cohort
# O_C: Outflow by cohort
# I: inflow (construction of buildings)

# O   = US_Bldg_DSM.compute_outflow_total() # Total outflow
# DS  = US_Bldg_DSM.compute_stock_change()  # Stock change
# Bal = US_Bldg_DSM.check_stock_balance()   # Vehicle balance
# print(np.abs(Bal).sum()) # show sum absolute of all mass balance mismatches.

plt2, = plt.plot(US_Bldg_DSM.t, US_Bldg_DSM.s)
plt4, = plt.plot([2020,2020],[0,1.4e11], color = 'k', LineStyle = '--')
plt.legend([plt2], ['Stock'], loc = 2)
plt.xlabel('Year')
plt.ylabel('Floor Area')
plt.title('Floor Area Stock')
plt.show();

plt1, = plt.plot(US_Bldg_DSM.t, US_Bldg_DSM.i)
plt3, = plt.plot(US_Bldg_DSM.t, US_Bldg_DSM.o)
plt4, = plt.plot([2020,2020],[0,1e4], color = 'k', LineStyle = '--')
plt.xlim(left=1905)
# plt.ylim(bottom=0, top=2.5e9)
plt.xlabel('Year')
plt.ylabel('Floor Area per year')
plt.title('Floor Area flows')
plt.legend([plt1,plt3], ['Inflow','Outflow'], loc = 2)
plt.show();

# Stock by age-cohort
plt.imshow(US_Bldg_DSM.s_c[:,1:],interpolation='nearest')   # exclude the first column to have the color scale work.
plt.xlabel('age-cohort')
plt.ylabel('year')
plt.title('Stock by age-cohort')
plt.show();

# -------- OLD CODE --------
# -----
# Implement a stock-driven model with evolution of initial stock
# US_Bldg_DSM = dsm.DynamicStockModel(t=years, s=FA_stock_res, lt=myLT)
# CheckStr = US_Bldg_DSM.dimension_check()
# print(CheckStr)
# S_C = US_Bldg_DSM.compute_stock_driven_model_initialstock(InitialStock=S_0_res_2015[0:115], SwitchTime=116)
# O = US_Bldg_DSM.compute_outflow_total()  # Total outflow
# DS = US_Bldg_DSM.compute_stock_change()  # Stock change
# Bal = US_Bldg_DSM.check_stock_balance()  # Vehicle balance
# print(np.abs(Bal).sum())  # show sum absolute of all mass balance mismatches.



# S_C, O_C, I = US_Bldg_DSM.compute_stock_driven_model()
# S_C, O_C, I = US_Bldg_DSM.compute_stock_driven_model_initialstock(InitialStock=S_0_res_2015[0:115],
#                                                     SwitchTime=116,
#                                                     NegativeInflowCorrect=True)
# O   = US_Bldg_DSM.compute_outflow_total() # Total outflow
# DS  = US_Bldg_DSM.compute_stock_change()  # Stock change
# Bal = US_Bldg_DSM.check_stock_balance()   # Vehicle balance
# print(np.abs(Bal).sum()) # show sum absolute of all mass balance mismatches.

## Old code
# S_C = US_Bldg_DSM.compute_evolution_initialstock(InitialStock=S_0_res,SwitchTime=2015)
# S_C, O_C, I, ExitFlag = US_Bldg_DSM.compute_stock_driven_model()
# S_C: Stock by cohort
# O_C: Outflow by cohort
# I: inflow (construction of buildings)

# O   = US_Bldg_DSM.compute_outflow_total() # Total outflow
# DS  = US_Bldg_DSM.compute_stock_change()  # Stock change
# Bal = US_Bldg_DSM.check_stock_balance()   # Vehicle balance
# print(np.abs(Bal).sum()) # show sum absolute of all mass balance mismatches.



