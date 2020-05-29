# Explore why the GFA is much higher than other studies

import pandas as pd
import matplotlib.pyplot as plt
from odym import dynamic_stock_model as dsm
import numpy as np

# import data
res_census_inflow = pd.read_excel('./GFA_hypothesis/Census_Units.xlsx', sheet_name='Moura_et_al_2015')

# filter data
my_inflow = res_census_inflow['Total_area_Mm2']
years = np.array(res_census_inflow['Year'])


def generate_lt(type, par1, par2):
    ''' Normal: par1  = mean, par2 = std. dev
        Weibull: par1 = shape, par2 = scale'''

    # ---- Building lifespan distributions ----
    # BldgLife_mean_res = 80  # years
    # BldgLife_StdDev_res = 0.2 *  np.array([BldgLife_mean_res] * len(years))
    # BldgLife_mean_com = 70  # years
    # BldgLife_StdDev_com = 0.2 *  np.array([BldgLife_mean_com] * len(years))
    # BldgLife_mean_pub = 90  # years
    # BldgLife_StdDev_pub = 0.2 * np.array([BldgLife_mean_com] * len(years))
    if type=='Normal':
        # Normal
        lt = {'Type': type, 'Mean': np.array([par1] * len(years)), 'StdDev': np.array([par2])}
    elif type=='Weibull':
        # Weibull
        # lt_res = {'Type': 'Weibull', 'Shape': np.array([4.16343417]), 'Scale': np.array([85.18683893])}     # deetman_2018_res_distr_weibull
        # lt_res = {'Type': 'Weibull', 'Shape': np.array([5.5]), 'Scale': np.array([85.8])}
        # lt_com = {'Type': 'Weibull', 'Shape': np.array([4.8]), 'Scale': np.array([75.1])}
        # lt_res = {'Type': type, 'Shape': np.array([5]), 'Scale': np.array([130])}
        # lt_com = {'Type': type, 'Shape': np.array([3]), 'Scale': np.array([100])}
        # lt_pub = {'Type': type, 'Shape': np.array([6.1]), 'Scale': np.array([95.6])}
        lt = {'Type': type, 'Shape': np.array([par1]), 'Scale': np.array([par2])}
    return lt

lt_res = generate_lt('Weibull',par1=8.1, par2=100)
# lt_res = generate_lt('Normal',par1=35, par2=10)

DSM_Inflow = dsm.DynamicStockModel(t=years, i=my_inflow, lt=lt_res)
CheckStr = DSM_Inflow.dimension_check()
print(CheckStr)

Stock_by_cohort = DSM_Inflow.compute_s_c_inflow_driven()
O_C = DSM_Inflow.compute_o_c_from_s_c()
S = DSM_Inflow.compute_stock_total()
DS = DSM_Inflow.compute_stock_change()
#
# Bal = DSM_Inflow
# print(Bal.shape) # dimensions of balance are: time step x process x chemical element
# print(np.abs(Bal).sum(axis = 0)) # reports the sum of all absolute balancing errors by process.

# total stock
plt2, = plt.plot(DSM_Inflow.t, DSM_Inflow.s)
plt.legend([plt2], ['Stock'], loc=2)
plt.xlabel('Year')
plt.ylabel('Floor Area (Mm2)')
plt.title('Floor Area Stock')
plt.show();

# plot inflows and outflows
plt1, = plt.plot(DSM_Inflow.t, DSM_Inflow.i)
plt3, = plt.plot(DSM_Inflow.t, np.sum(DSM_Inflow.o_c, axis=1))
plt.xlabel('Year')
plt.ylabel('Floor Area per year (Mm2)')
plt.title('Floor Area flows')
plt.legend([plt1, plt3], ['Inflow', 'Outflow'], loc=2)
plt.show();

# Stock by age-cohort
plt.imshow(DSM_Inflow.s_c[:, 1:], interpolation='nearest')  # exclude the first column to have the color scale work.
plt.xlabel('age-cohort')
plt.ylabel('year')
plt.title('Stock by age-cohort')
plt.show();

# look at survival functions

my_sf = DSM_Inflow.compute_sf()
plt1, = plt.plot(my_sf[:,0])
plt.xlabel('years')
plt.title('Survival function')
plt.show();

my_inflow.sum()