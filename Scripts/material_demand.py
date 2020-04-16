# Material Demands for the US Building Stock
#
# Calculate the future building stock-wide material demand for buildings
#   based upon the material intensity of buidings today. Validate this bottom-up approach
#   with top-down economic data.

# import libraries
import pandas as pd
import numpy as np

# load in data from other scripts
structure_data_base_year = pd.read_csv('./InputData/HAZUS_weight.csv')
FA_dsm_SSP1 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP1')
materials_intensity = pd.read_excel('./InputData/Material_data.xlsx', sheet_name='SSP1_density')

# get total floor area stock and flows
total_area = FA_dsm_SSP1[['time','stock_total','inflow_total','outflow_total']]
total_area = total_area.set_index('time', drop=True)

# break into structural systems
# total_area_by_ss =

# Calculate floor area inflow/outflow by structural system.



# Calculate annual material demands in the future:
FA_dsm_SSP1
