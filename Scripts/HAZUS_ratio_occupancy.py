# HAZUS Data processing

# Input = HAZUS results that were extracted
# Ouput = dataframe of weights for different occupancy types

import pandas as pd
import glob

data_FA_Arehart = pd.read_excel('./InputData/Floor_Area_Arehart2020.xlsx', sheet_name='by_state')
data_FA_Arehart = data_FA_Arehart.set_index('state', drop=False)
directoryPath='./InputData/HAZUS_Extract/'
summary_df_perc = pd.DataFrame()
summary_df_sqm = pd.DataFrame()
total_area = 0

# --------------------------------------------------
# Manipulate and output data by residential, commercial, public, and industrial

# file_name = './InputData/HAZUS_Extract/newjersey_type.txt'
files = glob.glob(directoryPath+'*occupancy.txt')
# files = glob.glob(directoryPath+'*type.txt')
for file_name in files:
    state_str = str.split(str.split(str.split(file_name, '/')[3], '.')[0], '_')[0]
    x = pd.read_csv(file_name, low_memory=False)
    x = x.set_index('Tract', drop=True)
    x_sum = x.sum(axis=0) * 1000 * 0.092903 / 10e6     # sum and convert to sqm
    x_tot_sm = x_sum.TotalSqft
    total_area = total_area + x_tot_sm          # sum of total area
    y_line = x_sum / x_tot_sm                   # percent weights of each
    y_df = pd.DataFrame({state_str: y_line})
    x_df = pd.DataFrame({state_str: x_sum})
    check = y_line.sum(axis=0) - 1              # check sum to be 100%
    print(state_str, '  ', check)
    summary_df_perc = pd.concat([summary_df_perc,y_df],axis=1)
    summary_df_sqm = pd.concat([summary_df_sqm, x_df],axis=1)

print('total area is ', total_area, ' million square meters')

summary_df_sqm = summary_df_sqm.transpose()
summary_df_sqm = summary_df_sqm.reset_index()
summary_df_sqm = summary_df_sqm.rename(columns={'index':'state', 'TotalSqft':'Sqm_millions_HAZUS'})
summary_df_sqm = summary_df_sqm.sort_values('state', ascending=True)
summary_df_sqm = summary_df_sqm.reset_index(drop=True)


summary_df_perc = summary_df_perc.transpose()
summary_df_perc = summary_df_perc.reset_index()
summary_df_perc = summary_df_perc.rename(columns={'index':'state', 'TotalSqft':'Sum_HAZUS'})
summary_df_perc = summary_df_perc.sort_values('state', ascending=True)
summary_df_perc = summary_df_perc.reset_index(drop=True)
summary_df_perc = summary_df_perc.set_index(keys='state', drop=True)

# df = pd.merge(summary_df_perc, data_FA_Arehart, on='state')
# df = df.sort_values('state', ascending=True)

# Structure type for the US weighted by the floor area in each state.
weight = data_FA_Arehart['Weight'].reindex_like(data_FA_Arehart)
dist_weighted = summary_df_perc.multiply(weight, axis=0)
dist_weighted = dist_weighted.drop(columns=['Sum_HAZUS'], axis=1)
type_weighted = dist_weighted.sum(axis=0)
# check
print('Sum of all structure type weights is: ', sum(dist_weighted.sum(axis=0)))
# drop structure types that have zero
type_weighted = type_weighted[(type_weighted != 0)]
types_df = pd.DataFrame({'weight':type_weighted}).transpose()

# Convert from HAZUS Definitions to my definitions

# HAZUS Appendix B. Classification Systems
# RES1 - Single Family Dwelling
# RES2 - Mobile Home
# RES3 - Multifamily Dwelling A=duplex, B=3-4 units, C=5-9 units, D=10-19 unnits, E=20-49 units, F=50+ units
# RES4 - Temporary lodging
# RES5 - Industrial Dormitory
# RES6 - Nursing home
# COM1 - Retail Trade
# COM2 - Wholesale Trade
# COM3 - Personal and Repair Services
# COM4 - Professional/Technical Services
# COM5 - Banks
# COM6 - Hospital
# COM7 - Medical Office/Clinic
# COM8 - Entertainment & Recreation
# COM9 - Theaters
# COM10 - Parking
# IND1 - Heavy
# IND2 - Light
# IND3 - Food/Drugs/Chemicals
# IND4 - Metals/Minerals Processing
# IND5 - High Technology
# IND6 - Construction
# AGR1 - Agriculture
# REL1 - Church/Non-Profit
# GOV1 - General services
# GOV2 - Emergency Response
# EDU1 - Grade Schools
# EDU2 - Colleges/Universities

my_weights_occupancy = (types_df
              .assign(RES=type_weighted['RES1F'] + type_weighted['RES2F'] + type_weighted['RES3AF'] +
                          type_weighted['RES3BF'] + type_weighted['RES3CF'] + type_weighted['RES3DF'] +
                          type_weighted['RES3EF'] + type_weighted['RES3FF'] + type_weighted['RES4F'] +
                          type_weighted['RES5F'] + type_weighted['RES6F'])
              .assign(COM=type_weighted['COM1F'] + type_weighted['COM2F'] + type_weighted['COM3F'] +
                          type_weighted['COM4F'] + type_weighted['COM5F'] + type_weighted['COM6F'] +
                          type_weighted['COM7F'] + type_weighted['COM8F'] + type_weighted['COM9F'])
              .assign(PUB=type_weighted['REL1F'] + type_weighted['GOV1F'] + type_weighted['GOV2F'] +
                          type_weighted['EDU1F'] + type_weighted['EDU2F'])

              .drop(columns=['RES1F', 'RES2F', 'RES3AF', 'RES3BF', 'RES3CF', 'RES3DF', 'RES3EF', 'RES3FF', 'RES4F',
                             'RES5F', 'RES6F', 'COM1F', 'COM2F', 'COM3F',
                             'COM4F', 'COM5F','COM6F', 'COM7F', 'COM8F', 'COM9F', 'IND1F', 'IND2F',
                             'IND3F', 'IND4F', 'IND5F', 'IND6F', 'AGR1F', 'REL1F', 'GOV1F', 'GOV2F', 'EDU1F', 'EDU2F'])
              )
print('Distribution of buildings into res/com/pub     ', my_weights_occupancy)
print('Sum of all weights =   ', my_weights_occupancy.sum(axis=1))

# save csv file
my_weights_occupancy.to_csv('./InputData/HAZUS_occupancy_weights.csv')