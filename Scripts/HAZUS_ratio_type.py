# HAZUS Data processing

# Input = HAZUS results that were extracted
# Ouput = dataframe of weights for different structural systems

import pandas as pd
import glob

data_FA_Arehart = pd.read_excel('./InputData/Floor_Area_Arehart2020.xlsx', sheet_name='by_state')
data_FA_Arehart = data_FA_Arehart.set_index('state', drop=False)
directoryPath='./InputData/HAZUS_Extract/'
summary_df_perc = pd.DataFrame()
summary_df_sqm = pd.DataFrame()
total_area = 0

file_name = './InputData/HAZUS_Extract/newjersey_type.txt'
# files = glob.glob(directoryPath+'*occupancy.txt')
files = glob.glob(directoryPath+'*type.txt')
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
# W1	Wood stud-wall frame with plywood/gypsum board sheathing.
# W2	Wood frame, heavy members (with area > 5000 sq. ft.)
# S1L	Steel moment frame low-rise
# S2L	Steel braced frame low-rise
# S3	Steel light frame
# S4L	Steel frame with cast-in-place concrete shear walls low-rise
# C1L	Ductile reinforced concrete moment frame with or without infill low-rise
# C2L	Reinforced concrete shear walls low-rise
# C3L	Nonductile reinforced concrete frame with masonry infill walls low-rise
# PC1	Precast concrete tilt-up walls
# PC2	Precast concrete frames with concrete shear walls
# RM1L	Reinforced masonry bearing walls with wood or metal deck diaphragms low-rise
# RM2L	Reinforced masonry bearing walls with concrete diaphragms low-rise
# URML	Unreinforced masonry low-rise
# MH	Mobile homes
my_weights = (types_df
              .assign(LF_wood = type_weighted['W1F'])
              .assign(Mass_Timber = type_weighted['W2F'])
              .assign(Steel = type_weighted['S1LF'] + type_weighted['S2LF'] + type_weighted['S3F'] + type_weighted['S4LF'] + type_weighted['S4HF'] + type_weighted['S5LF'])
              .assign(RC = type_weighted['C1LF'] + type_weighted['C2LF'] + type_weighted['C3LF'] + type_weighted['PC1F'] + type_weighted['PC2LF'])
              .assign(RM = type_weighted['RM1LF'] + type_weighted['RM1MF'] + type_weighted['RM2LF'])
              .assign(URM = type_weighted['URMLF'])
              .assign(MH = type_weighted['MHF'])
              .drop(columns=['W1F', 'W2F', 'S1LF', 'S2LF', 'S3F', 'S4LF', 'S4HF', 'S5LF',
                             'C1LF', 'C2LF', 'C3LF', 'PC1F', 'PC2LF', 'RM1LF', 'RM1MF', 'RM2LF', 'URMLF', 'MHF'])
              )
print('Distribution of buildings into structural systems     ', my_weights)
print('Sum of all weights =   ', my_weights.sum(axis=1))

# save csv file
my_weights.to_csv('./InputData/HAZUS_weight.csv')

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

my_weights_type = (types_df
              .assign(RES=type_weighted['RES1F'] + type_weighted['RES2F'] + type_weighted['RES3F'] +
                            type_weighted['RES4F'] + type_weighted['RES5F'] + type_weighted['RES6F'])
              .assign(COM=type_weighted['COM1F'] + type_weighted['COM2F'] + type_weighted['COM3F'] +
                          type_weighted['COM4F'] + type_weighted['COM5F'] + type_weighted['COM6F'] +
                          type_weighted['COM7F'] + type_weighted['COM8F'] + type_weighted['COM9F'] + type_weighted['COM10F'])
              .assign(PUB=type_weighted['REL1F'] + type_weighted['GOV1F'] + type_weighted['GOV2F'] +
                          type_weighted['EDU1F'] + type_weighted['EDU2F'])

              .drop(columns=['RES1F', 'RES2F', 'RES3F', 'RES4F', 'RES5F', 'RES6F', 'COM1F', 'COM2F', 'COM3F',
                             'COM4F', 'COM5F','COM6F', 'COM7F', 'COM8F', 'COM9F', 'COM10F', 'IND1F', 'IND2F',
                             'IND3F', 'IND4F', 'IND5F', 'IND6F', 'AGR1F', 'REL1F', 'GOV1F', 'GOV2F' 'EDU1F', 'EDU2F'])
              )
print('Distribution of buildings into res/com/pub     ', my_weights_type)
print('Sum of all weights =   ', my_weights_type.sum(axis=1))

# save csv file
my_weights_type.to_csv('./InputData/HAZUS_type_weights.csv')


