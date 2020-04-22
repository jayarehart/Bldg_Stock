# Material Demands for the US Building Stock
#
# Calculate the future building stock-wide material demand for buildings
#   based upon the material intensity of buidings today. Validate this bottom-up approach
#   with top-down economic data.

# import libraries
import pandas as pd
import matplotlib.pyplot as plt

# load in data from other scripts
structure_data_base_year = pd.read_csv('./InputData/HAZUS_weight.csv')
FA_dsm_SSP1 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP1')
materials_intensity = pd.read_excel('./InputData/Material_data.xlsx', sheet_name='SSP1_density')
materials_intensity_df = materials_intensity.set_index('Structure_Type', drop=True)
materials_intensity_df = materials_intensity_df.transpose()
materials_intensity_df = materials_intensity_df.drop(index='Source')


# get total floor area stock and flows
total_area = FA_dsm_SSP1[['time','stock_total','inflow_total','outflow_total']]
total_area = total_area.set_index('time', drop=True)

# Calculate floor area inflow/outflow by structural system (unit = million m2)
sio_area_by_ss = pd.DataFrame()
sio_area_by_ss['LF_wood_s'] = total_area['stock_total'] * structure_data_base_year.LF_wood[0]
sio_area_by_ss['LF_wood_i'] = total_area['inflow_total'] * structure_data_base_year.LF_wood[0]
sio_area_by_ss['LF_wood_o'] = total_area['outflow_total'] * structure_data_base_year.LF_wood[0]
sio_area_by_ss['Mass_Timber_s'] = total_area['stock_total'] * structure_data_base_year.Mass_Timber[0]
sio_area_by_ss['Mass_Timber_i'] = total_area['inflow_total'] * structure_data_base_year.Mass_Timber[0]
sio_area_by_ss['Mass_Timber_o'] = total_area['outflow_total'] * structure_data_base_year.Mass_Timber[0]
sio_area_by_ss['Steel_s'] = total_area['stock_total'] * structure_data_base_year.Steel[0]
sio_area_by_ss['Steel_i'] = total_area['inflow_total'] * structure_data_base_year.Steel[0]
sio_area_by_ss['Steel_o'] = total_area['outflow_total'] * structure_data_base_year.Steel[0]
sio_area_by_ss['RC_s'] = total_area['stock_total'] * structure_data_base_year.RC[0]
sio_area_by_ss['RC_i'] = total_area['inflow_total'] * structure_data_base_year.RC[0]
sio_area_by_ss['RC_o'] = total_area['outflow_total'] * structure_data_base_year.RC[0]
sio_area_by_ss['RM_s'] = total_area['stock_total'] * structure_data_base_year.RM[0]
sio_area_by_ss['RM_i'] = total_area['inflow_total'] * structure_data_base_year.RM[0]
sio_area_by_ss['RM_o'] = total_area['outflow_total'] * structure_data_base_year.RM[0]
sio_area_by_ss['URM_s'] = total_area['stock_total'] * structure_data_base_year.URM[0]
sio_area_by_ss['URM_i'] = total_area['inflow_total'] * structure_data_base_year.URM[0]
sio_area_by_ss['URM_o'] = total_area['outflow_total'] * structure_data_base_year.URM[0]
sio_area_by_ss['MH_s'] = total_area['stock_total'] * structure_data_base_year.MH[0]
sio_area_by_ss['MH_i'] = total_area['inflow_total'] * structure_data_base_year.MH[0]
sio_area_by_ss['MH_o'] = total_area['outflow_total'] * structure_data_base_year.MH[0]

# Calculate material demand by structural system

# Inflow of materials in (unit = Megaton, Mt)
inflow_mat = pd.DataFrame()
inflow_mat['LF_wood_steel'] = sio_area_by_ss['LF_wood_i'] * materials_intensity_df['LF_wood']['Steel_kgm2_mean'] * 10e6 / 10e9
inflow_mat['LF_wood_conc'] = sio_area_by_ss['LF_wood_i'] * materials_intensity_df['LF_wood']['Concrete_kgm2_mean'] * 10e6 / 10e9
inflow_mat['LF_wood_engwood'] = sio_area_by_ss['LF_wood_i'] * materials_intensity_df['LF_wood']['Eng_wood_kgm2_mean'] * 10e6 / 10e9
inflow_mat['LF_wood_dimlum'] = sio_area_by_ss['LF_wood_i'] * materials_intensity_df['LF_wood']['Dim_lumber_kgm2_mean'] * 10e6 / 10e9
inflow_mat['LF_wood_masonry'] = sio_area_by_ss['LF_wood_i'] * materials_intensity_df['LF_wood']['Masonry_kgm2_mean'] * 10e6 / 10e9

inflow_mat['Mass_Timber_steel'] = sio_area_by_ss['Mass_Timber_i'] * materials_intensity_df['Mass_Timber']['Steel_kgm2_mean'] * 10e6 / 10e9
inflow_mat['Mass_Timber_conc'] = sio_area_by_ss['Mass_Timber_i'] * materials_intensity_df['Mass_Timber']['Concrete_kgm2_mean'] * 10e6 / 10e9
inflow_mat['Mass_Timber_engwood'] = sio_area_by_ss['Mass_Timber_i'] * materials_intensity_df['Mass_Timber']['Eng_wood_kgm2_mean'] * 10e6 / 10e9
inflow_mat['Mass_Timber_dimlum'] = sio_area_by_ss['Mass_Timber_i'] * materials_intensity_df['Mass_Timber']['Dim_lumber_kgm2_mean'] * 10e6 / 10e9
inflow_mat['Mass_Timber_masonry'] = sio_area_by_ss['Mass_Timber_i'] * materials_intensity_df['Mass_Timber']['Masonry_kgm2_mean'] * 10e6 / 10e9

inflow_mat['Steel_steel'] = sio_area_by_ss['Steel_i'] * materials_intensity_df['Steel']['Steel_kgm2_mean'] * 10e6 / 10e9
inflow_mat['Steel_conc'] = sio_area_by_ss['Steel_i'] * materials_intensity_df['Steel']['Concrete_kgm2_mean'] * 10e6 / 10e9
inflow_mat['Steel_engwood'] = sio_area_by_ss['Steel_i'] * materials_intensity_df['Steel']['Eng_wood_kgm2_mean'] * 10e6 / 10e9
inflow_mat['Steel_dimlum'] = sio_area_by_ss['Steel_i'] * materials_intensity_df['Steel']['Dim_lumber_kgm2_mean'] * 10e6 / 10e9
inflow_mat['Steel_masonry'] = sio_area_by_ss['Steel_i'] * materials_intensity_df['Steel']['Masonry_kgm2_mean'] * 10e6 / 10e9

inflow_mat['RC_steel'] = sio_area_by_ss['RC_i'] * materials_intensity_df['RC']['Steel_kgm2_mean'] * 10e6 / 10e9
inflow_mat['RC_conc'] = sio_area_by_ss['RC_i'] * materials_intensity_df['RC']['Concrete_kgm2_mean'] * 10e6 / 10e9
inflow_mat['RC_engwood'] = sio_area_by_ss['RC_i'] * materials_intensity_df['RC']['Eng_wood_kgm2_mean'] * 10e6 / 10e9
inflow_mat['RC_dimlum'] = sio_area_by_ss['RC_i'] * materials_intensity_df['RC']['Dim_lumber_kgm2_mean'] * 10e6 / 10e9
inflow_mat['RC_masonry'] = sio_area_by_ss['RC_i'] * materials_intensity_df['RC']['Masonry_kgm2_mean'] * 10e6 / 10e9

inflow_mat['RM_steel'] = sio_area_by_ss['RM_i'] * materials_intensity_df['RM']['Steel_kgm2_mean'] * 10e6 / 10e9
inflow_mat['RM_conc'] = sio_area_by_ss['RM_i'] * materials_intensity_df['RM']['Concrete_kgm2_mean'] * 10e6 / 10e9
inflow_mat['RM_engwood'] = sio_area_by_ss['RM_i'] * materials_intensity_df['RM']['Eng_wood_kgm2_mean'] * 10e6 / 10e9
inflow_mat['RM_dimlum'] = sio_area_by_ss['RM_i'] * materials_intensity_df['RM']['Dim_lumber_kgm2_mean'] * 10e6 / 10e9
inflow_mat['RM_masonry'] = sio_area_by_ss['RM_i'] * materials_intensity_df['RM']['Masonry_kgm2_mean'] * 10e6 / 10e9

inflow_mat['URM_steel'] = sio_area_by_ss['URM_i'] * materials_intensity_df['URM']['Steel_kgm2_mean'] * 10e6 / 10e9
inflow_mat['URM_conc'] = sio_area_by_ss['URM_i'] * materials_intensity_df['URM']['Concrete_kgm2_mean'] * 10e6 / 10e9
inflow_mat['URM_engwood'] = sio_area_by_ss['URM_i'] * materials_intensity_df['URM']['Eng_wood_kgm2_mean'] * 10e6 / 10e9
inflow_mat['URM_dimlum'] = sio_area_by_ss['URM_i'] * materials_intensity_df['URM']['Dim_lumber_kgm2_mean'] * 10e6 / 10e9
inflow_mat['URM_masonry'] = sio_area_by_ss['URM_i'] * materials_intensity_df['URM']['Masonry_kgm2_mean'] * 10e6 / 10e9

inflow_mat['MH_steel'] = sio_area_by_ss['MH_i'] * materials_intensity_df['MH']['Steel_kgm2_mean'] * 10e6 / 10e9
inflow_mat['MH_conc'] = sio_area_by_ss['MH_i'] * materials_intensity_df['MH']['Concrete_kgm2_mean'] * 10e6 / 10e9
inflow_mat['MH_engwood'] = sio_area_by_ss['MH_i'] * materials_intensity_df['MH']['Eng_wood_kgm2_mean'] * 10e6 / 10e9
inflow_mat['MH_dimlum'] = sio_area_by_ss['MH_i'] * materials_intensity_df['MH']['Dim_lumber_kgm2_mean'] * 10e6 / 10e9
inflow_mat['MH_masonry'] = sio_area_by_ss['MH_i'] * materials_intensity_df['MH']['Masonry_kgm2_mean'] * 10e6 / 10e9

# Total inflows of each material (by structure type), unit = Mt
steel_tot_i = inflow_mat.filter(like='steel', axis=1)
conc_tot_i = inflow_mat.filter(like='conc', axis=1)
engwood_tot_i = inflow_mat.filter(like='engwood', axis=1)
dimlum_tot_i = inflow_mat.filter(like='dimlum', axis=1)
masonry_tot_i = inflow_mat.filter(like='masonry', axis=1)
# adding in a sum row for each year
steel_tot_i['Sum'] = steel_tot_i.sum(axis=1)
conc_tot_i['Sum'] = conc_tot_i.sum(axis=1)
engwood_tot_i['Sum'] = engwood_tot_i.sum(axis=1)
dimlum_tot_i['Sum'] = dimlum_tot_i.sum(axis=1)
masonry_tot_i['Sum'] = masonry_tot_i.sum(axis=1)

# plot total material demand each year (Mt)
plt1, = plt.plot(steel_tot_i.index, steel_tot_i.Sum)
plt2, = plt.plot(conc_tot_i.index, conc_tot_i.Sum)
plt3, = plt.plot(engwood_tot_i.index, engwood_tot_i.Sum)
plt4, = plt.plot(dimlum_tot_i.index, dimlum_tot_i.Sum)
plt5, = plt.plot(masonry_tot_i.index, masonry_tot_i.Sum)
plt6, = plt.plot([2020, 2020], [0, 200], color='k', LineStyle='--')
plt.legend([plt1, plt2, plt3, plt4, plt5],
           ['Steel', 'Concrete', 'Eng. Wood', 'Dim. Lumber', 'Masonry'],
           loc='best')
plt.xlabel('Year')
plt.ylabel('Mt / year')
plt.ylim(top=3000)
plt.xlim(left=2000)
plt.title('Annual material consumption by US building structural systems')
plt.show();

# print the material demand for a particular year
year = 2015
print('Total steel demand in ', str(year),  ' =   ', steel_tot_i['Sum'][year], ' Mt')
print('Total concrete demand in ', str(year),  ' =   ', conc_tot_i['Sum'][year], ' Mt')
print('Total engineered wood demand in ', str(year),  ' =   ', engwood_tot_i['Sum'][year], ' Mt')
print('Total dimensioned lumber demand in ', str(year),  ' =   ', dimlum_tot_i['Sum'][year], ' Mt')
print('Total masonry demand in ', str(year),  ' =   ', masonry_tot_i['Sum'][year], ' Mt')

