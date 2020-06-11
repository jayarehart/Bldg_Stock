# Material Demands for the US Building Stock
#
# Calculate the future building stock-wide material demand for buildings
#   based upon the material intensity of buidings today. Validate this bottom-up approach
#   with top-down economic data.

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from odym import dynamic_stock_model as dsm
import numpy as np

# load in data from other scripts and excels
structure_data_historical = pd.read_csv('./InputData/HAZUS_weight.csv')

FA_dsm_SSP1 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP1')
FA_dsm_SSP2 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP2')
FA_dsm_SSP3 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP3')
FA_dsm_SSP4 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP4')
FA_dsm_SSP5 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP5')

FA_sc_SSP1 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP1_sc')
FA_sc_SSP2 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP2_sc')
FA_sc_SSP3 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP3_sc')
FA_sc_SSP4 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP4_sc')
FA_sc_SSP5 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP5_sc')

materials_intensity = pd.read_excel('./InputData/Material_data.xlsx', sheet_name='SSP1_density')
materials_intensity_df = materials_intensity.set_index('Structure_Type', drop=True)
materials_intensity_df = materials_intensity_df.transpose()
materials_intensity_df = materials_intensity_df.drop(index='Source')

scenario_df = pd.read_excel('./InputData/Material_data.xlsx', sheet_name='Adoption_clean')
scenario_df = scenario_df.set_index('Scenario')



# get total floor area stock and flows for the specific SSP loaded
# total_area = FA_dsm_SSP1[['time','stock_total','inflow_total','outflow_total']]
# total_area = total_area.set_index('time', drop=True)

# Clean dataframes


# set years series
years_future = FA_dsm_SSP1['time'].loc[197:]
years_all = FA_dsm_SSP1.index.to_series()

# compute a lifetime distribution
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
        lt = {'Type': type, 'Mean': np.array([par1] * len(years_all)), 'StdDev': np.array([par2])}
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

lt_existing = generate_lt('Weibull',par1=5, par2=75)        # lifetime distribution for existing buildings (all)
lt_future = generate_lt('Weibull', par1=5, par2=100)
# lt_res = generate_lt('Normal',par1=35, par2=10)

# area of outflow of each structural system type for already built buildings (before 2017). No new construction is considered in this analysis
def determine_outflow_existing_bldgs(FA_sc_SSP, plot=True, plot_title=''):
    '''Input a floor area stock-cohort matrix for each SSP and compute the outflow for each structural system that are already built.
     Key assumption is that construction techniques are the same each year. '''

    # compute an outflow for the existing stock using a compute evolution from initial stock method
    def determine_outflow_by_ss(lt=lt_existing, FA_sc_df=FA_sc_SSP, switch_year=196, frac_stock=1.0):
        '''Compute the outflow of the existing building stock with no additional inflow.
        A switch year of 196 represents 2016.
        frac_stock is the ratio of the exisitng building stock that is a particular structural system.
        For frac_stock = 1.0, all of the stock is considered to be the same. '''
        DSM_existing_stock = dsm.DynamicStockModel(t=years_all, lt=lt_existing)
        S_C = DSM_existing_stock.compute_evolution_initialstock(
            InitialStock=FA_sc_SSP.loc[(switch_year - 1), 0:(switch_year - 1)] * frac_stock, SwitchTime=switch_year)
        # compute outflow
        O_C = DSM_existing_stock.o_c[1::, :] = -1 * np.diff(DSM_existing_stock.s_c, n=1, axis=0)
        O_C = DSM_existing_stock.o_c[np.diag_indices(len(DSM_existing_stock.t))] = 0 - np.diag(
            DSM_existing_stock.s_c)  # allow for outflow in year 0 already
        O = DSM_existing_stock.compute_outflow_total()
        outflow_df = pd.DataFrame({'time': DSM_existing_stock.t, 'outflow': O, 'stock': DSM_existing_stock.s})
        return outflow_df

    existing_outflow_LF_wood = determine_outflow_by_ss(lt=lt_existing,FA_sc_df=FA_sc_SSP,switch_year=196, frac_stock=structure_data_historical.LF_wood[0])
    existing_outflow_Mass_Timber = determine_outflow_by_ss(lt=lt_existing,FA_sc_df=FA_sc_SSP,switch_year=196, frac_stock=structure_data_historical.Mass_Timber[0])
    existing_outflow_Steel = determine_outflow_by_ss(lt=lt_existing,FA_sc_df=FA_sc_SSP,switch_year=196, frac_stock=structure_data_historical.Steel[0])
    existing_outflow_RC = determine_outflow_by_ss(lt=lt_existing,FA_sc_df=FA_sc_SSP,switch_year=196, frac_stock=structure_data_historical.RC[0])
    existing_outflow_RM = determine_outflow_by_ss(lt=lt_existing,FA_sc_df=FA_sc_SSP,switch_year=196, frac_stock=structure_data_historical.RM[0])
    existing_outflow_URM = determine_outflow_by_ss(lt=lt_existing,FA_sc_df=FA_sc_SSP,switch_year=196, frac_stock=structure_data_historical.URM[0])
    existing_outflow_MH = determine_outflow_by_ss(lt=lt_existing,FA_sc_df=FA_sc_SSP,switch_year=196, frac_stock=structure_data_historical.MH[0])

    existing_outflow_all = pd.DataFrame({'outflow_LF_wood': existing_outflow_LF_wood.outflow,
                                         'stock_LF_wood': existing_outflow_LF_wood.stock,
                                         'outflow_Mass_Timber': existing_outflow_Mass_Timber.outflow,
                                         'stock_Mass_Timber': existing_outflow_Mass_Timber.stock,
                                         'outflow_Steel': existing_outflow_Steel.outflow,
                                         'stock_Steel': existing_outflow_Steel.stock,
                                         'outflow_RC': existing_outflow_RC.outflow,
                                         'stock_RC': existing_outflow_RC.stock,
                                         'outflow_RM': existing_outflow_RM.outflow,
                                         'stock_RM': existing_outflow_RM.stock,
                                         'outflow_URM': existing_outflow_URM.outflow,
                                         'stock_URM': existing_outflow_URM.stock,
                                         'outflow_MH': existing_outflow_MH.outflow,
                                         'stock_MH': existing_outflow_MH.stock,
                                        })
    if plot == True:
        # plot the
        existing_outflow_all.iloc[197:].plot.line()
        plt.ylabel('Floor Area (Mm2)')
        plt.title(plot_title + ': Outflow of Buildings Constructed before 2017')
        plt.show()
    return existing_outflow_all

o_existing_SSP1 = determine_outflow_existing_bldgs(FA_sc_SSP=FA_sc_SSP1, plot=True, plot_title='SSP1')
o_existing_SSP2 = determine_outflow_existing_bldgs(FA_sc_SSP=FA_sc_SSP2, plot=True, plot_title='SSP2')
o_existing_SSP3 = determine_outflow_existing_bldgs(FA_sc_SSP=FA_sc_SSP3, plot=True, plot_title='SSP3')
o_existing_SSP4 = determine_outflow_existing_bldgs(FA_sc_SSP=FA_sc_SSP4, plot=True, plot_title='SSP4')
o_existing_SSP5 = determine_outflow_existing_bldgs(FA_sc_SSP=FA_sc_SSP5, plot=True, plot_title='SSP5')


def determine_inflow_outflow_new_bldg(scenario, FA_dsm_SSP=FA_dsm_SSP1, plot=True, plot_title='SSP1 '):
    # Select a scenario
    # scenario = 'S_0'    # new construction is same as exiting building stock
    # scenario = 'S_timber_high'      # High timber adoption

    # clean df
    FA_dsm_SSP = FA_dsm_SSP.set_index('time')
    structure_data_scenario = pd.DataFrame(
        {'LF_wood': scenario_df.LF_wood[scenario],
        'Mass_Timber': scenario_df.Mass_Timber[scenario],
        'Steel': scenario_df.Steel[scenario],
        'RC': scenario_df.RC[scenario],
        'RM': scenario_df.RM[scenario],
        'URM': scenario_df.URM[scenario],
        'MH': scenario_df.MH[scenario]},
    index=[0])

    # separate inflow by structural system ratio for each scenario
    inflow_SSP_all = pd.DataFrame(
        {'inflow_total' : FA_dsm_SSP.loc[2017:2100, 'inflow_total'],
         'inflow_LF_wood' : FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * structure_data_scenario.LF_wood[0],
         'inflow_Mass_Timber': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * structure_data_scenario.Mass_Timber[0],
         'inflow_Steel': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * structure_data_scenario.Steel[0],
         'inflow_RC': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * structure_data_scenario.RC[0],
         'inflow_RM': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * structure_data_scenario.RM[0],
         'inflow_URM': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * structure_data_scenario.URM[0],
         'inflow_MH': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * structure_data_scenario.MH[0],
        }
    )

    def compute_inflow_driven_model_ea_ss(structural_system):
        # compute a inflow driven model for each structural system
        DSM_Inflow_x = dsm.DynamicStockModel(t=years_future, i=inflow_SSP_all['inflow_' + structural_system], lt=lt_future)
        # CheckStr = DSM_Inflow.dimension_check()
        # print(CheckStr)

        S_C = DSM_Inflow_x.compute_s_c_inflow_driven()
        O_C = DSM_Inflow_x.compute_o_c_from_s_c()
        S = DSM_Inflow_x.compute_stock_total()
        O = DSM_Inflow_x.compute_outflow_total()
        DSM_Inflow_x.o = pd.Series(DSM_Inflow_x.o, index=DSM_Inflow_x.t)
        return DSM_Inflow_x

    DSM_Inflow_LF_wood = compute_inflow_driven_model_ea_ss(structural_system='LF_wood')
    DSM_Inflow_Mass_Timber = compute_inflow_driven_model_ea_ss('Mass_Timber')
    DSM_Inflow_Steel = compute_inflow_driven_model_ea_ss('Steel')
    DSM_Inflow_RC = compute_inflow_driven_model_ea_ss('RC')
    DSM_Inflow_RM = compute_inflow_driven_model_ea_ss('RM')
    DSM_Inflow_URM = compute_inflow_driven_model_ea_ss('URM')
    DSM_Inflow_MH = compute_inflow_driven_model_ea_ss('MH')

    # summary dataframe of all DSM stocks, inflows, outflows
    DSM_Future_all = pd.DataFrame({
        'LF_wood_inflow' : DSM_Inflow_LF_wood.i,
        'LF_wood_outflow': DSM_Inflow_LF_wood.o,
        'LF_wood_stock': DSM_Inflow_LF_wood.s,
        'Mass_Timber_inflow': DSM_Inflow_Mass_Timber.i,
        'Mass_Timber_outflow': DSM_Inflow_Mass_Timber.o,
        'Mass_Timber_stock': DSM_Inflow_Mass_Timber.s,
        'Steel_inflow': DSM_Inflow_Steel.i,
        'Steel_outflow': DSM_Inflow_Steel.o,
        'Steel_stock': DSM_Inflow_Steel.s,
        'RC_inflow': DSM_Inflow_RC.i,
        'RC_outflow': DSM_Inflow_RC.o,
        'RC_stock': DSM_Inflow_RC.s,
        'RM_inflow': DSM_Inflow_RM.i,
        'RM_outflow': DSM_Inflow_RM.o,
        'RM_stock': DSM_Inflow_RM.s,
        'URM_inflow': DSM_Inflow_URM.i,
        'URM_outflow': DSM_Inflow_URM.o,
        'URM_stock': DSM_Inflow_URM.s,
        'MH_inflow': DSM_Inflow_MH.i,
        'MH_outflow': DSM_Inflow_MH.o,
        'MH_stock': DSM_Inflow_MH.s,
    })

    if plot==True:
        DSM_Future_Inflow = DSM_Future_all.loc[:, DSM_Future_all.columns.str.contains('inflow')]
        DSM_Future_Outflow = DSM_Future_all.loc[:, DSM_Future_all.columns.str.contains('outflow')]
        DSM_Future_Stock = DSM_Future_all.loc[:, DSM_Future_all.columns.str.contains('stock')]

        DSM_Future_Inflow.plot.line()
        plt.ylabel('Floor Area (Mm2)')
        plt.title(plot_title + ' ' + scenario + ' ' + 'Floor Area Inflow (New Construction) by Structural System')
        plt.show();

        DSM_Future_Outflow.plot.line()
        plt.ylabel('Floor Area (Mm2)')
        plt.title(plot_title + ' ' + scenario + ' ' + 'Floor Area Outflow (New Construction) by Structural System')
        plt.show();

        DSM_Future_Stock.plot.line()
        plt.ylabel('Floor Area (Mm2)')
        plt.title(plot_title + ' ' + scenario + ' ' + 'Floor Area Stock (New Construction) by Structural System')
        plt.show();

    return DSM_Future_all

sio_new_bldg_SSP1 = determine_inflow_outflow_new_bldg('S_timber_high',FA_dsm_SSP=FA_dsm_SSP1, plot=False, plot_title='SSP1 ')
sio_new_bldg_SSP2 = determine_inflow_outflow_new_bldg('S_timber_high',FA_dsm_SSP=FA_dsm_SSP2, plot=False, plot_title='SSP2 ')
sio_new_bldg_SSP3 = determine_inflow_outflow_new_bldg('S_timber_high',FA_dsm_SSP=FA_dsm_SSP3, plot=False, plot_title='SSP3 ')
sio_new_bldg_SSP4 = determine_inflow_outflow_new_bldg('S_timber_high',FA_dsm_SSP=FA_dsm_SSP4, plot=False, plot_title='SSP4 ')
sio_new_bldg_SSP5 = determine_inflow_outflow_new_bldg('S_timber_high',FA_dsm_SSP=FA_dsm_SSP5, plot=False, plot_title='SSP5 ')





## Next steps
# - add together outflows
# - compute material inflows and material outflows from floor areas.














# # -------------------------------------------------------------------------------------------------------------------

# Calculate historical floor area inflow/outflow by structural system (unit = million m2)
sio_area_by_ss_1820_2020 = pd.DataFrame()
sio_area_by_ss_1820_2020['LF_wood_s'] = total_area.loc[0:2020, 'stock_total'] * structure_data_historical.LF_wood[0]
sio_area_by_ss_1820_2020['LF_wood_i'] = total_area.loc[0:2020, 'inflow_total'] * structure_data_historical.LF_wood[0]
sio_area_by_ss_1820_2020['LF_wood_o'] = total_area.loc[0:2020, 'outflow_total'] * structure_data_historical.LF_wood[0]
sio_area_by_ss_1820_2020['Mass_Timber_s'] = total_area.loc[0:2020, 'stock_total'] * structure_data_historical.Mass_Timber[0]
sio_area_by_ss_1820_2020['Mass_Timber_i'] = total_area.loc[0:2020, 'inflow_total'] * structure_data_historical.Mass_Timber[0]
sio_area_by_ss_1820_2020['Mass_Timber_o'] = total_area.loc[0:2020, 'outflow_total'] * structure_data_historical.Mass_Timber[0]
sio_area_by_ss_1820_2020['Steel_s'] = total_area.loc[0:2020, 'stock_total'] * structure_data_historical.Steel[0]
sio_area_by_ss_1820_2020['Steel_i'] = total_area.loc[0:2020, 'inflow_total'] * structure_data_historical.Steel[0]
sio_area_by_ss_1820_2020['Steel_o'] = total_area.loc[0:2020, 'outflow_total'] * structure_data_historical.Steel[0]
sio_area_by_ss_1820_2020['RC_s'] = total_area.loc[0:2020, 'stock_total'] * structure_data_historical.RC[0]
sio_area_by_ss_1820_2020['RC_i'] = total_area.loc[0:2020, 'inflow_total'] * structure_data_historical.RC[0]
sio_area_by_ss_1820_2020['RC_o'] = total_area.loc[0:2020, 'outflow_total'] * structure_data_historical.RC[0]
sio_area_by_ss_1820_2020['RM_s'] = total_area.loc[0:2020, 'stock_total'] * structure_data_historical.RM[0]
sio_area_by_ss_1820_2020['RM_i'] = total_area.loc[0:2020, 'inflow_total'] * structure_data_historical.RM[0]
sio_area_by_ss_1820_2020['RM_o'] = total_area.loc[0:2020, 'outflow_total'] * structure_data_historical.RM[0]
sio_area_by_ss_1820_2020['URM_s'] = total_area.loc[0:2020, 'stock_total'] * structure_data_historical.URM[0]
sio_area_by_ss_1820_2020['URM_i'] = total_area.loc[0:2020, 'inflow_total'] * structure_data_historical.URM[0]
sio_area_by_ss_1820_2020['URM_o'] = total_area.loc[0:2020, 'outflow_total'] * structure_data_historical.URM[0]
sio_area_by_ss_1820_2020['MH_s'] = total_area.loc[0:2020, 'stock_total'] * structure_data_historical.MH[0]
sio_area_by_ss_1820_2020['MH_i'] = total_area.loc[0:2020, 'inflow_total'] * structure_data_historical.MH[0]
sio_area_by_ss_1820_2020['MH_o'] = total_area.loc[0:2020, 'outflow_total'] * structure_data_historical.MH[0]

# Calculate future floor area inflow/outflow by structural system (unit = million m2)
sio_area_by_ss_2020_2100 = pd.DataFrame()
sio_area_by_ss_2020_2100['LF_wood_s'] = total_area.loc[2021:2100, 'stock_total'] * structure_data_historical.LF_wood[0]
sio_area_by_ss_2020_2100['LF_wood_i'] = total_area.loc[2021:2100, 'inflow_total'] * structure_data_historical.LF_wood[0]
sio_area_by_ss_2020_2100['LF_wood_o'] = total_area.loc[2021:2100, 'outflow_total'] * structure_data_historical.LF_wood[0]
sio_area_by_ss_2020_2100['Mass_Timber_s'] = total_area.loc[2021:2100, 'stock_total'] * structure_data_historical.Mass_Timber[0]
sio_area_by_ss_2020_2100['Mass_Timber_i'] = total_area.loc[2021:2100, 'inflow_total'] * structure_data_historical.Mass_Timber[0]
sio_area_by_ss_2020_2100['Mass_Timber_o'] = total_area.loc[2021:2100, 'outflow_total'] * structure_data_historical.Mass_Timber[0]
sio_area_by_ss_2020_2100['Steel_s'] = total_area.loc[2021:2100, 'stock_total'] * structure_data_historical.Steel[0]
sio_area_by_ss_2020_2100['Steel_i'] = total_area.loc[2021:2100, 'inflow_total'] * structure_data_historical.Steel[0]
sio_area_by_ss_2020_2100['Steel_o'] = total_area.loc[2021:2100, 'outflow_total'] * structure_data_historical.Steel[0]
sio_area_by_ss_2020_2100['RC_s'] = total_area.loc[2021:2100, 'stock_total'] * structure_data_historical.RC[0]
sio_area_by_ss_2020_2100['RC_i'] = total_area.loc[2021:2100, 'inflow_total'] * structure_data_historical.RC[0]
sio_area_by_ss_2020_2100['RC_o'] = total_area.loc[2021:2100, 'outflow_total'] * structure_data_historical.RC[0]
sio_area_by_ss_2020_2100['RM_s'] = total_area.loc[2021:2100, 'stock_total'] * structure_data_historical.RM[0]
sio_area_by_ss_2020_2100['RM_i'] = total_area.loc[2021:2100, 'inflow_total'] * structure_data_historical.RM[0]
sio_area_by_ss_2020_2100['RM_o'] = total_area.loc[2021:2100, 'outflow_total'] * structure_data_historical.RM[0]
sio_area_by_ss_2020_2100['URM_s'] = total_area.loc[2021:2100, 'stock_total'] * structure_data_historical.URM[0]
sio_area_by_ss_2020_2100['URM_i'] = total_area.loc[2021:2100, 'inflow_total'] * structure_data_historical.URM[0]
sio_area_by_ss_2020_2100['URM_o'] = total_area.loc[2021:2100, 'outflow_total'] * structure_data_historical.URM[0]
sio_area_by_ss_2020_2100['MH_s'] = total_area.loc[2021:2100, 'stock_total'] * structure_data_historical.MH[0]
sio_area_by_ss_2020_2100['MH_i'] = total_area.loc[2021:2100, 'inflow_total'] * structure_data_historical.MH[0]
sio_area_by_ss_2020_2100['MH_o'] = total_area.loc[2021:2100, 'outflow_total'] * structure_data_historical.MH[0]

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
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(steel_tot_i.index, steel_tot_i.Sum)
axs[0, 0].plot([2020, 2020], [0, 1.2*steel_tot_i.Sum.max()], color='k', LineStyle='--')
axs[0, 0].set_title('Steel')
axs[0, 0].set_xlim(2000)

axs[0, 1].plot(conc_tot_i.index, conc_tot_i.Sum, 'tab:orange')
axs[0, 1].plot([2020, 2020], [0, 1.2*conc_tot_i.Sum.max()], color='k', LineStyle='--')
axs[0, 1].set_title('Concrete')
axs[0, 1].set_xlim(2000)

axs[1, 0].plot(engwood_tot_i.index, engwood_tot_i.Sum, 'tab:green')
axs[1, 0].plot([2020, 2020], [0, 1.2*engwood_tot_i.Sum.max()], color='k', LineStyle='--')
axs[1, 0].set_title('Engineered Wood')
axs[1, 0].set_xlim(2000)

axs[1, 1].plot(dimlum_tot_i.index, dimlum_tot_i.Sum, 'tab:red')
axs[1, 1].plot([2020, 2020], [0, 1.2*dimlum_tot_i.Sum.max()], color='k', LineStyle='--')
axs[1, 1].set_title('Dimensioned Lumber')
axs[1, 1].set_xlim(2000)

axs[2, 0].plot(masonry_tot_i.index, masonry_tot_i.Sum, 'tab:purple')
axs[2, 0].plot([2020, 2020], [0, 1.2*masonry_tot_i.Sum.max()], color='k', LineStyle='--')
axs[2, 0].set_title('Masonry')
axs[2, 0].set_xlim(2000)
# delete the sixth space
fig.delaxes(axs[2,1])
# add ylabels
for ax in axs.flat:
    ax.set(ylabel='Mt/year')
fig.show()

# print the material demand for a particular year
year = 2015
print('Total steel demand in ', str(year),  ' =   ', steel_tot_i['Sum'][year], ' Mt')
print('Total concrete demand in ', str(year),  ' =   ', conc_tot_i['Sum'][year], ' Mt')
print('Total engineered wood demand in ', str(year),  ' =   ', engwood_tot_i['Sum'][year], ' Mt')
print('Total dimensioned lumber demand in ', str(year),  ' =   ', dimlum_tot_i['Sum'][year], ' Mt')
print('Total masonry demand in ', str(year),  ' =   ', masonry_tot_i['Sum'][year], ' Mt')



