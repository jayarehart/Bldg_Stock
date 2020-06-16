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
import scipy.stats as st

# ----------------------------------------------------------------------------------------------------
# load in data from other scripts and excels
structure_data_historical = pd.read_csv('./InputData/HAZUS_weight.csv')

FA_dsm_SSP1 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP1')
FA_dsm_SSP1 = FA_dsm_SSP1.set_index('time', drop=False)
# FA_dsm_SSP2 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP2')
# FA_dsm_SSP2 = FA_dsm_SSP2.set_index('time',drop=False)
FA_dsm_SSP3 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP3')
FA_dsm_SSP3 = FA_dsm_SSP3.set_index('time',drop=False)
# FA_dsm_SSP4 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP4')
# FA_dsm_SSP4 = FA_dsm_SSP4.set_index('time',drop=False)
# FA_dsm_SSP5 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP5')
# FA_dsm_SSP5 = FA_dsm_SSP5.set_index('time',drop=False)

FA_sc_SSP1 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP1_sc')
# FA_sc_SSP2 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP2_sc')
FA_sc_SSP3 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP3_sc')
# FA_sc_SSP4 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP4_sc')
# FA_sc_SSP5 = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP5_sc')

materials_intensity = pd.read_excel('./InputData/Material_data.xlsx', sheet_name='SSP1_density')
materials_intensity_df = materials_intensity.set_index('Structure_Type', drop=True)
materials_intensity_df = materials_intensity_df.transpose()
materials_intensity_df = materials_intensity_df.drop(index='Source')

scenario_df = pd.read_excel('./InputData/Material_data.xlsx', sheet_name='Scenarios')
scenario_df = scenario_df.set_index('Scenario')

## ----------------------------------------------------------------------------------------------------
# set years series
years_future = FA_dsm_SSP1['time'].iloc[197:]       #2017 is 197
years_all = FA_dsm_SSP1.index.to_series()

## ----------------------------------------------------------------------------------------------------
# All Functions

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
    if type == 'Normal':
        # Normal
        lt = {'Type': type, 'Mean': np.array([par1] * len(years_all)), 'StdDev': np.array([par2])}
    elif type == 'Weibull':
        # Weibull
        # lt_res = {'Type': 'Weibull', 'Shape': np.array([4.16343417]), 'Scale': np.array([85.18683893])}     # deetman_2018_res_distr_weibull
        # lt_res = {'Type': 'Weibull', 'Shape': np.array([5.5]), 'Scale': np.array([85.8])}
        # lt_com = {'Type': 'Weibull', 'Shape': np.array([4.8]), 'Scale': np.array([75.1])}
        # lt_res = {'Type': type, 'Shape': np.array([5]), 'Scale': np.array([130])}
        # lt_com = {'Type': type, 'Shape': np.array([3]), 'Scale': np.array([100])}
        # lt_pub = {'Type': type, 'Shape': np.array([6.1]), 'Scale': np.array([95.6])}
        lt = {'Type': type, 'Shape': np.array([par1]), 'Scale': np.array([par2])}
    return lt
lt_dummy = generate_lt('Weibull', par1=5, par2=100)

# of outflow of each structural system type for already built buildings (before 2017). No new construction is considered in this analysis
def determine_outflow_existing_bldgs(FA_sc_SSP, lt, plot=True, plot_title=''):
    '''Input a floor area stock-cohort matrix for each SSP and compute the outflow for each structural system that are already built.
     Key assumption is that construction techniques are the same each year. '''

    # compute an outflow for the existing stock using a "compute evolution from initial stock method"
    def determine_outflow_by_ss(lt=lt_dummy, FA_sc_df=FA_sc_SSP, switch_year=196, frac_stock=1.0):
        '''Compute the outflow of the existing building stock with no additional inflow.
        A switch year of 196 represents 2016.
        frac_stock is the ratio of the exisitng building stock that is a particular structural system.
        For frac_stock = 1.0, all of the stock is considered to be the same. '''
        DSM_existing_stock = dsm.DynamicStockModel(t=years_all, lt=lt)
        S_C = DSM_existing_stock.compute_evolution_initialstock(
            InitialStock=FA_sc_SSP.loc[(switch_year - 1), 0:(switch_year - 1)] * frac_stock, SwitchTime=switch_year)

        # compute outflow
        O_C = DSM_existing_stock.o_c[1::, :] = -1 * np.diff(DSM_existing_stock.s_c, n=1, axis=0)
        O_C = DSM_existing_stock.o_c[np.diag_indices(len(DSM_existing_stock.t))] = 0 - np.diag(
            DSM_existing_stock.s_c)  # allow for outflow in year 0 already
        O = DSM_existing_stock.compute_outflow_total()
        # compute stock
        S = DSM_existing_stock.s_c.sum(axis=1)
        outflow_df = pd.DataFrame({'time': DSM_existing_stock.t, 'outflow': O, 'stock': S})
        return outflow_df

    existing_outflow_total = determine_outflow_by_ss(lt=lt, FA_sc_df=FA_sc_SSP, switch_year=196,
                                                     frac_stock=1.0)
    existing_outflow_LF_wood = determine_outflow_by_ss(lt=lt, FA_sc_df=FA_sc_SSP, switch_year=196,
                                                       frac_stock=structure_data_historical.LF_wood[0])
    existing_outflow_Mass_Timber = determine_outflow_by_ss(lt=lt, FA_sc_df=FA_sc_SSP, switch_year=196,
                                                           frac_stock=structure_data_historical.Mass_Timber[0])
    existing_outflow_Steel = determine_outflow_by_ss(lt=lt, FA_sc_df=FA_sc_SSP, switch_year=196,
                                                     frac_stock=structure_data_historical.Steel[0])
    existing_outflow_RC = determine_outflow_by_ss(lt=lt, FA_sc_df=FA_sc_SSP, switch_year=196,
                                                  frac_stock=structure_data_historical.RC[0])
    existing_outflow_RM = determine_outflow_by_ss(lt=lt, FA_sc_df=FA_sc_SSP, switch_year=196,
                                                  frac_stock=structure_data_historical.RM[0])
    existing_outflow_URM = determine_outflow_by_ss(lt=lt, FA_sc_df=FA_sc_SSP, switch_year=196,
                                                   frac_stock=structure_data_historical.URM[0])
    existing_outflow_MH = determine_outflow_by_ss(lt=lt, FA_sc_df=FA_sc_SSP, switch_year=196,
                                                  frac_stock=structure_data_historical.MH[0])

    existing_outflow_all = pd.DataFrame({
        'outflow_LF_wood': existing_outflow_LF_wood.outflow,
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
        'total_outflow': existing_outflow_total.outflow,
        'total_stock': existing_outflow_total.stock
    })
    existing_outflow_all = existing_outflow_all.set_index(FA_dsm_SSP1['time'])
    if plot == True:
        # plot the outflow and the stock
        existing_outflow = existing_outflow_all.loc[:, existing_outflow_all.columns.str.contains('outflow')]
        existing_stock = existing_outflow_all.loc[:, existing_outflow_all.columns.str.contains('stock')]

        existing_outflow.iloc[197:].plot.line()
        plt.ylabel('Floor Area (Mm2)')
        plt.title(plot_title + ': Outflow of Buildings Constructed before 2017')
        plt.show();

        existing_stock.iloc[197:].plot.line()
        plt.ylabel('Floor Area (Mm2)')
        plt.title(plot_title + ': Stock of Buildings Constructed before 2017')
        plt.show();

        # existing_outflow_all.iloc[197:].plot.line()
        # plt.ylabel('Floor Area (Mm2)')
        # plt.title(plot_title + ': Outflow of Buildings Constructed before 2017')
        # plt.show()
    return existing_outflow_all

def determine_inflow_outflow_new_bldg(scenario, FA_dsm_SSP=FA_dsm_SSP1, lt=lt_dummy, plot=True, plot_title='SSP1 '):
    # Select a scenario
    # scenario = 'S_1'    # new construction is same as exiting building stock
    # scenario = 'S_timber_high'      # High timber adoption

    # clean df
    FA_dsm_SSP = FA_dsm_SSP.set_index('time')

    def construction_ea_year(year1=2017, year2=2100, scenario_df=scenario_df, type='linear', plot=False):
        ''' Create a dataframe of the construction in each year for each structure type based upon the scenario.
            Do this in a 'linear' or 's-curve' manner by specifying the adoption in the year2 (2100)
        '''
        if type=='flat':
            year1_adoption = scenario_df.loc[scenario, 'LF_wood':'RM']

            x = np.linspace(year1, year2, num=(year2 - year1 + 1), endpoint=True)
            x_array = np.repeat(x[:, np.newaxis], 7, 1)
            year1_adoption_array = np.repeat(year1_adoption[:, np.newaxis], (year2 - year1 + 1), 1).transpose()
            adoption_df = pd.DataFrame(year1_adoption_array)
            adoption_df['time'] = FA_dsm_SSP.loc[year1:year2].index
            adoption_df = adoption_df.set_index('time')
            adoption_df.columns = list(scenario_df.columns)[2:9]
            if plot==True:
                # plot adoption data:
                adoption_df.plot()
                # plt.text(x=2100, y=0.7, ha='right', fontsize=8, s='(Growth Rate: ' + str(growth_rate) + ', ' + 'Max Growth Year: ' + str(Max_growth_yr) + ')')
                plt.legend(loc=(1.04, 0.5), title='Flat Adoption')
                plt.title('Scenario: ' + scenario + ' Type of Construction each Year')
                plt.show()

            return adoption_df
        if type=='linear':
            year1_adoption = scenario_df.loc['S_0', 'LF_wood':'RM']
            year2_adoption = scenario_df.loc[scenario, 'LF_wood':'RM']
            adoption_rate = (year2_adoption - year1_adoption) / (year2 - year1)

            x = np.linspace(year1, year2, num=(year2 - year1 + 1), endpoint=True)
            x_array = np.repeat(x[:,np.newaxis], 7, 1)
            year1_adoption_array = np.repeat(year1_adoption[:, np.newaxis],(year2 - year1 + 1),1).transpose()
            adoption_df = pd.DataFrame(year1_adoption_array + (x_array - year1) * np.repeat(adoption_rate[:, np.newaxis], (year2 - year1 + 1), 1).transpose())
            adoption_df['time'] = FA_dsm_SSP.loc[year1:year2].index
            adoption_df = adoption_df.set_index('time')
            adoption_df.columns = list(scenario_df.columns)[2:9]
            if plot == True:
                # plot adoption data:
                adoption_df.plot()
                # plt.text(x=2100, y=0.7, ha='right', fontsize=8, s='(Growth Rate: ' + str(growth_rate) + ', ' + 'Max Growth Year: ' + str(Max_growth_yr) + ')')
                plt.legend(loc=(1.04, 0.5), title='Linear Adoption')
                plt.title('Scenario: ' + scenario + ' Type of Construction each Year')
                plt.show()

            return adoption_df
        if type=='richards-curve':
            year1_adoption = scenario_df.loc['S_0', 'LF_wood':'RM']
            year2_adoption = scenario_df.loc[scenario, 'LF_wood':'RM']

            def compute_logistic(structural_system, year1_adoption, year2_adoption, growth_rate=0.1, M=2050):
                # structural_system = 'LF_wood'
                # carrying capacity
                B = year2_adoption[structural_system]
                A = year1_adoption[structural_system]
                # logistic growth rate:
                t = np.linspace(year1, year2, num=(year2 - year1 + 1), endpoint=True)
                t_0 = (year1 + year2) / 2
                y = A + (B - A) / (1 +  np.exp(-growth_rate * (t-M)))
                # plt.plot(t, y)
                # plt.show();
                return y

            # INPUT PARAMETERS
            growth_rate = 0.2
            Max_growth_yr = 2030

            adoption_df = pd.DataFrame()
            adoption_df['time'] = FA_dsm_SSP.loc[year1:year2].index
            adoption_df['LF_wood'] = compute_logistic('LF_wood', year1_adoption, year2_adoption, growth_rate=growth_rate, M=Max_growth_yr)
            adoption_df['URM'] = compute_logistic('URM', year1_adoption, year2_adoption, growth_rate=growth_rate, M=Max_growth_yr)
            adoption_df['MH'] = compute_logistic('MH', year1_adoption, year2_adoption, growth_rate=growth_rate, M=Max_growth_yr)
            adoption_df['Mass_Timber'] = compute_logistic('Mass_Timber', year1_adoption, year2_adoption, growth_rate=growth_rate, M=Max_growth_yr)
            adoption_df['Steel'] = compute_logistic('Steel', year1_adoption, year2_adoption, growth_rate=growth_rate, M=Max_growth_yr)
            adoption_df['RC'] = compute_logistic('RC', year1_adoption, year2_adoption, growth_rate=growth_rate, M=Max_growth_yr)
            adoption_df['RM'] = compute_logistic('RM', year1_adoption, year2_adoption, growth_rate=growth_rate, M=Max_growth_yr)
            adoption_df = adoption_df.set_index('time')
            if plot == True:
                adoption_df.plot()
                # plt.text(x=2100, y=0.7, ha='right', fontsize=8, s='(Growth Rate: ' + str(growth_rate) + ', ' + 'Max Growth Year: ' + str(Max_growth_yr) + ')')
                plt.legend(loc=(1.04, 0.5), title='(Growth Rate: ' + str(growth_rate) + ', ' '\n' + 'Max Growth Year: ' + str(Max_growth_yr) + ')')
                plt.title('Scenario: ' + scenario + ' Type of Construction each Year')
                plt.show()

            return adoption_df

    construction_ea_year_df = construction_ea_year(year1=2017, year2=2100, scenario_df=scenario_df, type='richards-curve', plot=False)
    # construction_ea_year_df = construction_ea_year(year1=2017, year2=2100, scenario_df=scenario_df, type='flat')
    # construction_ea_year_df = construction_ea_year(year1=2017, year2=2100, scenario_df=scenario_df, type='linear')


    # Percentage in each year of structural systems built.
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
        {'inflow_total': FA_dsm_SSP.loc[2017:2100, 'inflow_total'],
         'inflow_LF_wood': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * construction_ea_year_df.LF_wood,
         'inflow_Mass_Timber': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * construction_ea_year_df.Mass_Timber,
         'inflow_Steel': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * construction_ea_year_df.Steel,
         'inflow_RC': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * construction_ea_year_df.RC,
         'inflow_RM': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * construction_ea_year_df.RM,
         'inflow_URM': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * construction_ea_year_df.URM,
         'inflow_MH': FA_dsm_SSP.loc[2017:2100, 'inflow_total'] * construction_ea_year_df.MH,
         }
    )

    def compute_inflow_driven_model_ea_ss(structural_system):
        ''' Compute an inflow driven model for each structural system.
         The lifetime distribution in the future is assumed to be the input lt for each scenario'''
        # compute a inflow driven model for each structural system
        DSM_Inflow_x = dsm.DynamicStockModel(t=years_future, i=inflow_SSP_all['inflow_' + structural_system], lt=lt)
        # CheckStr = DSM_Inflow.dimension_check()
        # print(CheckStr)

        S_C = DSM_Inflow_x.compute_s_c_inflow_driven()
        O_C = DSM_Inflow_x.compute_o_c_from_s_c()
        S = DSM_Inflow_x.compute_stock_total()
        O = DSM_Inflow_x.compute_outflow_total()
        DSM_Inflow_x.o = pd.Series(DSM_Inflow_x.o, index=DSM_Inflow_x.t)
        return DSM_Inflow_x

    # compute an inflow driven model for new construction in the future
    DSM_Inflow_LF_wood = compute_inflow_driven_model_ea_ss(structural_system='LF_wood')
    DSM_Inflow_Mass_Timber = compute_inflow_driven_model_ea_ss('Mass_Timber')
    DSM_Inflow_Steel = compute_inflow_driven_model_ea_ss('Steel')
    DSM_Inflow_RC = compute_inflow_driven_model_ea_ss('RC')
    DSM_Inflow_RM = compute_inflow_driven_model_ea_ss('RM')
    DSM_Inflow_URM = compute_inflow_driven_model_ea_ss('URM')
    DSM_Inflow_MH = compute_inflow_driven_model_ea_ss('MH')

    # summary dataframe of all DSM stocks, inflows, outflows
    DSM_Future_all = pd.DataFrame({
        'inflow_LF_wood': DSM_Inflow_LF_wood.i,
        'outflow_LF_wood': DSM_Inflow_LF_wood.o,
        'stock_LF_wood': DSM_Inflow_LF_wood.s,
        'inflow_Mass_Timber': DSM_Inflow_Mass_Timber.i,
        'outflow_Mass_Timber': DSM_Inflow_Mass_Timber.o,
        'stock_Mass_Timber': DSM_Inflow_Mass_Timber.s,
        'inflow_Steel': DSM_Inflow_Steel.i,
        'outflow_Steel': DSM_Inflow_Steel.o,
        'stock_Steel': DSM_Inflow_Steel.s,
        'inflow_RC': DSM_Inflow_RC.i,
        'outflow_RC': DSM_Inflow_RC.o,
        'stock_RC': DSM_Inflow_RC.s,
        'inflow_RM': DSM_Inflow_RM.i,
        'outflow_RM': DSM_Inflow_RM.o,
        'stock_RM': DSM_Inflow_RM.s,
        'inflow_URM': DSM_Inflow_URM.i,
        'outflow_URM': DSM_Inflow_URM.o,
        'stock_URM': DSM_Inflow_URM.s,
        'inflow_MH': DSM_Inflow_MH.i,
        'outflow_MH': DSM_Inflow_MH.o,
        'stock_MH': DSM_Inflow_MH.s,
        'total_inflow': DSM_Inflow_LF_wood.i + DSM_Inflow_Mass_Timber.i + DSM_Inflow_Steel.i + DSM_Inflow_RC.i + DSM_Inflow_RM.i + DSM_Inflow_URM.i + DSM_Inflow_MH.i,
        'total_outflow': DSM_Inflow_LF_wood.o + DSM_Inflow_Mass_Timber.o + DSM_Inflow_Steel.o + DSM_Inflow_RC.o + DSM_Inflow_RM.o + DSM_Inflow_URM.o + DSM_Inflow_MH.o,
        'total_stock': DSM_Inflow_LF_wood.s + DSM_Inflow_Mass_Timber.s + DSM_Inflow_Steel.s + DSM_Inflow_RC.s + DSM_Inflow_RM.s + DSM_Inflow_URM.s + DSM_Inflow_MH.s
    })

    if plot == True:
        DSM_Future_Inflow = DSM_Future_all.loc[:, DSM_Future_all.columns.str.contains('inflow')]
        DSM_Future_Outflow = DSM_Future_all.loc[:, DSM_Future_all.columns.str.contains('outflow')]
        DSM_Future_Stock = DSM_Future_all.loc[:, DSM_Future_all.columns.str.contains('stock')]

        # fig, axes = plt.subplots(nrows=3, ncols=1)
        #
        # DSM_Future_Inflow.plot(ax=axes[0, 0])
        # DSM_Future_Outflow.plot(ax=axes[0, 1])
        # DSM_Future_Stock.plot(ax=axes[0, 2])

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

# combine and plot area inflow and outflows after differentiating between existing buildings and new construction
def combine_area_existing_and_new(os_existing, sio_new_bldg, plot=True, plot_title='SSP#, Scenario'):
    area_stock_2017_2100 = pd.DataFrame({
        'stock_all': os_existing.loc[2017:]['total_stock'] + sio_new_bldg.loc[2017:]['total_stock'],
        'stock_LF_wood': os_existing.loc[2017:]['stock_LF_wood'] + sio_new_bldg.loc[2017:][
            'stock_LF_wood'],
        'stock_Mass_Timber': os_existing.loc[2017:]['stock_Mass_Timber'] + sio_new_bldg.loc[2017:][
            'stock_Mass_Timber'],
        'stock_Steel': os_existing.loc[2017:]['stock_Steel'] + sio_new_bldg.loc[2017:]['stock_Steel'],
        'stock_RC': os_existing.loc[2017:]['stock_RC'] + sio_new_bldg.loc[2017:]['stock_RC'],
        'stock_RM': os_existing.loc[2017:]['stock_RM'] + sio_new_bldg.loc[2017:]['stock_RM'],
        'stock_URM': os_existing.loc[2017:]['stock_URM'] + sio_new_bldg.loc[2017:]['stock_URM'],
        'stock_MH': os_existing.loc[2017:]['stock_MH'] + sio_new_bldg.loc[2017:]['stock_MH']
    })
    area_inflow_2017_2100 = pd.DataFrame({
        'inflow_all': sio_new_bldg.loc[2017:]['total_inflow'],
        'inflow_LF_wood': sio_new_bldg.loc[2017:]['inflow_LF_wood'],
        'inflow_Mass_Timber': sio_new_bldg.loc[2017:]['inflow_Mass_Timber'],
        'inflow_Steel': sio_new_bldg.loc[2017:]['inflow_Steel'],
        'inflow_RC': sio_new_bldg.loc[2017:]['inflow_RC'],
        'inflow_RM': sio_new_bldg.loc[2017:]['inflow_RM'],
        'inflow_URM': sio_new_bldg.loc[2017:]['inflow_URM'],
        'inflow_MH': sio_new_bldg.loc[2017:]['inflow_MH']
    })
    area_outflow_2017_2100 = pd.DataFrame({
        'outflow_all': os_existing.loc[2017:]['total_outflow'] + sio_new_bldg.loc[2017:]['total_outflow'],
        'outflow_LF_wood': os_existing.loc[2017:]['outflow_LF_wood'] + sio_new_bldg.loc[2017:][
            'outflow_LF_wood'],
        'outflow_Mass_Timber': os_existing.loc[2017:]['outflow_Mass_Timber'] + sio_new_bldg.loc[2017:][
            'outflow_Mass_Timber'],
        'outflow_Steel': os_existing.loc[2017:]['outflow_Steel'] + sio_new_bldg.loc[2017:][
            'outflow_Steel'],
        'outflow_RC': os_existing.loc[2017:]['outflow_RC'] + sio_new_bldg.loc[2017:]['outflow_RC'],
        'outflow_RM': os_existing.loc[2017:]['outflow_RM'] + sio_new_bldg.loc[2017:]['outflow_RM'],
        'outflow_URM': os_existing.loc[2017:]['outflow_URM'] + sio_new_bldg.loc[2017:]['outflow_URM'],
        'outflow_MH': os_existing.loc[2017:]['outflow_MH'] + sio_new_bldg.loc[2017:]['outflow_MH']
    })

    if plot == True:
        fig, axs = plt.subplots(3, 1, figsize=(6, 8))
        axs[0].plot(area_inflow_2017_2100.index, area_inflow_2017_2100['inflow_all'])
        axs[0].plot(area_inflow_2017_2100.index, area_inflow_2017_2100['inflow_LF_wood'])
        axs[0].plot(area_inflow_2017_2100.index, area_inflow_2017_2100['inflow_Mass_Timber'])
        axs[0].plot(area_inflow_2017_2100.index, area_inflow_2017_2100['inflow_Steel'])
        axs[0].plot(area_inflow_2017_2100.index, area_inflow_2017_2100['inflow_RC'])
        axs[0].plot(area_inflow_2017_2100.index, area_inflow_2017_2100['inflow_RM'])
        axs[0].plot(area_inflow_2017_2100.index, area_inflow_2017_2100['inflow_URM'])
        axs[0].plot(area_inflow_2017_2100.index, area_inflow_2017_2100['inflow_MH'])
        axs[0].set_ylabel('Floor Area ($10^6 m^2$)')
        axs[0].set_title('Inflow')

        # axs1.legend(['All', 'LF Wood', 'Mass Timber', 'Steel', 'RC', 'RM', 'URM', 'MH'])

        # plot outflow
        axs[1].plot(area_outflow_2017_2100.index, area_outflow_2017_2100['outflow_all'])
        axs[1].plot(area_outflow_2017_2100.index, area_outflow_2017_2100['outflow_LF_wood'])
        axs[1].plot(area_outflow_2017_2100.index, area_outflow_2017_2100['outflow_Mass_Timber'])
        axs[1].plot(area_outflow_2017_2100.index, area_outflow_2017_2100['outflow_Steel'])
        axs[1].plot(area_outflow_2017_2100.index, area_outflow_2017_2100['outflow_RC'])
        axs[1].plot(area_outflow_2017_2100.index, area_outflow_2017_2100['outflow_RM'])
        axs[1].plot(area_outflow_2017_2100.index, area_outflow_2017_2100['outflow_URM'])
        axs[1].plot(area_outflow_2017_2100.index, area_outflow_2017_2100['outflow_MH'])
        axs[1].set_ylabel('Floor Area ($10^6$ $m^2$)')
        axs[1].set_title('Outflow')
        axs[1].legend(['All', 'LF Wood', 'Mass Timber', 'Steel', 'RC', 'RM', 'URM', 'MH'],
                      bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # plot stock
        axs[2].plot(area_stock_2017_2100.index, area_stock_2017_2100['stock_all'])
        axs[2].plot(area_stock_2017_2100.index, area_stock_2017_2100['stock_LF_wood'])
        axs[2].plot(area_stock_2017_2100.index, area_stock_2017_2100['stock_Mass_Timber'])
        axs[2].plot(area_stock_2017_2100.index, area_stock_2017_2100['stock_Steel'])
        axs[2].plot(area_stock_2017_2100.index, area_stock_2017_2100['stock_RC'])
        axs[2].plot(area_stock_2017_2100.index, area_stock_2017_2100['stock_RM'])
        axs[2].plot(area_stock_2017_2100.index, area_stock_2017_2100['stock_URM'])
        axs[2].plot(area_stock_2017_2100.index, area_stock_2017_2100['stock_MH'])
        axs[2].set_title('Stock')
        axs[2].set_ylabel('Floor Area ($10^6$ $m^2$)')

        plt.figtext(0.95, 0.95, plot_title, fontsize=10, ha='right')

        plt.show();

    return area_inflow_2017_2100, area_outflow_2017_2100, area_stock_2017_2100

# Calculate material demand by structural system
def calc_inflow_outflow_stock_mats(area_inflow_2017_2100, area_outflow_2017_2100, area_stock_2017_2100,
                                   materials_intensity_df, print_year=0, detailed=False):
    '''Calculate the total material inflow and outflows based upon a material intensity dataframe'''
    calc_inflow = True
    if calc_inflow == True:
        # Inflow of materials in (unit = Megaton, Mt)
        inflow_mat = pd.DataFrame()
        inflow_mat['LF_wood_steel_inflow'] = area_inflow_2017_2100['inflow_LF_wood'] * \
                                             materials_intensity_df['LF_wood']['Steel_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['LF_wood_conc_inflow'] = area_inflow_2017_2100['inflow_LF_wood'] * materials_intensity_df['LF_wood'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['LF_wood_engwood_inflow'] = area_inflow_2017_2100['inflow_LF_wood'] * \
                                               materials_intensity_df['LF_wood']['Eng_wood_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['LF_wood_dimlum_inflow'] = area_inflow_2017_2100['inflow_LF_wood'] * \
                                              materials_intensity_df['LF_wood']['Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['LF_wood_masonry_inflow'] = area_inflow_2017_2100['inflow_LF_wood'] * \
                                               materials_intensity_df['LF_wood']['Masonry_kgm2_mean'] * 10e6 / 10e9

        inflow_mat['Mass_Timber_steel_inflow'] = area_inflow_2017_2100['inflow_Mass_Timber'] * \
                                                 materials_intensity_df['Mass_Timber']['Steel_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['Mass_Timber_conc_inflow'] = area_inflow_2017_2100['inflow_Mass_Timber'] * \
                                                materials_intensity_df['Mass_Timber'][
                                                    'Concrete_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['Mass_Timber_engwood_inflow'] = area_inflow_2017_2100['inflow_Mass_Timber'] * \
                                                   materials_intensity_df['Mass_Timber'][
                                                       'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['Mass_Timber_dimlum_inflow'] = area_inflow_2017_2100['inflow_Mass_Timber'] * \
                                                  materials_intensity_df['Mass_Timber'][
                                                      'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['Mass_Timber_masonry_inflow'] = area_inflow_2017_2100['inflow_Mass_Timber'] * \
                                                   materials_intensity_df['Mass_Timber'][
                                                       'Masonry_kgm2_mean'] * 10e6 / 10e9

        inflow_mat['Steel_steel_inflow'] = area_inflow_2017_2100['inflow_Steel'] * materials_intensity_df['Steel'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['Steel_conc_inflow'] = area_inflow_2017_2100['inflow_Steel'] * materials_intensity_df['Steel'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['Steel_engwood_inflow'] = area_inflow_2017_2100['inflow_Steel'] * materials_intensity_df['Steel'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['Steel_dimlum_inflow'] = area_inflow_2017_2100['inflow_Steel'] * materials_intensity_df['Steel'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['Steel_masonry_inflow'] = area_inflow_2017_2100['inflow_Steel'] * materials_intensity_df['Steel'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9

        inflow_mat['RC_steel_inflow'] = area_inflow_2017_2100['inflow_RC'] * materials_intensity_df['RC'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['RC_conc_inflow'] = area_inflow_2017_2100['inflow_RC'] * materials_intensity_df['RC'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['RC_engwood_inflow'] = area_inflow_2017_2100['inflow_RC'] * materials_intensity_df['RC'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['RC_dimlum_inflow'] = area_inflow_2017_2100['inflow_RC'] * materials_intensity_df['RC'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['RC_masonry_inflow'] = area_inflow_2017_2100['inflow_RC'] * materials_intensity_df['RC'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9

        inflow_mat['RM_steel_inflow'] = area_inflow_2017_2100['inflow_RM'] * materials_intensity_df['RM'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['RM_conc_inflow'] = area_inflow_2017_2100['inflow_RM'] * materials_intensity_df['RM'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['RM_engwood_inflow'] = area_inflow_2017_2100['inflow_RM'] * materials_intensity_df['RM'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['RM_dimlum_inflow'] = area_inflow_2017_2100['inflow_RM'] * materials_intensity_df['RM'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['RM_masonry_inflow'] = area_inflow_2017_2100['inflow_RM'] * materials_intensity_df['RM'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9

        inflow_mat['URM_steel_inflow'] = area_inflow_2017_2100['inflow_URM'] * materials_intensity_df['URM'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['URM_conc_inflow'] = area_inflow_2017_2100['inflow_URM'] * materials_intensity_df['URM'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['URM_engwood_inflow'] = area_inflow_2017_2100['inflow_URM'] * materials_intensity_df['URM'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['URM_dimlum_inflowv'] = area_inflow_2017_2100['inflow_URM'] * materials_intensity_df['URM'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        inflow_mat['URM_masonry_inflow'] = area_inflow_2017_2100['inflow_URM'] * materials_intensity_df['URM'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9

        # Total inflows of each material (by structure type), unit = Mt
        steel_tot_inflow = inflow_mat.filter(regex='_steel', axis=1)
        conc_tot_inflow = inflow_mat.filter(regex='_conc', axis=1)
        engwood_tot_inflow = inflow_mat.filter(regex='_engwood', axis=1)
        dimlum_tot_inflow = inflow_mat.filter(regex='_dimlum', axis=1)
        masonry_tot_inflow = inflow_mat.filter(regex='_masonry', axis=1)
        # adding in a sum row for each year
        steel_tot_inflow['Sum_steel_inflow'] = steel_tot_inflow.sum(axis=1)
        conc_tot_inflow['Sum_conc_inflow'] = conc_tot_inflow.sum(axis=1)
        engwood_tot_inflow['Sum_engwood_inflow'] = engwood_tot_inflow.sum(axis=1)
        dimlum_tot_inflow['Sum_dimlum_inflow'] = dimlum_tot_inflow.sum(axis=1)
        masonry_tot_inflow['Sum_masonry_inflow'] = masonry_tot_inflow.sum(axis=1)

        inflow_mat_all = pd.concat(
            [steel_tot_inflow, conc_tot_inflow, engwood_tot_inflow, dimlum_tot_inflow, masonry_tot_inflow], axis=1)

        # print the material demand for a particular year
        # print_year = 2020
        if print_year is not 0:
            print('Total steel demand in ', str(print_year), ' =   ', steel_tot_inflow['Sum_steel_inflow'][print_year],
                  ' Mt')
            print('Total concrete demand in ', str(print_year), ' =   ', conc_tot_inflow['Sum_conc_inflow'][print_year],
                  ' Mt')
            print('Total engineered wood demand in ', str(print_year), ' =   ',
                  engwood_tot_inflow['Sum_engwood_inflow'][print_year], ' Mt')
            print('Total dimensioned lumber demand in ', str(print_year), ' =   ',
                  dimlum_tot_inflow['Sum_dimlum_inflow'][print_year], ' Mt')
            print('Total masonry demand in ', str(print_year), ' =   ',
                  masonry_tot_inflow['Sum_masonry_inflow'][print_year], ' Mt')

    calc_outflow = True
    if calc_outflow == True:
        # Outflow of materials in (unit = Megaton, Mt)
        outflow_mat = pd.DataFrame()
        outflow_mat['LF_wood_steel_outflow'] = area_outflow_2017_2100['outflow_LF_wood'] * \
                                               materials_intensity_df['LF_wood'][
                                                   'Steel_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['LF_wood_conc_outflow'] = area_outflow_2017_2100['outflow_LF_wood'] * \
                                              materials_intensity_df['LF_wood'][
                                                  'Concrete_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['LF_wood_engwood_outflow'] = area_outflow_2017_2100['outflow_LF_wood'] * \
                                                 materials_intensity_df['LF_wood'][
                                                     'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['LF_wood_dimlum_outflow'] = area_outflow_2017_2100['outflow_LF_wood'] * \
                                                materials_intensity_df['LF_wood'][
                                                    'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['LF_wood_masonry_outflow'] = area_outflow_2017_2100['outflow_LF_wood'] * \
                                                 materials_intensity_df['LF_wood'][
                                                     'Masonry_kgm2_mean'] * 10e6 / 10e9

        outflow_mat['Mass_Timber_steel_outflow'] = area_outflow_2017_2100['outflow_Mass_Timber'] * \
                                                   materials_intensity_df['Mass_Timber'][
                                                       'Steel_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['Mass_Timber_conc_outflow'] = area_outflow_2017_2100['outflow_Mass_Timber'] * \
                                                  materials_intensity_df['Mass_Timber'][
                                                      'Concrete_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['Mass_Timber_engwood_outflow'] = area_outflow_2017_2100['outflow_Mass_Timber'] * \
                                                     materials_intensity_df['Mass_Timber'][
                                                         'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['Mass_Timber_dimlum_outflow'] = area_outflow_2017_2100['outflow_Mass_Timber'] * \
                                                    materials_intensity_df['Mass_Timber'][
                                                        'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['Mass_Timber_masonry_outflow'] = area_outflow_2017_2100['outflow_Mass_Timber'] * \
                                                     materials_intensity_df['Mass_Timber'][
                                                         'Masonry_kgm2_mean'] * 10e6 / 10e9

        outflow_mat['Steel_steel_outflow'] = area_outflow_2017_2100['outflow_Steel'] * materials_intensity_df['Steel'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['Steel_conc_outflow'] = area_outflow_2017_2100['outflow_Steel'] * materials_intensity_df['Steel'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['Steel_engwood_outflow'] = area_outflow_2017_2100['outflow_Steel'] * \
                                               materials_intensity_df['Steel'][
                                                   'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['Steel_dimlum_outflow'] = area_outflow_2017_2100['outflow_Steel'] * materials_intensity_df['Steel'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['Steel_masonry_outflow'] = area_outflow_2017_2100['outflow_Steel'] * \
                                               materials_intensity_df['Steel'][
                                                   'Masonry_kgm2_mean'] * 10e6 / 10e9

        outflow_mat['RC_steel_outflow'] = area_outflow_2017_2100['outflow_RC'] * materials_intensity_df['RC'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['RC_conc_outflow'] = area_outflow_2017_2100['outflow_RC'] * materials_intensity_df['RC'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['RC_engwood_outflow'] = area_outflow_2017_2100['outflow_RC'] * materials_intensity_df['RC'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['RC_dimlum_outflow'] = area_outflow_2017_2100['outflow_RC'] * materials_intensity_df['RC'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['RC_masonry_outflow'] = area_outflow_2017_2100['outflow_RC'] * materials_intensity_df['RC'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9

        outflow_mat['RM_steel_outflow'] = area_outflow_2017_2100['outflow_RM'] * materials_intensity_df['RM'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['RM_conc_outflow'] = area_outflow_2017_2100['outflow_RM'] * materials_intensity_df['RM'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['RM_engwood_outflow'] = area_outflow_2017_2100['outflow_RM'] * materials_intensity_df['RM'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['RM_dimlum_outflow'] = area_outflow_2017_2100['outflow_RM'] * materials_intensity_df['RM'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['RM_masonry_outflow'] = area_outflow_2017_2100['outflow_RM'] * materials_intensity_df['RM'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['URM_steel_outflow'] = area_outflow_2017_2100['outflow_URM'] * materials_intensity_df['URM'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['URM_conc_outflow'] = area_outflow_2017_2100['outflow_URM'] * materials_intensity_df['URM'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['URM_engwood_outflow'] = area_outflow_2017_2100['outflow_URM'] * materials_intensity_df['URM'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['URM_dimlum_outflow'] = area_outflow_2017_2100['outflow_URM'] * materials_intensity_df['URM'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        outflow_mat['URM_masonry_outflow'] = area_outflow_2017_2100['outflow_URM'] * materials_intensity_df['URM'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9

        # Total outflows of each material (by structure type), unit = Mt
        steel_tot_outflow = outflow_mat.filter(regex='_steel', axis=1)
        conc_tot_outflow = outflow_mat.filter(regex='_conc', axis=1)
        engwood_tot_outflow = outflow_mat.filter(regex='_engwood', axis=1)
        dimlum_tot_outflow = outflow_mat.filter(regex='_dimlum', axis=1)
        masonry_tot_outflow = outflow_mat.filter(regex='_masonry', axis=1)
        # adding in a sum row for each year
        steel_tot_outflow['Sum_steel_outflow'] = steel_tot_outflow.sum(axis=1)
        conc_tot_outflow['Sum_conc_outflow'] = conc_tot_outflow.sum(axis=1)
        engwood_tot_outflow['Sum_engwood_outflow'] = engwood_tot_outflow.sum(axis=1)
        dimlum_tot_outflow['Sum_dimlum_outflow'] = dimlum_tot_outflow.sum(axis=1)
        masonry_tot_outflow['Sum_masonry_outflow'] = masonry_tot_outflow.sum(axis=1)

        # print the material demand for a particular year
        # print_year = 2020
        if print_year is not 0:
            print('Total steel outflow in ', str(print_year), ' =   ', steel_tot_outflow['Sum_steel_outflow'][print_year],
                  ' Mt')
            print('Total concrete outflow in ', str(print_year), ' =   ', conc_tot_outflow['Sum_conc_outflow'][print_year],
                  ' Mt')
            print('Total engineered wood outflow in ', str(print_year), ' =   ',
                  engwood_tot_outflow['Sum_engwood_outflow'][print_year], ' Mt')
            print('Total dimensioned lumber outflow in ', str(print_year), ' =   ',
                  dimlum_tot_outflow['Sum_dimlum_outflow'][print_year], ' Mt')
            print('Total masonry outflow in ', str(print_year), ' =   ',
                  masonry_tot_outflow['Sum_masonry_outflow'][print_year], ' Mt')

        outflow_mat_all = pd.concat(
            [steel_tot_outflow, conc_tot_outflow, engwood_tot_outflow, dimlum_tot_outflow, masonry_tot_outflow], axis=1)

    calc_stock = True
    if calc_stock == True:
        # Stocks of materials in (unit = Megaton, Mt)
        stock_mat = pd.DataFrame()
        stock_mat['LF_wood_steel_stock'] = area_stock_2017_2100['stock_LF_wood'] * materials_intensity_df['LF_wood'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        stock_mat['LF_wood_conc_stock'] = area_stock_2017_2100['stock_LF_wood'] * materials_intensity_df['LF_wood'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        stock_mat['LF_wood_engwood_stock'] = area_stock_2017_2100['stock_LF_wood'] * materials_intensity_df['LF_wood'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        stock_mat['LF_wood_dimlum_stock'] = area_stock_2017_2100['stock_LF_wood'] * materials_intensity_df['LF_wood'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        stock_mat['LF_wood_masonry_stock'] = area_stock_2017_2100['stock_LF_wood'] * materials_intensity_df['LF_wood'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9

        stock_mat['Mass_Timber_steel_stock'] = area_stock_2017_2100['stock_Mass_Timber'] * \
                                               materials_intensity_df['Mass_Timber']['Steel_kgm2_mean'] * 10e6 / 10e9
        stock_mat['Mass_Timber_conc_stock'] = area_stock_2017_2100['stock_Mass_Timber'] * \
                                              materials_intensity_df['Mass_Timber']['Concrete_kgm2_mean'] * 10e6 / 10e9
        stock_mat['Mass_Timber_engwood_stock'] = area_stock_2017_2100['stock_Mass_Timber'] * \
                                                 materials_intensity_df['Mass_Timber'][
                                                     'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        stock_mat['Mass_Timber_dimlum_stock'] = area_stock_2017_2100['stock_Mass_Timber'] * \
                                                materials_intensity_df['Mass_Timber'][
                                                    'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        stock_mat['Mass_Timber_masonry_stock'] = area_stock_2017_2100['stock_Mass_Timber'] * \
                                                 materials_intensity_df['Mass_Timber'][
                                                     'Masonry_kgm2_mean'] * 10e6 / 10e9

        stock_mat['Steel_steel_stock'] = area_stock_2017_2100['stock_Steel'] * materials_intensity_df['Steel'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        stock_mat['Steel_conc_stock'] = area_stock_2017_2100['stock_Steel'] * materials_intensity_df['Steel'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        stock_mat['Steel_engwood_stock'] = area_stock_2017_2100['stock_Steel'] * materials_intensity_df['Steel'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        stock_mat['Steel_dimlum_stock'] = area_stock_2017_2100['stock_Steel'] * materials_intensity_df['Steel'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        stock_mat['Steel_masonry_stock'] = area_stock_2017_2100['stock_Steel'] * materials_intensity_df['Steel'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9

        stock_mat['RC_steel_stock'] = area_stock_2017_2100['stock_RC'] * materials_intensity_df['RC'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        stock_mat['RC_conc_stock'] = area_stock_2017_2100['stock_RC'] * materials_intensity_df['RC'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        stock_mat['RC_engwood_stock'] = area_stock_2017_2100['stock_RC'] * materials_intensity_df['RC'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        stock_mat['RC_dimlum_stock'] = area_stock_2017_2100['stock_RC'] * materials_intensity_df['RC'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        stock_mat['RC_masonry_stock'] = area_stock_2017_2100['stock_RC'] * materials_intensity_df['RC'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9

        stock_mat['RM_steel_stock'] = area_stock_2017_2100['stock_RM'] * materials_intensity_df['RM'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        stock_mat['RM_conc_stock'] = area_stock_2017_2100['stock_RM'] * materials_intensity_df['RM'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        stock_mat['RM_engwood_stock'] = area_stock_2017_2100['stock_RM'] * materials_intensity_df['RM'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        stock_mat['RM_dimlum_stock'] = area_stock_2017_2100['stock_RM'] * materials_intensity_df['RM'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        stock_mat['RM_masonry_stock'] = area_stock_2017_2100['stock_RM'] * materials_intensity_df['RM'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9

        stock_mat['URM_steel_stock'] = area_stock_2017_2100['stock_URM'] * materials_intensity_df['URM'][
            'Steel_kgm2_mean'] * 10e6 / 10e9
        stock_mat['URM_conc_stock'] = area_stock_2017_2100['stock_URM'] * materials_intensity_df['URM'][
            'Concrete_kgm2_mean'] * 10e6 / 10e9
        stock_mat['URM_engwood_stock'] = area_stock_2017_2100['stock_URM'] * materials_intensity_df['URM'][
            'Eng_wood_kgm2_mean'] * 10e6 / 10e9
        stock_mat['URM_dimlum_stock'] = area_stock_2017_2100['stock_URM'] * materials_intensity_df['URM'][
            'Dim_lumber_kgm2_mean'] * 10e6 / 10e9
        stock_mat['URM_masonry_stock'] = area_stock_2017_2100['stock_URM'] * materials_intensity_df['URM'][
            'Masonry_kgm2_mean'] * 10e6 / 10e9

        # Total stocks of each material (by structure type), unit = Mt
        steel_tot_stock = stock_mat.filter(regex='_steel', axis=1)
        conc_tot_stock = stock_mat.filter(regex='_conc', axis=1)
        engwood_tot_stock = stock_mat.filter(regex='_engwood', axis=1)
        dimlum_tot_stock = stock_mat.filter(regex='_dimlum', axis=1)
        masonry_tot_stock = stock_mat.filter(regex='_masonry', axis=1)
        # adding in a sum row for each year
        steel_tot_stock['Sum_steel_stock'] = steel_tot_stock.sum(axis=1)
        conc_tot_stock['Sum_conc_stock'] = conc_tot_stock.sum(axis=1)
        engwood_tot_stock['Sum_engwood_stock'] = engwood_tot_stock.sum(axis=1)
        dimlum_tot_stock['Sum_dimlum_stock'] = dimlum_tot_stock.sum(axis=1)
        masonry_tot_stock['Sum_masonry_stock'] = masonry_tot_stock.sum(axis=1)

        # print the material demand for a particular year
        # print_year = 2020
        if print_year is not 0:
            print('Total steel stock in ', str(print_year), ' =   ', steel_tot_stock['Sum_steel_stock'][print_year], ' Mt')
            print('Total concrete stock in ', str(print_year), ' =   ', conc_tot_stock['Sum_conc_stock'][print_year], ' Mt')
            print('Total engineered wood stock in ', str(print_year), ' =   ',
                  engwood_tot_stock['Sum_engwood_stock'][print_year], ' Mt')
            print('Total dimensioned lumber stock in ', str(print_year), ' =   ',
                  dimlum_tot_stock['Sum_dimlum_stock'][print_year], ' Mt')
            print('Total masonry stock in ', str(print_year), ' =   ', masonry_tot_stock['Sum_masonry_stock'][print_year],
                  ' Mt')

        stock_mat_all = pd.concat(
            [steel_tot_stock, conc_tot_stock, engwood_tot_stock, dimlum_tot_stock, masonry_tot_stock], axis=1)

    if detailed == True:
        return inflow_mat_all, outflow_mat_all, stock_mat_all

    return inflow_mat_all.loc[:, inflow_mat_all.columns.str.contains('Sum')], outflow_mat_all.loc[:,
                                                                              inflow_mat_all.columns.str.contains(
                                                                                  'Sum')], stock_mat_all.loc[:,
                                                                                           inflow_mat_all.columns.str.contains(
                                                                                               'Sum')]

## ----------------------------------------------------------------------------------------------------
# Lifetime parameters
lt_existing = generate_lt('Weibull', par1=((0.773497 * 5) + (0.142467 * 4.8) + (0.030018 * 6.1)), par2=(
            (0.773497 * 100) + (0.142467 * 75.1) + (0.030018 * 95.6)))  # weighted average of res, com, and pub
# lt_existing = generate_lt('Weibull',par1=5, par2=100)        # lifetime distribution for existing buildings (all)
lt_future = generate_lt('Weibull', par1=((0.773497 * 5) + (0.142467 * 4.8) + (0.030018 * 6.1)), par2=(
            (0.773497 * 100) + (0.142467 * 75.1) + (0.030018 * 95.6)))  # weighted average of res, com, and pub
# lt_future = generate_lt('Weibull', par1=5, par2=100)

## ----------------------------------------------------------------------------------------------------

# area by structural system of outlow and stock of all existing buildings (built before 2017)
os_existing_SSP1 = determine_outflow_existing_bldgs(FA_sc_SSP=FA_sc_SSP1,   lt=lt_existing, plot=False, plot_title='SSP1')
# os_existing_SSP2 = determine_outflow_existing_bldgs(FA_sc_SSP=FA_sc_SSP2, lt=lt_existing, plot=False, plot_title='SSP2')
os_existing_SSP3 = determine_outflow_existing_bldgs(FA_sc_SSP=FA_sc_SSP3,   lt=lt_existing, plot=False, plot_title='SSP3')
# os_existing_SSP4 = determine_outflow_existing_bldgs(FA_sc_SSP=FA_sc_SSP4, lt=lt_existing, plot=False, plot_title='SSP4')
# os_existing_SSP5 = determine_outflow_existing_bldgs(FA_sc_SSP=FA_sc_SSP5, lt=lt_existing, plot=False, plot_title='SSP5')

# # ------------------------------------------------------------------------------------
# SCENARIO DEFINITIONS
# Scenario 1
# scenario 1 MC:
num_iter = 10
S1_mat_i_list = {}
S1_mat_o_list = {}
S1_mat_s_list = {}
for i in range(num_iter):
    # print(lt_future)

    S1_sio_new = determine_inflow_outflow_new_bldg(scenario='S_timber_high',
                                                   FA_dsm_SSP=FA_dsm_SSP1,
                                                   lt=lt_future,
                                                   plot=False, plot_title='SSP1 ')
    # print(S1_sio_new)
    S1_area_i, S1_area_o, S1_area_s = combine_area_existing_and_new(
        os_existing=os_existing_SSP1,
        sio_new_bldg=S1_sio_new,
        plot=False, plot_title='SSP1, S0')
    S1_mat_i, S1_mat_o, S1_mat_s = calc_inflow_outflow_stock_mats(
        area_inflow_2017_2100=S1_area_i,
        area_outflow_2017_2100=S1_area_o,
        area_stock_2017_2100=S1_area_s,
        materials_intensity_df=materials_intensity_df,
        print_year=0)

    S1_mat_i_list[i] = S1_mat_i
    S1_mat_o_list[i] = S1_mat_o
    S1_mat_s_list[i] = S1_mat_s

S1_mat_i_mean = pd.concat(S1_mat_i_list).groupby(level=1).mean()
S1_mat_i_std = pd.concat(S1_mat_i_list).groupby(level=1).std()
S1_mat_o_mean = pd.concat(S1_mat_o_list).groupby(level=1).mean()
S1_mat_o_std = pd.concat(S1_mat_o_list).groupby(level=1).std()
S1_mat_s_mean = pd.concat(S1_mat_s_list).groupby(level=1).mean()
S1_mat_s_std = pd.concat(S1_mat_s_list).groupby(level=1).std()


# Scenario 2
S2_sio_new = determine_inflow_outflow_new_bldg(scenario='S_timber_high',
                                               FA_dsm_SSP=FA_dsm_SSP1,
                                               lt=lt_future,
                                               plot=False, plot_title='SSP1 ')
S2_area_i, S2_area_o, S2_area_s = combine_area_existing_and_new(
    os_existing=os_existing_SSP1,
    sio_new_bldg=S1_sio_new,
    plot=False, plot_title='SSP1, S_timber_high')
S2_mat_i, S2_mat_o, S2_mat_s = calc_inflow_outflow_stock_mats(
    area_inflow_2017_2100=S2_area_i,
    area_outflow_2017_2100=S2_area_o,
    area_stock_2017_2100=S2_area_s,
    materials_intensity_df=materials_intensity_df,
    print_year=0)

# Scenario 3
S3_sio_new = determine_inflow_outflow_new_bldg(scenario='S_densification_40p_LF_wood',
                                               FA_dsm_SSP=FA_dsm_SSP1,
                                               lt=lt_future,
                                               plot=False, plot_title='SSP1 ')
S3_area_i, S3_area_o, S3_area_s = combine_area_existing_and_new(
    os_existing=os_existing_SSP1,
    sio_new_bldg=S3_sio_new,
    plot=False, plot_title='SSP1, dens-40p')
S3_mat_i, S3_mat_o, S3_mat_s, = calc_inflow_outflow_stock_mats(
    area_inflow_2017_2100=S3_area_i,
    area_outflow_2017_2100=S3_area_o,
    area_stock_2017_2100= S3_area_s,
    materials_intensity_df= materials_intensity_df,
    print_year=0)

# Scenario 4
S4_sio_new = determine_inflow_outflow_new_bldg(scenario='S_0', FA_dsm_SSP=FA_dsm_SSP3,
                                               lt=lt_future,
                                               plot=False, plot_title='SSP3 ')
S4_area_i, S4_area_o, S4_area_s, = combine_area_existing_and_new(
    os_existing=os_existing_SSP3,
    sio_new_bldg=S4_sio_new,
    plot=False, plot_title='SSP3, S0')
S4_mat_i, S4_mat_o, S4_mat_s, = calc_inflow_outflow_stock_mats(
    area_inflow_2017_2100=S4_area_i,
    area_outflow_2017_2100=S4_area_o,
    area_stock_2017_2100= S4_area_s,
    materials_intensity_df= materials_intensity_df,
    print_year=0)

# Scenario 5
S5_sio_new = determine_inflow_outflow_new_bldg(scenario='S_timber_high', FA_dsm_SSP=FA_dsm_SSP3,
                                               lt=lt_future,
                                               plot=False, plot_title='SSP3 ')
S5_area_i, S5_area_o, S5_area_s = combine_area_existing_and_new(
    os_existing = os_existing_SSP3,
    sio_new_bldg=S5_sio_new,
    plot=False, plot_title='SSP3, S_timber_high')
S5_mat_i, S5_mat_o, S5_mat_s = calc_inflow_outflow_stock_mats(
    area_inflow_2017_2100=S5_area_i,
    area_outflow_2017_2100=S5_area_o,
    area_stock_2017_2100= S5_area_s,
    materials_intensity_df=materials_intensity_df,
    print_year=0)


# # ------------------------------------------------------------------------------------
# Function to plot stock/inflow/outflow for different scenarios
def plot_sio_materials(s1_inflow=None, s2_inflow=None, s3_inflow=None, s4_inflow=None, s5_inflow=None,
                       s1_inflow_ci=None, s2_inflow_ci=None, s3_inflow_ci=None, s4_inflow_ci=None, s5_inflow_ci=None,
                       s1_outflow=None, s2_outflow=None, s3_outflow=None, s4_outflow=None, s5_outflow=None,
                       s1_outflow_ci=None, s2_outflow_ci=None, s3_outflow_ci=None, s4_outflow_ci=None, s5_outflow_ci=None,
                       s1_stock=None, s2_stock=None, s3_stock=None, s4_stock=None, s5_stock=None,
                       s1_stock_ci=None, s2_stock_ci=None, s3_stock_ci=None, s4_stock_ci=None, s5_stock_ci=None,
                       legend=['S1', 'S2', 'S3'], plot_error=True):
    ''' Plot the inflow and outflow of each scenario
        Note that stocks are plotted in Gt and inflows/outflows are plotted in Mt/year
    '''

    ## Plot total material inflows each year (Mt/year) by scenario
    fig, axs = plt.subplots(3, 2, figsize=(6, 8))
    if s1_inflow is not None:
        # plot mean
        axs[0, 0].plot(s1_inflow.index, s1_inflow['Sum_steel_inflow'], 'tab:blue')
        axs[0, 1].plot(s1_inflow.index, s1_inflow['Sum_conc_inflow'], 'tab:blue')
        axs[1, 0].plot(s1_inflow.index, s1_inflow['Sum_engwood_inflow'], 'tab:blue')
        axs[1, 1].plot(s1_inflow.index, s1_inflow['Sum_dimlum_inflow'], 'tab:blue')
        axs[2, 0].plot(s1_inflow.index, s1_inflow['Sum_masonry_inflow'], 'tab:blue')
        # plot error bars
        if plot_error==True:
            axs[0, 0].fill_between(s1_inflow.index,
                                   (s1_inflow['Sum_steel_inflow'] - s1_inflow_ci['Sum_steel_inflow']),
                                   (s1_inflow['Sum_steel_inflow'] + s1_inflow_ci['Sum_steel_inflow']),
                                   color='blue', alpha=.1)
            axs[0, 1].fill_between(s1_inflow.index,
                                   (s1_inflow['Sum_conc_inflow'] - s1_inflow_ci['Sum_conc_inflow']),
                                   (s1_inflow['Sum_conc_inflow'] + s1_inflow_ci['Sum_conc_inflow']),
                                   color='blue', alpha=.1)
            axs[1, 0].fill_between(s1_inflow.index,
                                   (s1_inflow['Sum_engwood_inflow'] - s1_inflow_ci['Sum_engwood_inflow']),
                                   (s1_inflow['Sum_engwood_inflow'] + s1_inflow_ci['Sum_engwood_inflow']),
                                   color='blue', alpha=.1)
            axs[1, 1].fill_between(s1_inflow.index,
                                   (s1_inflow['Sum_dimlum_inflow'] - s1_inflow_ci['Sum_dimlum_inflow']),
                                   (s1_inflow['Sum_dimlum_inflow'] + s1_inflow_ci['Sum_dimlum_inflow']),
                                   color='blue', alpha=.1)
            axs[2, 0].fill_between(s1_inflow.index,
                                   (s1_inflow['Sum_masonry_inflow'] - s1_inflow_ci['Sum_masonry_inflow']),
                                   (s1_inflow['Sum_masonry_inflow'] + s1_inflow_ci['Sum_masonry_inflow']),
                                   color='blue', alpha=.1)

        axs[0, 0].set_title('Steel')
        axs[0, 1].set_title('Concrete')
        axs[1, 0].set_title('Eng. Wood')
        axs[1, 1].set_title('Dim. Lumber')
        axs[2, 0].set_title('Masonry')
        if s2_inflow is not None:
            axs[0, 0].plot(s2_inflow.index, s2_inflow['Sum_steel_inflow'], 'tab:red')
            axs[0, 1].plot(s2_inflow.index, s2_inflow['Sum_conc_inflow'], 'tab:red')
            axs[1, 0].plot(s2_inflow.index, s2_inflow['Sum_engwood_inflow'], 'tab:red')
            axs[1, 1].plot(s2_inflow.index, s2_inflow['Sum_dimlum_inflow'], 'tab:red')
            axs[2, 0].plot(s2_inflow.index, s2_inflow['Sum_masonry_inflow'], 'tab:red')
            # plot error bars
            if plot_error == True:
                axs[0, 0].fill_between(s2_inflow.index,
                                       (s2_inflow['Sum_steel_inflow'] - s2_inflow_ci['Sum_steel_inflow']),
                                       (s2_inflow['Sum_steel_inflow'] + s2_inflow_ci['Sum_steel_inflow']),
                                       color='red', alpha=.1)
                axs[0, 1].fill_between(s2_inflow.index,
                                       (s2_inflow['Sum_conc_inflow'] - s2_inflow_ci['Sum_conc_inflow']),
                                       (s2_inflow['Sum_conc_inflow'] + s2_inflow_ci['Sum_conc_inflow']),
                                       color='red', alpha=.1)
                axs[1, 0].fill_between(s2_inflow.index,
                                       (s2_inflow['Sum_engwood_inflow'] - s2_inflow_ci['Sum_engwood_inflow']),
                                       (s2_inflow['Sum_engwood_inflow'] + s2_inflow_ci['Sum_engwood_inflow']),
                                       color='red', alpha=.1)
                axs[1, 1].fill_between(s2_inflow.index,
                                       (s2_inflow['Sum_dimlum_inflow'] - s2_inflow['Sum_dimlum_inflow']),
                                       (s2_inflow['Sum_dimlum_inflow'] + s2_inflow['Sum_dimlum_inflow']),
                                       color='red', alpha=.1)
                axs[2, 0].fill_between(s2_inflow.index,
                                       (s2_inflow['Sum_masonry_inflow'] - s2_inflow['Sum_masonry_inflow']),
                                       (s2_inflow['Sum_masonry_inflow'] + s2_inflow['Sum_masonry_inflow']),
                                       color='red', alpha=.1)

            if s3_inflow is not None:
                axs[0, 0].plot(s3_inflow.index, s3_inflow['Sum_steel_inflow'], 'tab:green')
                axs[0, 1].plot(s3_inflow.index, s3_inflow['Sum_conc_inflow'], 'tab:green')
                axs[1, 0].plot(s3_inflow.index, s3_inflow['Sum_engwood_inflow'], 'tab:green')
                axs[1, 1].plot(s3_inflow.index, s3_inflow['Sum_dimlum_inflow'], 'tab:green')
                axs[2, 0].plot(s3_inflow.index, s3_inflow['Sum_masonry_inflow'], 'tab:green')
                # plot error bars
                if plot_error == True:
                    axs[0, 0].fill_between(s3_inflow.index,
                                           (s3_inflow['Sum_steel_inflow'] - s3_inflow_ci['Sum_steel_inflow']),
                                           (s3_inflow['Sum_steel_inflow'] + s3_inflow_ci['Sum_steel_inflow']),
                                           color='green', alpha=.1)
                    axs[0, 1].fill_between(s3_inflow.index,
                                           (s3_inflow['Sum_conc_inflow'] - s3_inflow_ci['Sum_conc_inflow']),
                                           (s3_inflow['Sum_conc_inflow'] + s3_inflow_ci['Sum_conc_inflow']),
                                           color='green', alpha=.1)
                    axs[1, 0].fill_between(s3_inflow.index,
                                           (s3_inflow['Sum_engwood_inflow'] - s3_inflow_ci['Sum_engwood_inflow']),
                                           (s3_inflow['Sum_engwood_inflow'] + s3_inflow_ci['Sum_engwood_inflow']),
                                           color='green', alpha=.1)
                    axs[1, 1].fill_between(s3_inflow.index,
                                           (s3_inflow['Sum_dimlum_inflow'] - s3_inflow['Sum_dimlum_inflow']),
                                           (s3_inflow['Sum_dimlum_inflow'] + s3_inflow['Sum_dimlum_inflow']),
                                           color='green', alpha=.1)
                    axs[2, 0].fill_between(s3_inflow.index,
                                           (s3_inflow['Sum_masonry_inflow'] - s3_inflow['Sum_masonry_inflow']),
                                           (s3_inflow['Sum_masonry_inflow'] + s3_inflow['Sum_masonry_inflow']),
                                           color='green', alpha=.1)
                if s4_inflow is not None:
                    axs[0, 0].plot(s4_inflow.index, s4_inflow['Sum_steel_inflow'], 'tab:purple')
                    axs[0, 1].plot(s4_inflow.index, s4_inflow['Sum_conc_inflow'], 'tab:purple')
                    axs[1, 0].plot(s4_inflow.index, s4_inflow['Sum_engwood_inflow'], 'tab:purple')
                    axs[1, 1].plot(s4_inflow.index, s4_inflow['Sum_dimlum_inflow'], 'tab:purple')
                    axs[2, 0].plot(s4_inflow.index, s4_inflow['Sum_masonry_inflow'], 'tab:purple')
                    if plot_error == True:
                        axs[0, 0].fill_between(s4_inflow.index,
                                               (s4_inflow['Sum_steel_inflow'] - s4_inflow_ci['Sum_steel_inflow']),
                                               (s4_inflow['Sum_steel_inflow'] + s4_inflow_ci['Sum_steel_inflow']),
                                               color='purple', alpha=.1)
                        axs[0, 1].fill_between(s4_inflow.index,
                                               (s4_inflow['Sum_conc_inflow'] - s4_inflow_ci['Sum_conc_inflow']),
                                               (s4_inflow['Sum_conc_inflow'] + s4_inflow_ci['Sum_conc_inflow']),
                                               color='purple', alpha=.1)
                        axs[1, 0].fill_between(s4_inflow.index,
                                               (s4_inflow['Sum_engwood_inflow'] - s4_inflow_ci['Sum_engwood_inflow']),
                                               (s4_inflow['Sum_engwood_inflow'] + s4_inflow_ci['Sum_engwood_inflow']),
                                               color='purple', alpha=.1)
                        axs[1, 1].fill_between(s4_inflow.index,
                                               (s4_inflow['Sum_dimlum_inflow'] - s4_inflow['Sum_dimlum_inflow']),
                                               (s4_inflow['Sum_dimlum_inflow'] + s4_inflow['Sum_dimlum_inflow']),
                                               color='purple', alpha=.1)
                        axs[2, 0].fill_between(s4_inflow.index,
                                               (s4_inflow['Sum_masonry_inflow'] - s4_inflow['Sum_masonry_inflow']),
                                               (s4_inflow['Sum_masonry_inflow'] + s4_inflow['Sum_masonry_inflow']),
                                               color='purple', alpha=.1)
                    if s5_inflow is not None:
                        axs[0, 0].plot(s5_inflow.index, s5_inflow['Sum_steel_inflow'], 'tab:orange')
                        axs[0, 1].plot(s5_inflow.index, s5_inflow['Sum_conc_inflow'], 'tab:orange')
                        axs[1, 0].plot(s5_inflow.index, s5_inflow['Sum_engwood_inflow'], 'tab:orange')
                        axs[1, 1].plot(s5_inflow.index, s5_inflow['Sum_dimlum_inflow'], 'tab:orange')
                        axs[2, 0].plot(s5_inflow.index, s5_inflow['Sum_masonry_inflow'], 'tab:orange')
                        if plot_error == True:
                            axs[0, 0].fill_between(s5_inflow.index,
                                                   (s5_inflow['Sum_steel_inflow'] - s5_inflow_ci['Sum_steel_inflow']),
                                                   (s5_inflow['Sum_steel_inflow'] + s5_inflow_ci['Sum_steel_inflow']),
                                                   color='orange', alpha=.1)
                            axs[0, 1].fill_between(s5_inflow.index,
                                                   (s5_inflow['Sum_conc_inflow'] - s5_inflow_ci['Sum_conc_inflow']),
                                                   (s5_inflow['Sum_conc_inflow'] + s5_inflow_ci['Sum_conc_inflow']),
                                                   color='orange', alpha=.1)
                            axs[1, 0].fill_between(s5_inflow.index,
                                                   (s5_inflow['Sum_engwood_inflow'] - s5_inflow_ci['Sum_engwood_inflow']),
                                                   (s5_inflow['Sum_engwood_inflow'] + s5_inflow_ci['Sum_engwood_inflow']),
                                                   color='orange', alpha=.1)
                            axs[1, 1].fill_between(s5_inflow.index,
                                                   (s5_inflow['Sum_dimlum_inflow'] - s5_inflow_ci['Sum_dimlum_inflow']),
                                                   (s5_inflow['Sum_dimlum_inflow'] + s5_inflow_ci['Sum_dimlum_inflow']),
                                                   color='orange', alpha=.1)
                            axs[2, 0].fill_between(s5_inflow.index,
                                                   (s5_inflow['Sum_masonry_inflow'] - s5_inflow_ci['Sum_masonry_inflow']),
                                                   (s5_inflow['Sum_masonry_inflow'] + s5_inflow_ci['Sum_masonry_inflow']),
                                                   color='orange', alpha=.1)

    # add legend in place of 6th plot
    axs[2, 1].axis('off')
    fig.legend(legend, title='Material Inflow Scenarios', loc='lower right', bbox_to_anchor=(0.9, 0.1), fancybox=True)
    # add ylabels
    for ax in axs.flat:
        ax.set(ylabel='Mt/year')
    fig.show()

    ## Plot total material outflows each year (Mt/year) by scenario
    fig, axs = plt.subplots(3, 2, figsize=(6, 8))
    if s1_outflow is not None:
        axs[0, 0].plot(s1_outflow.index, s1_outflow['Sum_steel_outflow'], 'tab:blue')
        axs[0, 1].plot(s1_outflow.index, s1_outflow['Sum_conc_outflow'], 'tab:blue')
        axs[1, 0].plot(s1_outflow.index, s1_outflow['Sum_engwood_outflow'], 'tab:blue')
        axs[1, 1].plot(s1_outflow.index, s1_outflow['Sum_dimlum_outflow'], 'tab:blue')
        axs[2, 0].plot(s1_outflow.index, s1_outflow['Sum_masonry_outflow'], 'tab:blue')
        if plot_error==True:
            axs[0, 0].fill_between(s1_outflow.index,
                                   (s1_outflow['Sum_steel_outflow'] - s1_outflow_ci['Sum_steel_outflow']),
                                   (s1_outflow['Sum_steel_outflow'] + s1_outflow_ci['Sum_steel_outflow']),
                                   color='blue', alpha=.1)
            axs[0, 1].fill_between(s1_outflow.index,
                                   (s1_outflow['Sum_conc_outflow'] - s1_outflow_ci['Sum_conc_outflow']),
                                   (s1_outflow['Sum_conc_outflow'] + s1_outflow_ci['Sum_conc_outflow']),
                                   color='blue', alpha=.1)
            axs[1, 0].fill_between(s1_outflow.index,
                                   (s1_outflow['Sum_engwood_outflow'] - s1_outflow_ci['Sum_engwood_outflow']),
                                   (s1_outflow['Sum_engwood_outflow'] + s1_outflow_ci['Sum_engwood_outflow']),
                                   color='blue', alpha=.1)
            axs[1, 1].fill_between(s1_outflow.index,
                                   (s1_outflow['Sum_dimlum_outflow'] - s1_outflow_ci['Sum_dimlum_outflow']),
                                   (s1_outflow['Sum_dimlum_outflow'] + s1_outflow_ci['Sum_dimlum_outflow']),
                                   color='blue', alpha=.1)
            axs[2, 0].fill_between(s1_outflow.index,
                                   (s1_outflow['Sum_masonry_outflow'] - s1_outflow_ci['Sum_masonry_outflow']),
                                   (s1_outflow['Sum_masonry_outflow'] + s1_outflow_ci['Sum_masonry_outflow']),
                                   color='blue', alpha=.1)

        axs[0, 0].set_title('Steel')
        axs[0, 1].set_title('Concrete')
        axs[1, 0].set_title('Eng. Wood')
        axs[1, 1].set_title('Dim. Lumber')
        axs[2, 0].set_title('Masonry')
        if s2_outflow is not None:
            axs[0, 0].plot(s2_outflow.index, s2_outflow['Sum_steel_outflow'], 'tab:red')
            axs[0, 1].plot(s2_outflow.index, s2_outflow['Sum_conc_outflow'], 'tab:red')
            axs[1, 0].plot(s2_outflow.index, s2_outflow['Sum_engwood_outflow'], 'tab:red')
            axs[1, 1].plot(s2_outflow.index, s2_outflow['Sum_dimlum_outflow'], 'tab:red')
            axs[2, 0].plot(s2_outflow.index, s2_outflow['Sum_masonry_outflow'], 'tab:red')
            if plot_error == True:
                axs[0, 0].fill_between(s2_outflow.index,
                                       (s2_outflow['Sum_steel_outflow'] - s2_outflow_ci['Sum_steel_outflow']),
                                       (s2_outflow['Sum_steel_outflow'] + s2_outflow_ci['Sum_steel_outflow']),
                                       color='red', alpha=.1)
                axs[0, 1].fill_between(s2_outflow.index,
                                       (s2_outflow['Sum_conc_outflow'] - s2_outflow_ci['Sum_conc_outflow']),
                                       (s2_outflow['Sum_conc_outflow'] + s2_outflow_ci['Sum_conc_outflow']),
                                       color='red', alpha=.1)
                axs[1, 0].fill_between(s2_outflow.index,
                                       (s2_outflow['Sum_engwood_outflow'] - s2_outflow_ci['Sum_engwood_outflow']),
                                       (s2_outflow['Sum_engwood_outflow'] + s2_outflow_ci['Sum_engwood_outflow']),
                                       color='red', alpha=.1)
                axs[1, 1].fill_between(s2_outflow.index,
                                       (s2_outflow['Sum_dimlum_outflow'] - s2_outflow_ci['Sum_dimlum_outflow']),
                                       (s2_outflow['Sum_dimlum_outflow'] + s2_outflow_ci['Sum_dimlum_outflow']),
                                       color='red', alpha=.1)
                axs[2, 0].fill_between(s2_outflow.index,
                                       (s2_outflow['Sum_masonry_outflow'] - s2_outflow_ci['Sum_masonry_outflow']),
                                       (s2_outflow['Sum_masonry_outflow'] + s2_outflow_ci['Sum_masonry_outflow']),
                                       color='red', alpha=.1)

            if s3_outflow is not None:
                axs[0, 0].plot(s3_outflow.index, s3_outflow['Sum_steel_outflow'], 'tab:green')
                axs[0, 1].plot(s3_outflow.index, s3_outflow['Sum_conc_outflow'], 'tab:green')
                axs[1, 0].plot(s3_outflow.index, s3_outflow['Sum_engwood_outflow'], 'tab:green')
                axs[1, 1].plot(s3_outflow.index, s3_outflow['Sum_dimlum_outflow'], 'tab:green')
                axs[2, 0].plot(s3_outflow.index, s3_outflow['Sum_masonry_outflow'], 'tab:green')
                if plot_error == True:
                    axs[0, 0].fill_between(s3_outflow.index,
                                           (s3_outflow['Sum_steel_outflow'] - s3_outflow_ci['Sum_steel_outflow']),
                                           (s3_outflow['Sum_steel_outflow'] + s3_outflow_ci['Sum_steel_outflow']),
                                           color='green', alpha=.1)
                    axs[0, 1].fill_between(s3_outflow.index,
                                           (s3_outflow['Sum_conc_outflow'] - s3_outflow_ci['Sum_conc_outflow']),
                                           (s3_outflow['Sum_conc_outflow'] + s3_outflow_ci['Sum_conc_outflow']),
                                           color='green', alpha=.1)
                    axs[1, 0].fill_between(s3_outflow.index,
                                           (s3_outflow['Sum_engwood_outflow'] - s3_outflow_ci['Sum_engwood_outflow']),
                                           (s3_outflow['Sum_engwood_outflow'] + s3_outflow_ci['Sum_engwood_outflow']),
                                           color='green', alpha=.1)
                    axs[1, 1].fill_between(s3_outflow.index,
                                           (s3_outflow['Sum_dimlum_outflow'] - s3_outflow_ci['Sum_dimlum_outflow']),
                                           (s3_outflow['Sum_dimlum_outflow'] + s3_outflow_ci['Sum_dimlum_outflow']),
                                           color='green', alpha=.1)
                    axs[2, 0].fill_between(s3_outflow.index,
                                           (s3_outflow['Sum_masonry_outflow'] - s3_outflow_ci['Sum_masonry_outflow']),
                                           (s3_outflow['Sum_masonry_outflow'] + s3_outflow_ci['Sum_masonry_outflow']),
                                           color='green', alpha=.1)

                if s4_outflow is not None:
                    axs[0, 0].plot(s4_outflow.index, s4_outflow['Sum_steel_outflow'], 'tab:purple')
                    axs[0, 1].plot(s4_outflow.index, s4_outflow['Sum_conc_outflow'], 'tab:purple')
                    axs[1, 0].plot(s4_outflow.index, s4_outflow['Sum_engwood_outflow'], 'tab:purple')
                    axs[1, 1].plot(s4_outflow.index, s4_outflow['Sum_dimlum_outflow'], 'tab:purple')
                    axs[2, 0].plot(s4_outflow.index, s4_outflow['Sum_masonry_outflow'], 'tab:purple')
                    if plot_error == True:
                        axs[0, 0].fill_between(s4_outflow.index,
                                               (s4_outflow['Sum_steel_outflow'] - s4_outflow_ci['Sum_steel_outflow']),
                                               (s4_outflow['Sum_steel_outflow'] + s4_outflow_ci['Sum_steel_outflow']),
                                               color='purple', alpha=.1)
                        axs[0, 1].fill_between(s4_outflow.index,
                                               (s4_outflow['Sum_conc_outflow'] - s4_outflow_ci['Sum_conc_outflow']),
                                               (s4_outflow['Sum_conc_outflow'] + s4_outflow_ci['Sum_conc_outflow']),
                                               color='purple', alpha=.1)
                        axs[1, 0].fill_between(s4_outflow.index,
                                               (s4_outflow['Sum_engwood_outflow'] - s4_outflow_ci['Sum_engwood_outflow']),
                                               (s4_outflow['Sum_engwood_outflow'] + s4_outflow_ci['Sum_engwood_outflow']),
                                               color='purple', alpha=.1)
                        axs[1, 1].fill_between(s4_outflow.index,
                                               (s4_outflow['Sum_dimlum_outflow'] - s4_outflow_ci['Sum_dimlum_outflow']),
                                               (s4_outflow['Sum_dimlum_outflow'] + s4_outflow_ci['Sum_dimlum_outflow']),
                                               color='purple', alpha=.1)
                        axs[2, 0].fill_between(s4_outflow.index,
                                               (s4_outflow['Sum_masonry_outflow'] - s4_outflow_ci['Sum_masonry_outflow']),
                                               (s4_outflow['Sum_masonry_outflow'] + s4_outflow_ci['Sum_masonry_outflow']),
                                               color='purple', alpha=.1)

                    if s5_outflow is not None:
                        axs[0, 0].plot(s5_outflow.index, s5_outflow['Sum_steel_outflow'], 'tab:orange')
                        axs[0, 1].plot(s5_outflow.index, s5_outflow['Sum_conc_outflow'], 'tab:orange')
                        axs[1, 0].plot(s5_outflow.index, s5_outflow['Sum_engwood_outflow'], 'tab:orange')
                        axs[1, 1].plot(s5_outflow.index, s5_outflow['Sum_dimlum_outflow'], 'tab:orange')
                        axs[2, 0].plot(s5_outflow.index, s5_outflow['Sum_masonry_outflow'], 'tab:orange')
                        if plot_error == True:
                            axs[0, 0].fill_between(s5_outflow.index,
                                                   (s5_outflow['Sum_steel_outflow'] - s5_outflow_ci['Sum_steel_outflow']),
                                                   (s5_outflow['Sum_steel_outflow'] + s5_outflow_ci['Sum_steel_outflow']),
                                                   color='orange', alpha=.1)
                            axs[0, 1].fill_between(s5_outflow.index,
                                                   (s5_outflow['Sum_conc_outflow'] - s5_outflow_ci['Sum_conc_outflow']),
                                                   (s5_outflow['Sum_conc_outflow'] + s5_outflow_ci['Sum_conc_outflow']),
                                                   color='orange', alpha=.1)
                            axs[1, 0].fill_between(s5_outflow.index,
                                                   (s5_outflow['Sum_engwood_outflow'] - s5_outflow_ci['Sum_engwood_outflow']),
                                                   (s5_outflow['Sum_engwood_outflow'] + s5_outflow_ci['Sum_engwood_outflow']),
                                                   color='orange', alpha=.1)
                            axs[1, 1].fill_between(s5_outflow.index,
                                                   (s5_outflow['Sum_dimlum_outflow'] - s5_outflow_ci['Sum_dimlum_outflow']),
                                                   (s5_outflow['Sum_dimlum_outflow'] + s5_outflow_ci['Sum_dimlum_outflow']),
                                                   color='orange', alpha=.1)
                            axs[2, 0].fill_between(s5_outflow.index,
                                                   (s5_outflow['Sum_masonry_outflow'] - s5_outflow_ci['Sum_masonry_outflow']),
                                                   (s5_outflow['Sum_masonry_outflow'] + s5_outflow_ci['Sum_masonry_outflow']),
                                                   color='orange', alpha=.1)

    # add legend in place of 6th plot
    axs[2, 1].axis('off')
    fig.legend(legend, title='Material Outflows by Scenarios', loc='lower right', bbox_to_anchor=(0.9, 0.1), fancybox=True)
    # add ylabels
    for ax in axs.flat:
        ax.set(ylabel='Mt/year')
    fig.show()

    ## Plot total material stocks each year (Gt) by scenario
    fig, axs = plt.subplots(3, 2, figsize=(6, 8))
    if s1_stock is not None:
        axs[0, 0].plot(s1_stock.index, s1_stock['Sum_steel_stock'], 'tab:blue')
        axs[0, 1].plot(s1_stock.index, s1_stock['Sum_conc_stock'], 'tab:blue')
        axs[1, 0].plot(s1_stock.index, s1_stock['Sum_engwood_stock'], 'tab:blue')
        axs[1, 1].plot(s1_stock.index, s1_stock['Sum_dimlum_stock'], 'tab:blue')
        axs[2, 0].plot(s1_stock.index, s1_stock['Sum_masonry_stock'], 'tab:blue')
        if plot_error==True:
            axs[0, 0].fill_between(s1_stock.index,
                                   (s1_stock['Sum_steel_stock'] - s1_stock_ci['Sum_steel_stock']),
                                   (s1_stock['Sum_steel_stock'] + s1_stock_ci['Sum_steel_stock']),
                                   color='blue', alpha=.1)
            axs[0, 1].fill_between(s1_stock.index,
                                   (s1_stock['Sum_conc_stock'] - s1_stock_ci['Sum_conc_stock']),
                                   (s1_stock['Sum_conc_stock'] + s1_stock_ci['Sum_conc_stock']),
                                   color='blue', alpha=.1)
            axs[1, 0].fill_between(s1_stock.index,
                                   (s1_stock['Sum_engwood_stock'] - s1_stock_ci['Sum_engwood_stock']),
                                   (s1_stock['Sum_engwood_stock'] + s1_stock_ci['Sum_engwood_stock']),
                                   color='blue', alpha=.1)
            axs[1, 1].fill_between(s1_stock.index,
                                   (s1_stock['Sum_dimlum_stock'] - s1_stock_ci['Sum_dimlum_stock']),
                                   (s1_stock['Sum_dimlum_stock'] + s1_stock_ci['Sum_dimlum_stock']),
                                   color='blue', alpha=.1)
            axs[2, 0].fill_between(s1_stock.index,
                                   (s1_stock['Sum_masonry_stock'] - s1_stock_ci['Sum_masonry_stock']),
                                   (s1_stock['Sum_masonry_stock'] + s1_stock_ci['Sum_masonry_stock']),
                                   color='blue', alpha=.1)

        axs[0, 0].set_title('Steel')
        axs[0, 1].set_title('Concrete')
        axs[1, 0].set_title('Eng. Wood')
        axs[1, 1].set_title('Dim. Lumber')
        axs[2, 0].set_title('Masonry')
        if s2_stock is not None:
            axs[0, 0].plot(s2_stock.index, s2_stock['Sum_steel_stock'], 'tab:red')
            axs[0, 1].plot(s2_stock.index, s2_stock['Sum_conc_stock'], 'tab:red')
            axs[1, 0].plot(s2_stock.index, s2_stock['Sum_engwood_stock'], 'tab:red')
            axs[1, 1].plot(s2_stock.index, s2_stock['Sum_dimlum_stock'], 'tab:red')
            axs[2, 0].plot(s2_stock.index, s2_stock['Sum_masonry_stock'], 'tab:red')
            if plot_error == True:
                axs[0, 0].fill_between(s2_stock.index,
                                       (s2_stock['Sum_steel_stock'] - s2_stock_ci['Sum_steel_stock']),
                                       (s2_stock['Sum_steel_stock'] + s2_stock_ci['Sum_steel_stock']),
                                       color='red', alpha=.1)
                axs[0, 1].fill_between(s2_stock.index,
                                       (s2_stock['Sum_conc_stock'] - s2_stock_ci['Sum_conc_stock']),
                                       (s2_stock['Sum_conc_stock'] + s2_stock_ci['Sum_conc_stock']),
                                       color='red', alpha=.1)
                axs[1, 0].fill_between(s2_stock.index,
                                       (s2_stock['Sum_engwood_stock'] - s2_stock_ci['Sum_engwood_stock']),
                                       (s2_stock['Sum_engwood_stock'] + s2_stock_ci['Sum_engwood_stock']),
                                       color='red', alpha=.1)
                axs[1, 1].fill_between(s2_stock.index,
                                       (s2_stock['Sum_dimlum_stock'] - s2_stock_ci['Sum_dimlum_stock']),
                                       (s2_stock['Sum_dimlum_stock'] + s2_stock_ci['Sum_dimlum_stock']),
                                       color='red', alpha=.1)
                axs[2, 0].fill_between(s2_stock.index,
                                       (s2_stock['Sum_masonry_stock'] - s2_stock_ci['Sum_masonry_stock']),
                                       (s2_stock['Sum_masonry_stock'] + s2_stock_ci['Sum_masonry_stock']),
                                       color='red', alpha=.1)

            if s3_stock is not None:
                axs[0, 0].plot(s3_stock.index, s3_stock['Sum_steel_stock'], 'tab:green')
                axs[0, 1].plot(s3_stock.index, s3_stock['Sum_conc_stock'], 'tab:green')
                axs[1, 0].plot(s3_stock.index, s3_stock['Sum_engwood_stock'], 'tab:green')
                axs[1, 1].plot(s3_stock.index, s3_stock['Sum_dimlum_stock'], 'tab:green')
                axs[2, 0].plot(s3_stock.index, s3_stock['Sum_masonry_stock'], 'tab:green')
                if plot_error == True:
                    axs[0, 0].fill_between(s3_stock.index,
                                           (s3_stock['Sum_steel_stock'] - s3_stock_ci['Sum_steel_stock']),
                                           (s3_stock['Sum_steel_stock'] + s3_stock_ci['Sum_steel_stock']),
                                           color='green', alpha=.1)
                    axs[0, 1].fill_between(s3_stock.index,
                                           (s3_stock['Sum_conc_stock'] - s3_stock_ci['Sum_conc_stock']),
                                           (s3_stock['Sum_conc_stock'] + s3_stock_ci['Sum_conc_stock']),
                                           color='green', alpha=.1)
                    axs[1, 0].fill_between(s3_stock.index,
                                           (s3_stock['Sum_engwood_stock'] - s3_stock_ci['Sum_engwood_stock']),
                                           (s3_stock['Sum_engwood_stock'] + s3_stock_ci['Sum_engwood_stock']),
                                           color='green', alpha=.1)
                    axs[1, 1].fill_between(s3_stock.index,
                                           (s3_stock['Sum_dimlum_stock'] - s3_stock_ci['Sum_dimlum_stock']),
                                           (s3_stock['Sum_dimlum_stock'] + s3_stock_ci['Sum_dimlum_stock']),
                                           color='green', alpha=.1)
                    axs[2, 0].fill_between(s3_stock.index,
                                           (s3_stock['Sum_masonry_stock'] - s3_stock_ci['Sum_masonry_stock']),
                                           (s3_stock['Sum_masonry_stock'] + s3_stock_ci['Sum_masonry_stock']),
                                           color='green', alpha=.1)

                if s4_stock is not None:
                    axs[0, 0].plot(s4_stock.index, s4_stock['Sum_steel_stock'], 'tab:purple')
                    axs[0, 1].plot(s4_stock.index, s4_stock['Sum_conc_stock'], 'tab:purple')
                    axs[1, 0].plot(s4_stock.index, s4_stock['Sum_engwood_stock'], 'tab:purple')
                    axs[1, 1].plot(s4_stock.index, s4_stock['Sum_dimlum_stock'], 'tab:purple')
                    axs[2, 0].plot(s4_stock.index, s4_stock['Sum_masonry_stock'], 'tab:purple')
                    if plot_error == True:
                        axs[0, 0].fill_between(s4_stock.index,
                                               (s4_stock['Sum_steel_stock'] - s4_stock_ci['Sum_steel_stock']),
                                               (s4_stock['Sum_steel_stock'] + s4_stock_ci['Sum_steel_stock']),
                                               color='purple', alpha=.1)
                        axs[0, 1].fill_between(s4_stock.index,
                                               (s4_stock['Sum_conc_stock'] - s4_stock_ci['Sum_conc_stock']),
                                               (s4_stock['Sum_conc_stock'] + s4_stock_ci['Sum_conc_stock']),
                                               color='purple', alpha=.1)
                        axs[1, 0].fill_between(s4_stock.index,
                                               (s4_stock['Sum_engwood_stock'] - s4_stock_ci['Sum_engwood_stock']),
                                               (s4_stock['Sum_engwood_stock'] + s4_stock_ci['Sum_engwood_stock']),
                                               color='purple', alpha=.1)
                        axs[1, 1].fill_between(s4_stock.index,
                                               (s4_stock['Sum_dimlum_stock'] - s4_stock_ci['Sum_dimlum_stock']),
                                               (s4_stock['Sum_dimlum_stock'] + s4_stock_ci['Sum_dimlum_stock']),
                                               color='purple', alpha=.1)
                        axs[2, 0].fill_between(s4_stock.index,
                                               (s4_stock['Sum_masonry_stock'] - s4_stock_ci['Sum_masonry_stock']),
                                               (s4_stock['Sum_masonry_stock'] + s4_stock_ci['Sum_masonry_stock']),
                                               color='purple', alpha=.1)

                    if s5_stock is not None:
                        axs[0, 0].plot(s5_stock.index, s5_stock['Sum_steel_stock'], 'tab:orange')
                        axs[0, 1].plot(s5_stock.index, s5_stock['Sum_conc_stock'], 'tab:orange')
                        axs[1, 0].plot(s5_stock.index, s5_stock['Sum_engwood_stock'], 'tab:orange')
                        axs[1, 1].plot(s5_stock.index, s5_stock['Sum_dimlum_stock'], 'tab:orange')
                        axs[2, 0].plot(s5_stock.index, s5_stock['Sum_masonry_stock'], 'tab:orange')
                        if plot_error == True:
                            axs[0, 0].fill_between(s5_stock.index,
                                                   (s5_stock['Sum_steel_stock'] - s5_stock_ci['Sum_steel_stock']),
                                                   (s5_stock['Sum_steel_stock'] + s5_stock_ci['Sum_steel_stock']),
                                                   color='orange', alpha=.1)
                            axs[0, 1].fill_between(s5_stock.index,
                                                   (s5_stock['Sum_conc_stock'] - s5_stock_ci['Sum_conc_stock']),
                                                   (s5_stock['Sum_conc_stock'] + s5_stock_ci['Sum_conc_stock']),
                                                   color='orange', alpha=.1)
                            axs[1, 0].fill_between(s5_stock.index,
                                                   (s5_stock['Sum_engwood_stock'] - s5_stock_ci['Sum_engwood_stock']),
                                                   (s5_stock['Sum_engwood_stock'] + s5_stock_ci['Sum_engwood_stock']),
                                                   color='orange', alpha=.1)
                            axs[1, 1].fill_between(s5_stock.index,
                                                   (s5_stock['Sum_dimlum_stock'] - s5_stock_ci['Sum_dimlum_stock']),
                                                   (s5_stock['Sum_dimlum_stock'] + s5_stock_ci['Sum_dimlum_stock']),
                                                   color='orange', alpha=.1)
                            axs[2, 0].fill_between(s5_stock.index,
                                                   (s5_stock['Sum_masonry_stock'] - s5_stock_ci['Sum_masonry_stock']),
                                                   (s5_stock['Sum_masonry_stock'] + s5_stock_ci['Sum_masonry_stock']),
                                                   color='orange', alpha=.1)

    # add legend in place of 6th plot
    axs[2, 1].axis('off')
    fig.legend(legend, title='Material Stocks by Scenario', loc='lower right', bbox_to_anchor=(0.9, 0.1), fancybox=True)
    # add ylabels
    for ax in axs.flat:
        ax.set(ylabel='Gt')
    fig.show()


# Plot scenarios against one another for stock/inflow/outflow (up to 5 scenarios)
# plot_sio_materials(s1_inflow=S1_mat_i_mean, s1_outflow=S1_mat_o_mean, s1_stock=S1_mat_s_mean/1000,
#                    s1_inflow_ci=S1_mat_i_std, s1_outflow_ci=S1_mat_o_std, s1_stock_ci=S1_mat_o_std/1000,
#                    s2_inflow= S2_mat_i, s2_outflow= S2_mat_o, s2_stock=S2_mat_s/1000,
#                    s2_inflow_ci=S2_mat_i_std, s2_outflow_ci=S2_mat_o_std, s2_stock_ci=S2_mat_o_std/1000,
#                    s3_inflow=S3_mat_i, s3_outflow=S3_mat_o, s3_stock=S3_mat_s/1000,
#                    s3_inflow_ci=S3_mat_i_std, s3_outflow_ci=S3_mat_o_std, s3_stock_ci=S3_mat_o_std/1000,
#                    s4_inflow=S4_mat_i, s4_outflow=S4_mat_o, s4_stock=S4_mat_s/1000,
#                    s4_inflow_ci=S4_mat_i_std, s4_outflow_ci=S4_mat_o_std, s4_stock_ci=S4_mat_o_std/1000,
#                    s5_inflow=S5_mat_i, s5_outflow=S5_mat_o, s5_stock=S5_mat_s/1000,
#                    s5_inflow_ci=S5_mat_i_std, s5_outflow_ci=S5_mat_o_std, s5_stock_ci=S5_mat_o_std/1000,
#                    legend=['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4', 'Scenario 5'])

plot_sio_materials(s1_inflow=S1_mat_i_mean, s1_outflow=S1_mat_o_mean, s1_stock=S1_mat_s_mean/1000,
                   s1_inflow_ci=S1_mat_i_std, s1_outflow_ci=S1_mat_o_std, s1_stock_ci=S1_mat_s_std/1000,
                   legend=['Scenario 1'])




# Check to see if how much the inflow, outflow, and stocks differ from the original stock-driven model for SSP1
check_stocks = False
if check_stocks == True:
    check_df_stock = pd.DataFrame({
        'existing_stock': os_existing_SSP1.loc[2017:]['total_stock'],
        'new_stock': S1_sio_new.loc[2017:]['total_stock'],
        'sum_stock': os_existing_SSP1.loc[2017:]['total_stock'] + S1_sio_new.loc[2017:]['total_stock'],
        'correct_stock': FA_dsm_SSP1.loc[2017:]['stock_total'],
        'percent_difference': (FA_dsm_SSP1.loc[2017:]['stock_total'] - (
                    os_existing_SSP1.loc[2017:]['total_stock'] + S1_sio_new[2017:]['total_stock'])) /
                              FA_dsm_SSP1.loc[2017:]['stock_total']
    })
    check_df_inflow = pd.DataFrame({
        'new_inflow': S1_sio_new.loc[2017:]['total_inflow'],
        'correct_inflow': FA_dsm_SSP1.loc[2017:]['inflow_total'],
        'difference': S1_sio_new.loc[2017:]['total_inflow'] - FA_dsm_SSP1.loc[2017:]['inflow_total']
    })
    check_df_outflow = pd.DataFrame({
        'existing_outflow': os_existing_SSP1.loc[2017:]['total_outflow'],
        'new_outflow': S1_sio_new.loc[2017:]['total_outflow'],
        'correct_outflow': FA_dsm_SSP1.loc[2017:]['outflow_total'],
        'percent_difference': (os_existing_SSP1.loc[2017:]['total_outflow'] + S1_sio_new.loc[2017:][
            'total_outflow'] - FA_dsm_SSP1.loc[2017:]['outflow_total']) / FA_dsm_SSP1.loc[2017:]['outflow_total']
    })
    # print results of the check
    print('Mean percent difference in stock is = ' + str(np.mean(check_df_stock['percent_difference']) * 100) + '%')
    print('Mean  difference in inflow is = ' + str(np.mean(check_df_inflow['difference'])))
    print('Mean percent difference in outflow is = ' + str(np.mean(check_df_outflow['percent_difference']) * 100) + '%')


## Next steps
# - check that materials are calculated correctly
# - Montecarlo simulation for