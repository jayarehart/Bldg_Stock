# Dynamic stock model (DSM) for residential buildings

# Import libraries
import pandas as pd
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from odym import dynamic_stock_model as dsm

# Load in datasets
data_pop_WiC = pd.read_excel('./InputData/Pop_Data.xlsx', sheet_name='US_pop_WiC')
data_pop_UN = pd.read_excel('./InputData/Pop_Data.xlsx', sheet_name='US_pop_UN')
data_gdp = pd.read_excel('./InputData/Pop_Data.xlsx', sheet_name='GDP_pc')
RECS_Weights = pd.read_excel('./InputData/Pop_Data.xlsx', sheet_name='res_weight')
CBECS_Weights = pd.read_excel('./InputData/Pop_Data.xlsx', sheet_name='com_weight')


# function to interpolate the population data
def interpolate_population(data_pop=data_pop_UN, data_source='UN', year1=1900, year2=2100, proj='median', kind='cubic', plot=True):
    """ Interpolate the population data between the specified years for the US.
        Choose a data source (UN, or WiC)
        Choose a scenario for future population data (median, upper_95, lower_95, upper_80, or lower_80).
        Options for 'kind' are [linear, cubic, nearest, previous, and next] """
    if data_source=='UN':
        # Create interpolations for population
        f_median = interp1d(data_pop.Year, data_pop.Median, kind=kind)
        f_upper_95 = interp1d(data_pop.Year, data_pop.Upper_95, kind=kind)
        f_lower_95 = interp1d(data_pop.Year, data_pop.Lower_95, kind=kind)
        f_upper_80 = interp1d(data_pop.Year, data_pop.Upper_80, kind=kind)
        f_lower_80 = interp1d(data_pop.Year, data_pop.Lower_80, kind=kind)

        # Study Period
        # year1 = 1900
        # year2 = 2100
        years = np.linspace(year1, year2, num=(year2 - year1 + 1), endpoint=True)

        if proj == 'median':
            US_pop = f_median(years)
        elif proj == 'upper_95':
            US_pop = f_upper_95(years)
        elif proj == 'upper_80':
            US_pop = f_upper_80(years)
        elif proj == 'lower_95':
            US_pop = f_lower_95(years)
        elif proj == 'lower_80':
            US_pop = f_lower_80(years)
        else:
            US_pop = None
        US_pop_years = pd.DataFrame({'Year': years,
                                     'US_pop': US_pop})

        if plot == True:
            # Plot of population forecasts
            plt1, = plt.plot(years, f_upper_95(years))
            plt2, = plt.plot(years, f_upper_80(years))
            plt3, = plt.plot(years, f_median(years))
            plt4, = plt.plot(years, f_lower_80(years))
            plt5, = plt.plot(years, f_lower_95(years))
            plt6, = plt.plot([base_year, base_year], [2.4e8, 4.5e8], color='k', LineStyle='--')
            plt.legend([plt1, plt2, plt3, plt4, plt5],
                       ['Upper 95th', 'Upper 80th', 'Median', 'Lower 80th', 'Lower 95th'],
                       loc=2)
            plt.xlabel('Year')
            plt.ylabel('US Population')
            plt.title('Historical and Forecast of Population in the US (US Census and UN)')
            plt.show();
    elif data_source=='WiC':
        # Create interpolations for population
        f_SSP1 = interp1d(data_pop.Year, data_pop.SSP1, kind=kind)
        f_SSP2 = interp1d(data_pop.Year, data_pop.SSP2, kind=kind)
        f_SSP3 = interp1d(data_pop.Year, data_pop.SSP3, kind=kind)
        f_SSP4 = interp1d(data_pop.Year, data_pop.SSP4, kind=kind)
        f_SSP5 = interp1d(data_pop.Year, data_pop.SSP5, kind=kind)

        # Study Period
        # year1 = 1900
        # year2 = 2100
        years = np.linspace(year1, year2, num=(year2 - year1 + 1), endpoint=True)

        if proj == 'SSP1':
            US_pop = f_SSP1(years)
            US_pop_years = pd.DataFrame({'Year': years,
                                         'US_pop_' + str(proj): US_pop})
        elif proj == 'SSP2':
            US_pop = f_SSP2(years)
            US_pop_years = pd.DataFrame({'Year': years,
                                         'US_pop_' + str(proj): US_pop})
        elif proj == 'SSP3':
            US_pop = f_SSP3(years)
            US_pop_years = pd.DataFrame({'Year': years,
                                         'US_pop_' + str(proj): US_pop})
        elif proj == 'SSP4':
            US_pop = f_SSP4(years)
            US_pop_years = pd.DataFrame({'Year': years,
                                         'US_pop_' + str(proj): US_pop})
        elif proj == 'SSP5':
            US_pop = f_SSP5(years)
            US_pop_years = pd.DataFrame({'Year': years,
                                         'US_pop_' + str(proj): US_pop})
        elif proj == 'All':
            US_pop_years = pd.DataFrame({'Year': years,
                                   'US_pop_SSP1': f_SSP1(years),
                                   'US_pop_SSP2': f_SSP2(years),
                                   'US_pop_SSP3': f_SSP3(years),
                                   'US_pop_SSP4': f_SSP4(years),
                                   'US_pop_SSP5': f_SSP5(years),
                                   })
        else:
            US_pop_years = None


        if plot == True:
            # Plot of population forecasts
            plt1, = plt.plot(years, f_SSP1(years))
            plt2, = plt.plot(years, f_SSP2(years))
            plt3, = plt.plot(years, f_SSP3(years))
            plt4, = plt.plot(years, f_SSP4(years))
            plt5, = plt.plot(years, f_SSP5(years))
            plt6, = plt.plot([base_year, base_year], [2.4e8, 4.5e8], color='k', LineStyle='--')
            plt.legend([plt1, plt2, plt3, plt4, plt5],
                       ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'],
                       loc=2)
            plt.xlabel('Year')
            plt.ylabel('US Population')
            plt.title(r'Historical and Forecast of Population in the US' + '\n' + '(Census and WiC)')
            plt.show();

    return years, US_pop_years

# function to interpolate the gdp data
def interpolate_gdp(data_gdp, year1=1900, year2=2100, SSP='SSP1', kind='cubic', plot=True):
    """ Interpolate the GDP data between the specified years for the US.
        Choose a UN projection for future population data (median, upper_95, lower_95, upper_80, or lower_80).
        Options for 'kind' are [linear, cubic, nearest, previous, and next] """

    # Create interpolations for population
    f_SSP1 = interp1d(data_gdp.Year, data_gdp.SSP1, kind=kind)
    f_SSP2 = interp1d(data_gdp.Year, data_gdp.SSP2, kind=kind)
    f_SSP3 = interp1d(data_gdp.Year, data_gdp.SSP3, kind=kind)
    f_SSP4 = interp1d(data_gdp.Year, data_gdp.SSP4, kind=kind)
    f_SSP5 = interp1d(data_gdp.Year, data_gdp.SSP5, kind=kind)

    # Study Period
    # year1 = 1900
    # year2 = 2100
    years = np.linspace(year1, year2, num=(year2 - year1 + 1), endpoint=True)

    if SSP == 'SSP1':
        US_gdp = f_SSP1(years)
    elif SSP == 'SSP2':
        US_gdp = f_SSP2(years)
    elif SSP == 'SSP3':
        US_gdp = f_SSP3(years)
    elif SSP == 'SSP4':
        US_gdp = f_SSP4(years)
    elif SSP == 'SSP5':
        US_gdp = f_SSP5(years)
    elif SSP == 'All':
        US_gdp = pd.DataFrame({'gdp_SSP1': f_SSP1(years),
                               'gdp_SSP2': f_SSP2(years),
                               'gdp_SSP3': f_SSP3(years),
                               'gdp_SSP4': f_SSP4(years),
                               'gdp_SSP5': f_SSP5(years)})
    else:
        US_gdp = None
    years_df = pd.DataFrame({'Year': years})
    US_gdp_years = pd.concat([years_df, US_gdp], axis=1)

    if plot == True:
        # Plot of population forecasts
        plt1, = plt.plot(years, f_SSP1(years))
        plt2, = plt.plot(years, f_SSP2(years))
        plt3, = plt.plot(years, f_SSP3(years))
        plt4, = plt.plot(years, f_SSP4(years))
        plt5, = plt.plot(years, f_SSP5(years))
        # plt6, = plt.plot([base_year, base_year], [2.4e8, 4.5e8], color='k', LineStyle='--')
        plt.legend([plt1, plt2, plt3, plt4, plt5],
                   ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'],
                   loc=2)
        plt.xlabel('Year')
        plt.ylabel('US GDP per capita')
        plt.title('Historical and Forecast of per-capita GDP in the US')
        plt.show();

    return US_gdp_years

# function to interpolate a simple linear increase in floor area elasticity
def FA_elasticity_linear(year1=1900, year2=2100, base_year=2016,
                         FA_elas_year1=100, FA_elas_base_year=200, FA_elas_year2=200,
                         plot=True, plot_name='Residential'):
    """ Determine the floor area elasticity to be used for each year based upon a linear interpolation
    between historical years, base year, and future year. Assumed interpolation between each year is linear.
    Essentially, it's two lines with two different slopes."""

    years = np.linspace(year1, year2, num=(year2 - year1 + 1), endpoint=True)
    FA_historic = np.linspace(FA_elas_year1, FA_elas_base_year, num=(base_year - year1 + 1), endpoint=True)
    FA_future = np.linspace(FA_elas_base_year, FA_elas_year2, num=(year2 - base_year), endpoint=True)

    FA_elas = pd.DataFrame({'Year': years,
                            'FA_elas': np.concatenate((FA_historic, FA_future), axis=0)
                            }, )
    if plot == True:
        # Plot of population forecasts
        plt1, = plt.plot(FA_elas.Year, FA_elas.FA_elas)
        plt2, = plt.plot([base_year, base_year], [0, 300], color='k', LineStyle='--')
        plt.xlabel('Year')
        plt.ylabel('Floor Area Elasticity')
        plt.title(plot_name + ' Linear interpolation')
        plt.show();
    return FA_elas

# function to calculate the floor area elasticity by the methodology of the EDGE model
def FA_elasticity_EDGE(US_gdp, US_pop, SSP='All',
                       base_year=2016,FA_base_year=347, Area_country=9.14759e6, gamma=-0.03,
                       plot=True):
    """ Area of the USA is 9.834 million km².
        Base year floor are elasticity for all buildings is 347 m2/person as determined by article (in review)"""

    def calc_FA_elas(gdp, SSP, SSP_split_year=1985):
        # Beta values for each SSP
        Beta_SSP = {'SSP1': 0.3,
                    'SSP2': 0.7,
                    'SSP3': 0.8,
                    'SSP4': 0.7,
                    'SSP5': 1.0}
        # General Beta for floor space demand:
        Beta = 0.42

        FA_df = pd.merge(US_pop, gdp, on='Year')
        FA_df = FA_df.set_index('Year', drop=False)

        # calculate historical FA
        FA_df['Pop_Dens_'+SSP] = FA_df['US_pop_'+SSP] / Area_country
        base_year_0_df = FA_df.loc[[base_year]]
        alpha  = FA_base_year / (base_year_0_df['gdp_'+SSP] ** Beta * base_year_0_df['Pop_Dens_'+SSP] ** gamma)
        # alpha reported by EDGE model is 0.61.
        # alpha from Arehart et al. 2020 high   = 5.002223
        #                                median = 4.350305
        #                                low    = 3.635701
        # print('Alpha is = ' + str(alpha))

        FA_df.loc[FA_df['Year'] <= base_year, 'FA_elas_'+SSP] = alpha[base_year] * (FA_df['gdp_'+SSP] ** Beta) * FA_df['Pop_Dens_'+SSP] ** gamma
        for i in range(1,len(FA_df)):
            year_i = FA_df.index[i]
            if year_i > base_year:
                row_t_1 = FA_df.loc[year_i-1]
                FA_t_1 = row_t_1['FA_elas_'+SSP]
                I_t_1 = row_t_1['gdp_'+SSP]
                D_t_1 = row_t_1['Pop_Dens_'+SSP]
                row_t = FA_df.loc[year_i]
                I_t = row_t['gdp_'+SSP]
                D_t = row_t['Pop_Dens_'+SSP]
                FA_df.loc[int(year_i),'FA_elas_'+SSP] = FA_t_1 * (I_t/I_t_1) ** Beta_SSP[SSP] * (D_t/D_t_1) ** gamma
        return FA_df

    FA_SSP1 = calc_FA_elas(US_gdp, 'SSP1')
    FA_SSP2 = calc_FA_elas(US_gdp, 'SSP2')
    FA_SSP3 = calc_FA_elas(US_gdp, 'SSP3')
    FA_SSP4 = calc_FA_elas(US_gdp, 'SSP4')
    FA_SSP5 = calc_FA_elas(US_gdp, 'SSP5')

    if SSP=='SSP1':
        df_return = FA_SSP1
    elif SSP=='SSP2':
        df_return = FA_SSP2
    elif SSP=='SSP3':
        df_return = FA_SSP3
    elif SSP=='SSP4':
        df_return = FA_SSP4
    elif SSP=='SSP5':
        df_return = FA_SSP5
    elif SSP=='All':
        df_return = pd.DataFrame({'Year': FA_SSP1['Year'],
                                  'US_pop_SSP1': FA_SSP1['US_pop_SSP1'],
                                  'US_pop_SSP2': FA_SSP1['US_pop_SSP2'],
                                  'US_pop_SSP3': FA_SSP1['US_pop_SSP3'],
                                  'US_pop_SSP4': FA_SSP1['US_pop_SSP4'],
                                  'US_pop_SSP5': FA_SSP1['US_pop_SSP5'],
                                  'US_gdp_SSP1': FA_SSP1['gdp_SSP1'],
                                  'US_gdp_SSP2': FA_SSP1['gdp_SSP2'],
                                  'US_gdp_SSP3': FA_SSP1['gdp_SSP3'],
                                  'US_gdp_SSP4': FA_SSP1['gdp_SSP4'],
                                  'US_gdp_SSP5': FA_SSP1['gdp_SSP5'],
                                  'FA_SSP1': FA_SSP1['FA_elas_SSP1'],
                                  'FA_SSP2': FA_SSP2['FA_elas_SSP2'],
                                  'FA_SSP3': FA_SSP3['FA_elas_SSP3'],
                                  'FA_SSP4': FA_SSP4['FA_elas_SSP4'],
                                  'FA_SSP5': FA_SSP5['FA_elas_SSP5'], })

        if plot == True:
            # Plot GFA vs time.
            max_GFA = max(FA_SSP1.FA_elas_SSP1.max(), FA_SSP2.FA_elas_SSP2.max(), FA_SSP3.FA_elas_SSP3.max(), FA_SSP4.FA_elas_SSP4.max(), FA_SSP5.FA_elas_SSP5.max())
            plt1, = plt.plot(df_return.index, df_return.FA_SSP1)
            plt2, = plt.plot(df_return.index, df_return.FA_SSP2)
            plt3, = plt.plot(df_return.index, df_return.FA_SSP3)
            plt4, = plt.plot(df_return.index, df_return.FA_SSP4)
            plt5, = plt.plot(df_return.index, df_return.FA_SSP5)
            plt6, = plt.plot([base_year, base_year], [0, max_GFA], color='k', LineStyle='--')
            plt.legend([plt1, plt2, plt3, plt4, plt5],
                       ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], loc=2)
            plt.xlabel('Year')
            plt.ylabel('Floor Area Elasticity')
            plt.title('Floor Area Elasticity for various SSPs')
            plt.show();
            # Plot GFA vs GDP.
            plt1, = plt.plot(df_return.US_gdp_SSP1, df_return.FA_SSP1)
            plt2, = plt.plot(df_return.US_gdp_SSP2, df_return.FA_SSP2)
            plt3, = plt.plot(df_return.US_gdp_SSP3, df_return.FA_SSP3)
            plt4, = plt.plot(df_return.US_gdp_SSP4, df_return.FA_SSP4)
            plt5, = plt.plot(df_return.US_gdp_SSP5, df_return.FA_SSP5)
            plt.legend([plt1, plt2, plt3, plt4, plt5],
                       ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], loc=2)
            plt.xlabel('GDP')
            plt.ylabel('Floor Area Elasticity')
            plt.title('Floor Area Elasticity for various SSPs')
            plt.show();


    return df_return


    # FA_historic = np.linspace(FA_elas_year1, FA_elas_base_year, num=(base_year - year1 + 1), endpoint=True)
    # FA_future = np.linspace(FA_elas_base_year, FA_elas_year2, num=(year2 - base_year), endpoint=True)
    #
    # FA_elas = pd.DataFrame({'Year': years,
    #                         'FA_elas': np.concatenate((FA_historic, FA_future), axis=0)
    #                         }, )

# Time period input variables
year1 = 1900
year2 = 2100
base_year = 2016

# interpolate population data for the US.
years, US_pop = interpolate_population(data_pop=data_pop_WiC, data_source='WiC', year1=year1, year2=year2, proj='All', plot=True)

# interpolate gdp data for the US.
US_gdp = interpolate_gdp(data_gdp, year1=1900, year2=2100, SSP='All', kind='cubic', plot=True)
# calculate total floor area elasticity
FA_all = FA_elasticity_EDGE(US_gdp, US_pop, SSP='All',
                       base_year=2016,FA_base_year=347, Area_country=9.14759e6, gamma=-0.03,
                       plot=True)


US_pop = US_pop.set_index('Year', drop=False)
US_gdp = US_gdp.set_index('Year', drop=False)


# ratio of residential floor area to total floor area:
ratio_res = 0.773497
ratio_com = 0.142467
ratio_pub = 0.030018

def plot_dsm(dsm, plot_name):
    plt.subplot(211)
    max_val = dsm.s.max()
    plt2, = plt.plot(dsm.t, dsm.s)
    plt4, = plt.plot([base_year, base_year], [0, max_val], color='k', LineStyle='--')
    plt.legend([plt2], ['Stock'], loc=2)
    plt.xlabel('Year')
    plt.ylabel('Floor Area')
    plt.title(plot_name + ' Floor Area Stock')
    # plt.show();

    plt.subplot(212)
    # max val
    max_val = max(dsm.i[1:].max(), dsm.o[1:].max())
    plt1, = plt.plot(dsm.t, dsm.i)
    plt3, = plt.plot(dsm.t, dsm.o)
    plt4, = plt.plot([base_year, base_year], [0, 1.15 * max_val], color='k', LineStyle='--')
    plt.xlim(left=dsm.t[0] + 5)
    plt.ylim(top=1.2 * max_val)
    plt.xlabel('Year')
    plt.ylabel('Floor Area per year')
    plt.title(plot_name + ' Floor Area flows')
    plt.legend([plt1, plt3], ['Inflow', 'Outflow'], loc=2)
    plt.show();

    # Stock by age-cohort
    plt.imshow(dsm.s_c[:, 1:], interpolation='nearest')  # exclude the first column to have the color scale work.
    plt.xlabel('age-cohort')
    plt.ylabel('year')
    plt.title(plot_name + ' Stock by age-cohort')
    plt.show();

def do_stock_driven_model(t, s, lt, InitialStock, SwitchTime, plot=True, plot_name='Residential'):
    """ Compute a stock driven model from an initial stock.
        Returns an object with class dynamic_stock_model with age-cohort matrix
        with inputs, outputs, and stocks computed"""

    my_dsm = dsm.DynamicStockModel(t=t, s=s, lt=lt)
    CheckStr = my_dsm.dimension_check()
    print(CheckStr)

    S_C = my_dsm.compute_evolution_initialstock(InitialStock=InitialStock, SwitchTime=SwitchTime)
    S_C, O_C, I = my_dsm.compute_stock_driven_model(NegativeInflowCorrect=True)

    O = my_dsm.compute_outflow_total()  # Total outflow
    DS = my_dsm.compute_stock_change()  # Stock change
    Bal = my_dsm.check_stock_balance()  # Stock balance
    print('The mass balance between inflows and outflows is:   ')
    print(np.abs(Bal).sum())  # show sum absolute of all mass balance mismatches.
    if plot==True: plot_dsm(my_dsm, plot_name=plot_name)
    print('Difference in stock in base year is: ')
    # print(sum(S_C[list(t).index(base_year)]) - sum(InitialStock))
    print(sum(S_C[SwitchTime-1]) - sum(InitialStock))

    return my_dsm

def calc_MFA(scenario, lifetime):
    # select a scenario to consider
    # scenario = 'SSP2'
    scenario_pop = US_pop['US_pop_'+scenario]
    scenario_gdp = US_gdp['gdp_'+scenario]
    scenario_FAE_res = FA_all['FA_'+scenario] * ratio_res
    scenario_FAE_com = FA_all['FA_'+scenario] * ratio_com
    scenario_FAE_pub = FA_all['FA_' + scenario] * ratio_pub

    # calculate demanded floor area stock
    stock_res = scenario_pop.mul(scenario_FAE_res) / 1000000
    stock_com = scenario_pop.mul(scenario_FAE_com) / 1000000
    stock_pub = scenario_pop.mul(scenario_FAE_pub) / 1000000

    # Summary of input data for material flow analysis
    MFA_input_data = pd.DataFrame({'Year': years,
                                   'US_pop_'+scenario: scenario_pop,
                                   'US_gdp_'+scenario: scenario_gdp,
                                   'FA_elasticity_res_'+scenario: scenario_FAE_res,
                                   'FA_elasticity_com_'+scenario: scenario_FAE_com,
                                   'FA_elasticity_pub_' + scenario: scenario_FAE_pub,
                                   'stock_res_'+scenario: stock_res,
                                   'stock_com_'+scenario: stock_com,
                                   'stock_pub_'+scenario: stock_pub})

    # ---- Building lifespan distributions ----
    BldgLife_mean_res = 80  # years
    BldgLife_StdDev_res = 0.2 *  np.array([BldgLife_mean_res] * len(years))
    BldgLife_mean_com = 70  # years
    BldgLife_StdDev_com = 0.2 *  np.array([BldgLife_mean_com] * len(years))
    BldgLife_mean_pub = 90  # years
    BldgLife_StdDev_pub = 0.2 * np.array([BldgLife_mean_com] * len(years))
    if lifetime=='Normal':
        # Normal
        lt_res = {'Type': 'Normal', 'Mean': np.array([BldgLife_mean_res] * len(years)), 'StdDev': BldgLife_StdDev_res}
        lt_com = {'Type': 'Normal', 'Mean': np.array([BldgLife_mean_com] * len(years)), 'StdDev': BldgLife_StdDev_com}
        lt_pub = {'Type': 'Normal', 'Mean': np.array([BldgLife_mean_pub] * len(years)), 'StdDev': BldgLife_StdDev_pub}
    elif lifetime=='Weibull':
        # Weibull
        lt_res = {'Type': 'Weibull', 'Shape': np.array([5.5]), 'Scale': np.array([85.8])}
        lt_com = {'Type': 'Weibull', 'Shape': np.array([4.8]), 'Scale': np.array([75.1])}
        lt_pub = {'Type': 'Weibull', 'Shape': np.array([6.1]), 'Scale': np.array([95.6])}
    elif lifetime=='Gamma':
        # Gamma (currently not working)
        lt_res = {'Type': 'Gamma', 'Scale': np.array([2.7]), 'Shape': np.array([32.9])}
        lt_com = {'Type': 'Gamma', 'Scale': np.array([2.7]), 'Shape': np.array([32.9])}
        print('Gamma distribution for lifetime not working...')
    elif lifetime=='Lognormal':
        # Lognormal (currently not working)
        lt_res = {'Type': 'LogNormal', 'Mean': np.array([50]), 'StdDev': np.array([10])}
        lt_com = {'Type': 'LogNormal', 'Mean': np.array([50]), 'StdDev': np.array([10])}
        print('Lognormal distribution for lifetime not working...')
    elif lifetime=='Fixed':
        # Fixed lifetime
        lt_res = {'Type': 'Fixed', 'Mean': np.array([BldgLife_mean_res] * len(years))}
        lt_com = {'Type': 'Fixed', 'Mean': np.array([BldgLife_mean_com] * len(years))}
        lt_pub = {'Type': 'Fixed', 'Mean': np.array([BldgLife_mean_pub] * len(years))}
    else:
        print('Lifetime is not supported. Please choose another one...')

    # ---- Initial Conditions ----
    # Residential floor area  age-cohort in base year: 2015
    S_0_res_2015 = np.flipud(RECS_Weights.Res_Weight) * stock_res[2015]

    # Commercial floor area  age-cohort in base year: 2012
    S_0_com_2012 = np.flipud(CBECS_Weights.Com_Weight) * stock_com[2012]

    # Public floor area  age-cohort in base year: 2012
    S_0_pub_2012 = np.flipud(CBECS_Weights.Com_Weight) * stock_pub[2012]

    # Residential
    t = years
    s = np.array(stock_res)
    InitialStock_res = S_0_res_2015
    SwitchTime = 116
    US_stock_res = do_stock_driven_model(t, s, lt_res, InitialStock_res, SwitchTime, plot=False, plot_name='Residential')

    # Commercial
    t = years
    s = np.array(stock_com)
    lt = lt_com
    InitialStock_com = S_0_com_2012
    SwitchTime = 113
    US_stock_com = do_stock_driven_model(t, s, lt_com, InitialStock_com, SwitchTime, plot=False,
                                             plot_name='Commercial')

    # Public
    t = years
    s = np.array(stock_pub)
    lt = lt_pub
    InitialStock_pub = S_0_pub_2012
    SwitchTime = 113
    US_stock_pub = do_stock_driven_model(t, s, lt_pub, InitialStock_pub, SwitchTime, plot=False,
                                             plot_name='Public')

    return US_stock_res, US_stock_com, US_stock_pub

# Calculate MFA for individual scenarios
SSP1_dsm_res, SSP1_dsm_com, SSP1_dsm_pub = calc_MFA('SSP1', 'Weibull')
SSP2_dsm_res, SSP2_dsm_com, SSP2_dsm_pub = calc_MFA('SSP2', 'Weibull')
SSP3_dsm_res, SSP3_dsm_com, SSP3_dsm_pub = calc_MFA('SSP3', 'Weibull')
SSP4_dsm_res, SSP4_dsm_com, SSP4_dsm_pub = calc_MFA('SSP4', 'Weibull')
SSP5_dsm_res, SSP5_dsm_com, SSP5_dsm_pub = calc_MFA('SSP5', 'Weibull')

# Save the floor area models as .csv files.

SSP1_dsm_df = pd.DataFrame({'time': SSP1_dsm_res.t,
                            'stock_res': SSP1_dsm_res.s,
                            'inflow_res': SSP1_dsm_res.i,
                            'outflow_res': SSP1_dsm_res.o,
                            'stock_com': SSP1_dsm_com.s,
                            'inflow_com': SSP1_dsm_com.i,
                            'outflow_com': SSP1_dsm_com.o,
                            'stock_pub': SSP1_dsm_pub.s,
                            'inflow_pub': SSP1_dsm_pub.i,
                            'outflow_pub': SSP1_dsm_pub.o,
                            'stock_total': SSP1_dsm_res.s + SSP1_dsm_com.s + SSP1_dsm_pub.s,
                            'inflow_total': SSP1_dsm_res.i + SSP1_dsm_com.i + SSP1_dsm_pub.i,
                            'outflow_total': SSP1_dsm_res.o + SSP1_dsm_com.o + SSP1_dsm_pub.o
                            })
SSP2_dsm_df = pd.DataFrame({'time': SSP2_dsm_res.t,
                            'stock_res': SSP2_dsm_res.s,
                            'inflow_res': SSP2_dsm_res.i,
                            'outflow_res': SSP2_dsm_res.o,
                            'stock_com': SSP2_dsm_com.s,
                            'inflow_com': SSP2_dsm_com.i,
                            'outflow_com': SSP2_dsm_com.o,
                            'stock_pub': SSP2_dsm_pub.s,
                            'inflow_pub': SSP2_dsm_pub.i,
                            'outflow_pub': SSP2_dsm_pub.o,
                            'stock_total': SSP2_dsm_res.s + SSP2_dsm_com.s + SSP2_dsm_pub.s,
                            'inflow_total': SSP2_dsm_res.i + SSP2_dsm_com.i + SSP2_dsm_pub.i,
                            'outflow_total': SSP2_dsm_res.o + SSP2_dsm_com.o + SSP2_dsm_pub.o
                            })
SSP3_dsm_df = pd.DataFrame({'time': SSP3_dsm_res.t,
                            'stock_res': SSP3_dsm_res.s,
                            'inflow_res': SSP3_dsm_res.i,
                            'outflow_res': SSP3_dsm_res.o,
                            'stock_com': SSP3_dsm_com.s,
                            'inflow_com': SSP3_dsm_com.i,
                            'outflow_com': SSP3_dsm_com.o,
                            'stock_pub': SSP3_dsm_pub.s,
                            'inflow_pub': SSP3_dsm_pub.i,
                            'outflow_pub': SSP3_dsm_pub.o,
                            'stock_total': SSP3_dsm_res.s + SSP3_dsm_com.s + SSP3_dsm_pub.s,
                            'inflow_total': SSP3_dsm_res.i + SSP3_dsm_com.i + SSP3_dsm_pub.i,
                            'outflow_total': SSP3_dsm_res.o + SSP3_dsm_com.o + SSP3_dsm_pub.o
                            })
SSP4_dsm_df = pd.DataFrame({'time': SSP4_dsm_res.t,
                            'stock_res': SSP4_dsm_res.s,
                            'inflow_res': SSP4_dsm_res.i,
                            'outflow_res': SSP4_dsm_res.o,
                            'stock_com': SSP4_dsm_com.s,
                            'inflow_com': SSP4_dsm_com.i,
                            'outflow_com': SSP4_dsm_com.o,
                            'stock_pub': SSP4_dsm_pub.s,
                            'inflow_pub': SSP4_dsm_pub.i,
                            'outflow_pub': SSP4_dsm_pub.o,
                            'stock_total': SSP4_dsm_res.s + SSP4_dsm_com.s + SSP4_dsm_pub.s,
                            'inflow_total': SSP4_dsm_res.i + SSP4_dsm_com.i + SSP4_dsm_pub.i,
                            'outflow_total': SSP4_dsm_res.o + SSP4_dsm_com.o + SSP4_dsm_pub.o
                            })
SSP5_dsm_df = pd.DataFrame({'time': SSP5_dsm_res.t,
                            'stock_res': SSP5_dsm_res.s,
                            'inflow_res': SSP5_dsm_res.i,
                            'outflow_res': SSP5_dsm_res.o,
                            'stock_com': SSP5_dsm_com.s,
                            'inflow_com': SSP5_dsm_com.i,
                            'outflow_com': SSP5_dsm_com.o,
                            'stock_pub': SSP5_dsm_pub.s,
                            'inflow_pub': SSP5_dsm_pub.i,
                            'outflow_pub': SSP5_dsm_pub.o,
                            'stock_total': SSP5_dsm_res.s + SSP5_dsm_com.s + SSP5_dsm_pub.s,
                            'inflow_total': SSP5_dsm_res.i + SSP5_dsm_com.i + SSP5_dsm_pub.i,
                            'outflow_total': SSP5_dsm_res.o + SSP5_dsm_com.o + SSP5_dsm_pub.o
                            })

# write to excel
writer = pd.ExcelWriter('./Results/SSP_dsm.xlsx', engine='xlsxwriter')
SSP1_dsm_df.to_excel(writer, sheet_name='SSP1')
SSP2_dsm_df.to_excel(writer, sheet_name='SSP2')
SSP3_dsm_df.to_excel(writer, sheet_name='SSP3')
SSP4_dsm_df.to_excel(writer, sheet_name='SSP4')
SSP5_dsm_df.to_excel(writer, sheet_name='SSP5')
writer.save()


# Plot the material flow analyses
plot_dsm(SSP1_dsm_res, 'SSP1 Residential')
plot_dsm(SSP1_dsm_com, 'SSP1 Commercial')
plot_dsm(SSP1_dsm_pub, 'SSP1 Public')
# plot_dsm(SSP2_dsm_res, 'SSP2 Residential')
# plot_dsm(SSP2_dsm_com, 'SSP2 Commercial')
# plot_dsm(SSP2_dsm_pub, 'SSP2 Public')
# plot_dsm(SSP3_dsm_res, 'SSP3 Residential')
# plot_dsm(SSP3_dsm_com, 'SSP3 Commercial')
# plot_dsm(SSP3_dsm_pub, 'SSP3 Public')
# plot_dsm(SSP4_dsm_res, 'SSP4 Residential')
# plot_dsm(SSP4_dsm_com, 'SSP4 Commercial')
# plot_dsm(SSP4_dsm_pub, 'SSP4 Public')
# plot_dsm(SSP5_dsm_res, 'SSP5 Residential')
# plot_dsm(SSP5_dsm_com, 'SSP5 Commercial')
# plot_dsm(SSP5_dsm_pub, 'SSP5 Public')

# ----------------------------------------------------------------------------------------------------------------------
# Plot all scenarios together for residential buildings
plt.subplot(211)
plt1, = plt.plot(SSP1_dsm_res.t, SSP1_dsm_res.s)
plt2, = plt.plot(SSP2_dsm_res.t, SSP2_dsm_res.s)
plt3, = plt.plot(SSP3_dsm_res.t, SSP3_dsm_res.s)
plt4, = plt.plot(SSP4_dsm_res.t, SSP4_dsm_res.s)
plt5, = plt.plot(SSP5_dsm_res.t, SSP5_dsm_res.s)
plt16, = plt.plot([base_year, base_year], [0, 100000], color='k', LineStyle='--')
plt.legend([plt1, plt2, plt3, plt4, plt5], ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], loc=(1.05, 0.5))
# plt.legend(loc=(1.05, 0.5))
plt.tight_layout()
plt.xlabel('Year')
plt.ylabel('Floor Area')
plt.title('Residential Floor Space - Stock')
# plt.show();

plt.subplot(212)
plt1, = plt.plot(SSP1_dsm_res.t, SSP1_dsm_res.i, LineStyle='dashed')
plt2, = plt.plot(SSP1_dsm_res.t, SSP1_dsm_res.o)
plt3, = plt.plot(SSP2_dsm_res.t, SSP2_dsm_res.i, LineStyle='dashed')
plt4, = plt.plot(SSP2_dsm_res.t, SSP2_dsm_res.o)
plt5, = plt.plot(SSP3_dsm_res.t, SSP3_dsm_res.i, LineStyle='dashed')
plt6, = plt.plot(SSP3_dsm_res.t, SSP3_dsm_res.o)
plt7, = plt.plot(SSP4_dsm_res.t, SSP4_dsm_res.i, LineStyle='dashed')
plt8, = plt.plot(SSP4_dsm_res.t, SSP4_dsm_res.o)
plt9, = plt.plot(SSP5_dsm_res.t, SSP5_dsm_res.i, LineStyle='dashed')
plt0, = plt.plot(SSP5_dsm_res.t, SSP5_dsm_res.o)

plt11, = plt.plot([base_year, base_year], [0, 4000], color='k', LineStyle='--')
plt.legend([plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9, plt0],
           ['Inflow SSP1', 'Outflow SSP1',
            'Inflow SSP2', 'Outflow SSP2',
            'Inflow SSP3', 'Outflow SSP3',
            'Inflow SSP4', 'Outflow SSP4',
            'Inflow SSP5', 'Outflow SSP5'], loc='center left', bbox_to_anchor=(1, 0.5))
# plt.legend([plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8],
#            ['Inflow SSP1', 'Outflow SSP1',
#             'Inflow SSP2', 'Outflow SSP2',
#             'Inflow SSP3', 'Outflow SSP3',
#             'Inflow SSP4', 'Outflow SSP4'], loc='center left', bbox_to_anchor=(1, 0.5))
# plt.ylim(top=5000)
plt.xlim(left=SSP1_dsm_res.t[0] + 5)
plt.xlabel('Year')
plt.ylabel('Floor Area per year')
plt.title('Residential Floor Area flows')
plt.show();





# ----------------------------------------------------------------------------------------------------------------------
# Plot weibull lifetime distributions:
x = np.arange(1,150)
def weib(x,n,a):
    return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

# count, bins, ignored = plt.hist(np.random.weibull(5.5,1000))
# scale = count.max()/weib(x, 85.8, 5.5).max()
plt1, = plt.plot(x, weib(x, 85.5, 5.5)*85.5)
plt2, = plt.plot(x, weib(x, 75.1, 4.8)*75.1)
plt3, = plt.plot(x, weib(x, 95.6, 6.1)*95.6)
plt.legend([plt1, plt2, plt3],
           ['Weibull - Residential', 'Weibull - Commercial', 'Weibull - Public'], loc=2)
plt.title('Weibull distributions from Bureau of Ecomonic Analysis 2003')
plt.xlabel('Building lifespan')
plt.show()










# ----------------------------------------------------------------------------------------------------------------------
# OLD CODE
#
# 'compute_stock_driven_model_initialstock
# BldgLife_mean = 120  # years
# Normal_StdDev = 0.6 *  np.array([BldgLife_mean] * len(years))
# lifetime_NormalLT = {'Type': 'Normal', 'Mean': np.array([BldgLife_mean] * len(years)), 'StdDev': Normal_StdDev}
#
# US_Bldg_DSM_3 = dsm.DynamicStockModel(t=years, s=FA_stock_res, lt=lifetime_NormalLT)
# CheckStr = US_Bldg_DSM_3.dimension_check()
# print(CheckStr)
#
# Initial_Stock = np.flipud(S_0_res_2015)
#
# S_C, O_C, I = US_Bldg_DSM_3.compute_stock_driven_model_initialstock(InitialStock=Initial_Stock,
#                                                                   SwitchTime=list(years).index(base_year + 2),
#                                                                   NegativeInflowCorrect=False)
# O = US_Bldg_DSM_3.compute_outflow_total()  # Total outflow
# DS = US_Bldg_DSM_3.compute_stock_change()  # Stock change
# Bal = US_Bldg_DSM_3.check_stock_balance()  # Vehicle balance
# print(np.abs(Bal).sum())  # show sum absolute of all mass balance mismatches.
# plot_dsm(US_Bldg_DSM_3)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 'compute_stock_driven_model':
# US_Bldg_DSM_1 = dsm.DynamicStockModel(t=years, s=FA_stock_res, lt=lifetime_NormalLT)
# CheckStr = US_Bldg_DSM_1.dimension_check()
# print(CheckStr)
# S_C, O_C, I = US_Bldg_DSM_1.compute_stock_driven_model()
# O = US_Bldg_DSM_1.compute_outflow_total()  # Total outflow
# DS = US_Bldg_DSM_1.compute_stock_change()  # Stock change
# Bal = US_Bldg_DSM_1.check_stock_balance()  # Vehicle balance
# print(np.abs(Bal).sum())  # show sum absolute of all mass balance mismatches.
# plot_dsm(US_Bldg_DSM_1)






