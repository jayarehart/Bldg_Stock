# Dynamic stock model (DSM) for residential buildings

# Import libraries
import pandas as pd
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from odym import dynamic_stock_model as dsm
from scipy import stats
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
import statsmodels.distributions

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
                                  'US_pop_SSP2': FA_SSP2['US_pop_SSP2'],
                                  'US_pop_SSP3': FA_SSP3['US_pop_SSP3'],
                                  'US_pop_SSP4': FA_SSP4['US_pop_SSP4'],
                                  'US_pop_SSP5': FA_SSP5['US_pop_SSP5'],
                                  'US_gdp_SSP1': FA_SSP1['gdp_SSP1'],
                                  'US_gdp_SSP2': FA_SSP2['gdp_SSP2'],
                                  'US_gdp_SSP3': FA_SSP3['gdp_SSP3'],
                                  'US_gdp_SSP4': FA_SSP4['gdp_SSP4'],
                                  'US_gdp_SSP5': FA_SSP5['gdp_SSP5'],
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
year1 = 1820
year2 = 2100
base_year = 2016

# interpolate population data for the US.
years, US_pop = interpolate_population(data_pop=data_pop_WiC, data_source='WiC', year1=year1, year2=year2, proj='All', plot=True)

# interpolate gdp data for the US.
US_gdp = interpolate_gdp(data_gdp, year1=year1, year2=year2, SSP='All', kind='cubic', plot=True)
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

def do_stock_driven_model(t, s, lt, plot=True, plot_name='Residential'):
    """ Compute a stock driven model from an initial stock.
        Returns an object with class dynamic_stock_model with age-cohort matrix
        with inputs, outputs, and stocks computed"""

    my_dsm = dsm.DynamicStockModel(t=t, s=s, lt=lt)
    CheckStr = my_dsm.dimension_check()
    print(CheckStr)

    # S_C = my_dsm.compute_evolution_initialstock(InitialStock=InitialStock, SwitchTime=SwitchTime)
    S_C, O_C, I = my_dsm.compute_stock_driven_model(NegativeInflowCorrect=True)

    O = my_dsm.compute_outflow_total()  # Total outflow
    DS = my_dsm.compute_stock_change()  # Stock change
    Bal = my_dsm.check_stock_balance()  # Stock balance
    print('The mass balance between inflows and outflows is:   ')
    print(np.abs(Bal).sum())  # show sum absolute of all mass balance mismatches.
    if plot==True: plot_dsm(my_dsm, plot_name=plot_name)
    return my_dsm

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
        lt = {'Type': type, 'Mean': np.array([par1] * len(years)), 'StdDev': par2}
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

lt_res = generate_lt('Weibull',par1=130, par2=3)
lt_com = generate_lt('Weibull',par1=100, par2=5)
lt_pub = generate_lt('Weibull',par1=150, par2=5)

# Plot lifetime distributions:
plot_lifetime_distr=True
if plot_lifetime_distr==True:
    x = np.arange(1,200)
    def weib(x,n,a):
        return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

    # count, bins, ignored = plt.hist(np.random.weibull(5.5,1000))
    # scale = count.max()/weib(x, 85.8, 5.5).max()
    plt1, = plt.plot(x, weib(x, lt_res['Shape'][0], lt_res['Scale'][0])*lt_res['Shape'][0], label='Weibull Residential')
    plt2, = plt.plot(x, weib(x, lt_com['Shape'][0], lt_com['Scale'][0])*lt_com['Shape'][0], label='Weibull Commercial')
    plt3, = plt.plot(x, weib(x, lt_pub['Shape'][0], lt_pub['Scale'][0])*lt_pub['Shape'][0], label='Weibull Public')
    plt.legend(loc=2)
    plt.title('Input Weibull Distributions')
    plt.xlabel('Building lifespan')
    plt.show()


def calc_MFA(scenario, lt_res, lt_com, lt_pub):
    # select a scenario to consider during debugging
    # scenario = 'SSP1'
    # lifetime = 'Weibull'
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



    # ---- Initial Conditions ----
    # Residential floor area  age-cohort in base year: 2015
    # S_0_res_2015 = np.flipud(RECS_Weights.Res_Weight) * stock_res[2015]

    # Commercial floor area  age-cohort in base year: 2012
    # S_0_com_2012 = np.flipud(CBECS_Weights.Com_Weight) * stock_com[2012]

    # Public floor area  age-cohort in base year: 2012
    # S_0_pub_2012 = np.flipud(CBECS_Weights.Com_Weight) * stock_pub[2012]

    # Residential
    t = years
    s = np.array(stock_res)
    # InitialStock_res = S_0_res_2015
    SwitchTime = 196
    US_stock_res = do_stock_driven_model(t, s, lt_res, plot=False, plot_name='Residential')

    # Commercial
    t = years
    s = np.array(stock_com)
    lt = lt_com
    # InitialStock_com = S_0_com_2012
    SwitchTime = 193
    US_stock_com = do_stock_driven_model(t, s, lt_com, plot=False, plot_name='Commercial')

    # Public
    t = years
    s = np.array(stock_pub)
    lt = lt_pub
    # InitialStock_pub = S_0_pub_2012
    SwitchTime = 193
    US_stock_pub = do_stock_driven_model(t, s, lt_pub, plot=False, plot_name='Public')

    return US_stock_res, US_stock_com, US_stock_pub, MFA_input_data

# Calculate MFA for individual scenarios
SSP1_dsm_res, SSP1_dsm_com, SSP1_dsm_pub, SSP1_MFA_input = calc_MFA('SSP1', lt_res, lt_com, lt_pub)
SSP2_dsm_res, SSP2_dsm_com, SSP2_dsm_pub, SSP2_MFA_input = calc_MFA('SSP2', lt_res, lt_com, lt_pub)
SSP3_dsm_res, SSP3_dsm_com, SSP3_dsm_pub, SSP3_MFA_input = calc_MFA('SSP3', lt_res, lt_com, lt_pub)
SSP4_dsm_res, SSP4_dsm_com, SSP4_dsm_pub, SSP4_MFA_input = calc_MFA('SSP4', lt_res, lt_com, lt_pub)
SSP5_dsm_res, SSP5_dsm_com, SSP5_dsm_pub, SSP5_MFA_input = calc_MFA('SSP5', lt_res, lt_com, lt_pub)


#
# # ----------------------------------------------------------------------------------------------------------------------

n_bins = 20
kde_flag = True
rug_flag = False
RECS_comparison = True
CBECS_comparison = False
plot_all = False

# Plot all RECS data against the DSM simulation distribution
if RECS_comparison==True:
    # number of bins for histogram comparisons
    # function to comput the desnity function of simualtion and RECS data

    def compare_RECS_and_plot(year=2015, index=196, RECS_series=RECS_Weights.Res_Weight_2015, plot=False, n_bins=20):
        age_in_year = np.full(index, year) - SSP1_dsm_res.t[0:index]

        # Create density for dsm values in 2015 (dsm_for_histogram)
        dsm_summary_df = pd.DataFrame({'year': SSP1_dsm_res.t[0:index],
                                       'area': SSP1_dsm_res.s_c[index][0:index].astype(int),
                                       'age': age_in_year[0:index]})
        dsm_summary_df['year'] = dsm_summary_df['year'].astype(int)
        dsm_summary_df['area'] = dsm_summary_df['area'].astype(int)
        dsm_summary_df['age'] = dsm_summary_df['age'].astype(int)
        dsm_for_histogram = []
        for index, row in dsm_summary_df.iterrows():
            age_reps = list(np.full(row['area'], row['age']))
            dsm_for_histogram.extend(age_reps)

        # Create desnity for RECS valeus in 2015
        RECS_summary_df = pd.DataFrame({'year': SSP1_dsm_res.t[0:index],
                                        'area': RECS_series[0:index] * SSP1_dsm_res.s_c[index][0:index].astype(int).sum(),
                                        'age': age_in_year[0:index]})
        RECS_summary_df['year'] = RECS_summary_df['year'].astype(int)
        RECS_summary_df['area'] = RECS_summary_df['area'].astype(int)
        RECS_summary_df['age'] = RECS_summary_df['age'].astype(int)
        RECS_for_histogram = []
        for index, row in RECS_summary_df.iterrows():
            age_reps = list(np.full(row['area'], row['age']))
            RECS_for_histogram.extend(age_reps)

        dsm_for_histogram = np.array(dsm_for_histogram)
        RECS_for_histogram = np.array(RECS_for_histogram)
        print(stats.describe(dsm_for_histogram))
        print(stats.describe(RECS_for_histogram))

        if plot == True:
            sns.distplot(dsm_for_histogram, kde=kde_flag, rug=rug_flag, color="r", bins=n_bins, label='DSM Simulation')
            sns.distplot(RECS_for_histogram, kde=kde_flag, rug=rug_flag, color="black", bins=n_bins,
                         label=str(year) + ' RECS')
            plt.legend();
            # plt.title('Residential Floor Age Structure ' + str(year))
            # plt.xlabel('Age')
            plt.show();

            # qq plot
            # Calculate quantiles
            dsm_for_histogram.sort()
            quantile_levels1 = np.arange(len(dsm_for_histogram), dtype=float) / len(dsm_for_histogram)

            RECS_for_histogram.sort()
            quantile_levels2 = np.arange(len(RECS_for_histogram), dtype=float) / len(RECS_for_histogram)
            # Use the smaller set of quantile levels to create the plot
            quantile_levels = quantile_levels2
            # We already have the set of quantiles for the smaller data set
            quantiles2 = RECS_for_histogram
            # We find the set of quantiles for the larger data set using linear interpolation
            quantiles1 = np.interp(quantile_levels, quantile_levels1, dsm_for_histogram)
            # Plot the quantiles to create the qq plot
            plt.plot(quantiles1, quantiles2)
            # Add a reference line
            maxval = max(dsm_for_histogram[-1], RECS_for_histogram[-1])
            minval = min(dsm_for_histogram[0], RECS_for_histogram[0])
            plt.plot([minval, maxval], [minval, maxval], 'k-')
            plt.xlabel('Simulation Quantiles')
            plt.ylabel('RECS Quantiles')
            plt.show();

            # plot cdfs next to one another
            plt.hist(quantiles1, bins=100, cumulative=True, alpha=0.8, histtype='step', label='Simulation', color='black')
            plt.hist(quantiles2, bins=100, cumulative=True, alpha=0.8, histtype='step', label=str(year) + ' RECS', color='red')
            plt.legend(loc=2);
            plt.show();
            # ks test
            print(stats.ks_2samp(dsm_for_histogram, RECS_for_histogram))





        return dsm_for_histogram, RECS_for_histogram

    if plot_all==True:
        # Compute the density functions for each
        DSM_age_2015, RECS_age_2015 = compare_RECS_and_plot(year=2015, index=196, RECS_series=RECS_Weights.Res_Weight_2015)
        DSM_age_2009, RECS_age_2009 = compare_RECS_and_plot(year=2009, index=190, RECS_series=RECS_Weights.Res_Weight_2009)
        DSM_age_2005, RECS_age_2005 = compare_RECS_and_plot(year=2005, index=186, RECS_series=RECS_Weights.Res_Weight_2005)
        DSM_age_2001, RECS_age_2001 = compare_RECS_and_plot(year=2001, index=182, RECS_series=RECS_Weights.Res_Weight_2001)
        DSM_age_1997, RECS_age_1997 = compare_RECS_and_plot(year=1997, index=178, RECS_series=RECS_Weights.Res_Weight_1997)
        DSM_age_1993, RECS_age_1993 = compare_RECS_and_plot(year=1993, index=174, RECS_series=RECS_Weights.Res_Weight_1993)
        DSM_age_1987, RECS_age_1987 = compare_RECS_and_plot(year=1987, index=168, RECS_series=RECS_Weights.Res_Weight_1987)
        DSM_age_1980, RECS_age_1980 = compare_RECS_and_plot(year=1980, index=161, RECS_series=RECS_Weights.Res_Weight_1980)

        # multiplot of each RECS year data against the data for that year in the DSM simulation
        plt.subplot(421)
        plt1 = sns.distplot(DSM_age_2015, kde=kde_flag, color="r", bins=10, label='DSM Simulation'),
        plt2 = sns.distplot(RECS_age_2015, kde=kde_flag, color="black", bins=10, label='2015 RECS'),
        # plt1 = plt.hist(DSM_age_2015, alpha=0.5, color="r", bins=n_bins, label='DSM Simulation')
        # plt2 = plt.hist(RECS_age_2015, alpha=0.5, color="black", bins=n_bins, label='2015 RECS')
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(422)
        plt1 = sns.distplot(DSM_age_2009, rug=rug_flag, kde=kde_flag, color="r", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(RECS_age_2009, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='2009 RECS')
        # plt1 = plt.hist(DSM_age_2009, alpha=0.5, color="r", bins=n_bins, label='DSM Simulation')
        # plt2 = plt.hist(RECS_age_2009, alpha=0.5, color="black", bins=n_bins, label='2009 RECS')
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(423)
        plt1 = sns.distplot(DSM_age_2005, rug=rug_flag, kde=kde_flag, color="r", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(RECS_age_2005, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='2005 RECS')
        # plt1 = plt.hist(DSM_age_2005, alpha=0.5, color="r", bins=n_bins, label='DSM Simulation')
        # plt2 = plt.hist(RECS_age_2005, alpha=0.5, color="black", bins=n_bins, label='2005 RECS')
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(424)
        plt1 = sns.distplot(DSM_age_2001, rug=rug_flag, kde=kde_flag, color="r", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(RECS_age_2001, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='2001 RECS')
        # plt1 = plt.hist(DSM_age_2001, alpha=0.5, color="r", bins=n_bins, label='DSM Simulation')
        # plt2 = plt.hist(RECS_age_2001, alpha=0.5, color="black", bins=n_bins, label='2001 RECS')
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(425)
        plt1 = sns.distplot(DSM_age_1997, rug=rug_flag, kde=kde_flag, color="r", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(RECS_age_1997, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='1997 RECS')
        # plt1 = plt.hist(DSM_age_1997, alpha=0.5, color="r", bins=n_bins, label='DSM Simulation')
        # plt2 = plt.hist(RECS_age_1997, alpha=0.5, color="black", bins=n_bins, label='1997 RECS')
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(426)
        plt1 = sns.distplot(DSM_age_1993, rug=rug_flag, kde=kde_flag, color="r", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(RECS_age_1993, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='1993 RECS')
        # plt1 = plt.hist(DSM_age_1993, alpha=0.5, color="r", bins=n_bins, label='DSM Simulation')
        # plt2 = plt.hist(RECS_age_1993, alpha=0.5, color="black", bins=n_bins, label='1993 RECS')
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(427)
        plt1 = sns.distplot(DSM_age_1987, rug=rug_flag, kde=kde_flag, color="r", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(RECS_age_1987, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='1987 RECS')
        # plt1 = plt.hist(DSM_age_1987, alpha=0.5, color="r", bins=n_bins, label='DSM Simulation')
        # plt2 = plt.hist(RECS_age_1987, alpha=0.5, color="black", bins=n_bins, label='1987 RECS')
        plt.legend(fontsize='x-small');
        plt.xlabel('Age')

        plt.subplot(428)
        plt1 = sns.distplot(DSM_age_1980, rug=rug_flag, kde=kde_flag, color="r", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(RECS_age_1980, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='1980 RECS')
        # plt1 = plt.hist(DSM_age_1980, alpha=0.5, color="r", bins=n_bins, label='DSM Simulation')
        # plt2 = plt.hist(RECS_age_1980, alpha=0.5, color="black", bins=n_bins, label='1980 RECS')
        plt.legend(fontsize='x-small');
        plt.xlabel('Age')
        plt.show();
    else:
        dsm_for_histogram, RECS_for_histogram = compare_RECS_and_plot(year=2015, index=196, RECS_series=RECS_Weights.Res_Weight_2015, plot=True, n_bins=n_bins)

# Plot all CBECS data against the DSM simulation distribution
if CBECS_comparison == True:

    # function to comput the desnity function of simualtion and CBECS data
    def compare_CBECS_and_plot(year=2012, index=193, CBECS_series=CBECS_Weights.Com_Weight_2012, plot=False, n_bins=20):
        age_in_year = np.full(index, year) - SSP1_dsm_com.t[0:index]

        # Create density for dsm values in 2015 (dsm_for_histogram)
        dsm_summary_df = pd.DataFrame({'year': SSP1_dsm_com.t[0:index],
                                       'area': SSP1_dsm_com.s_c[index][0:index].astype(int),
                                       'age': age_in_year[0:index]})
        dsm_summary_df['year'] = dsm_summary_df['year'].astype(int)
        dsm_summary_df['area'] = dsm_summary_df['area'].astype(int)
        dsm_summary_df['age'] = dsm_summary_df['age'].astype(int)

        dsm_for_histogram = []
        for index, row in dsm_summary_df.iterrows():
            age_reps = list(np.full(row['area'], row['age']))
            dsm_for_histogram.extend(age_reps)

        # Create desnity for CBECS valeus in 2015
        CBECS_summary_df = pd.DataFrame({'year': SSP1_dsm_com.t[0:index],
                                        'area': CBECS_series[0:index] * SSP1_dsm_com.s_c[index][0:index].sum(),
                                        'age': age_in_year[0:index]})
        CBECS_summary_df['year'] = CBECS_summary_df['year'].astype(int)
        CBECS_summary_df['area'] = CBECS_summary_df['area'].astype(int)
        CBECS_summary_df['age'] = CBECS_summary_df['age'].astype(int)

        CBECS_for_histogram = []
        for index, row in CBECS_summary_df.iterrows():
            age_reps = list(np.full(row['area'], row['age']))
            CBECS_for_histogram.extend(age_reps)

        stats.describe(dsm_for_histogram)
        stats.describe(CBECS_for_histogram)

        if plot == True:
            sns.distplot(dsm_for_histogram, kde=kde_flag, rug=rug_flag, color="blue", bins=n_bins, label='DSM Simulation')
            sns.distplot(CBECS_for_histogram, kde=kde_flag, rug=rug_flag, color="black", bins=n_bins,
                         label=str(year) + ' CBECS')
            plt.legend();
            # plt.title('Residential Floor Age Structure ' + str(year))
            # plt.xlabel('Age')
            plt.show();

            # qq plot
            # Calculate quantiles
            dsm_for_histogram.sort()
            quantile_levels1 = np.arange(len(dsm_for_histogram), dtype=float) / len(dsm_for_histogram)

            CBECS_for_histogram.sort()
            quantile_levels2 = np.arange(len(CBECS_for_histogram), dtype=float) / len(CBECS_for_histogram)
            # Use the smaller set of quantile levels to create the plot
            quantile_levels = quantile_levels2
            # We already have the set of quantiles for the smaller data set
            quantiles2 = CBECS_for_histogram
            # We find the set of quantiles for the larger data set using linear interpolation
            quantiles1 = np.interp(quantile_levels, quantile_levels1, dsm_for_histogram)
            # Plot the quantiles to create the qq plot
            plt.plot(quantiles1, quantiles2)
            # Add a reference line
            maxval = max(dsm_for_histogram[-1], CBECS_for_histogram[-1])
            minval = min(dsm_for_histogram[0], CBECS_for_histogram[0])
            plt.plot([minval, maxval], [minval, maxval], 'k-')
            plt.xlabel('Simulation Quantiles')
            plt.ylabel('CBECS Quantiles')
            plt.show();

            # plot cdfs next to one another
            plt.hist(quantiles1, bins=100, cumulative=True, alpha=0.8, histtype='step', label='Simulation', color='black')
            plt.hist(quantiles2, bins=100, cumulative=True, alpha=0.8, histtype='step', label=str(year) + ' CBECS', color='blue')
            plt.legend(loc=2);
            plt.show();
            # ks test
            print(stats.ks_2samp(dsm_for_histogram, CBECS_for_histogram))


        return dsm_for_histogram, CBECS_for_histogram

    # def compare_CBECS(year=2012, index=193, CBECS_series=CBECS_Weights.Com_Weight_2012, plot=True, n_bins=10):
    #     age_in_year = np.full(index, year) - SSP1_dsm_res.t[0:index]
    #     computed_distr_year = SSP1_dsm_com.s_c[index][0:index] / SSP1_dsm_com.s_c[index][0:index].sum()
    #     calc_age_weighted_year = age_in_year * computed_distr_year * (index - 1)
    #     S_0_com_year = age_in_year * (CBECS_series[0:index]) * (index - 1)
    #     # sns.distplot(calc_age_weighted_year, color="r", bins=n_bins, label='DSM Simulation')
    #     # sns.distplot(S_0_com_year, color="black", bins=n_bins, label=str(year)+' CBECS')
    #     # plt.legend();
    #     # plt.title('Residential Floor Area Age Distribution in ' + str(year))
    #     # plt.xlabel('Age')
    #     # plt.show();
    #
    #     return calc_age_weighted_year, S_0_com_year

    if plot_all==True:
        # Compute the density functions for each
        DSM_age_2012, CBECS_age_2012 = compare_CBECS_and_plot(year=2012, index=193, CBECS_series=CBECS_Weights.Com_Weight_2012)
        DSM_age_2003, CBECS_age_2003 = compare_CBECS_and_plot(year=2003, index=184, CBECS_series=CBECS_Weights.Com_Weight_2003)
        DSM_age_1999, CBECS_age_1999 = compare_CBECS_and_plot(year=1999, index=180, CBECS_series=CBECS_Weights.Com_Weight_1999)
        DSM_age_1995, CBECS_age_1995 = compare_CBECS_and_plot(year=1995, index=176, CBECS_series=CBECS_Weights.Com_Weight_1995)
        DSM_age_1992, CBECS_age_1992 = compare_CBECS_and_plot(year=1992, index=173, CBECS_series=CBECS_Weights.Com_Weight_1992)
        DSM_age_1986, CBECS_age_1986 = compare_CBECS_and_plot(year=1986, index=167, CBECS_series=CBECS_Weights.Com_Weight_1986)
        DSM_age_1983, CBECS_age_1983 = compare_CBECS_and_plot(year=1983, index=164, CBECS_series=CBECS_Weights.Com_Weight_1983)
        DSM_age_1979, CBECS_age_1979 = compare_CBECS_and_plot(year=1979, index=160, CBECS_series=CBECS_Weights.Com_Weight_1979)

        # multiplot of each RECS year data against the data for that year in the DSM simulation
        plt.subplot(421)
        plt1 = sns.distplot(DSM_age_2012, rug=rug_flag, kde=kde_flag, color="blue", bins=10, label='DSM Simulation'),
        plt2 = sns.distplot(CBECS_age_2012, rug=rug_flag, kde=kde_flag, color="black", bins=10, label='2012 CBECS'),
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(422)
        plt1 = sns.distplot(DSM_age_2003, rug=rug_flag, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(CBECS_age_2003, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='2003 CBECS')
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(423)
        plt1 = sns.distplot(DSM_age_1999, rug=rug_flag, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(CBECS_age_1999, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='1999 CBECS')
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(424)
        plt1 = sns.distplot(DSM_age_1995, rug=rug_flag, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(CBECS_age_1995, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='1995 CBECS')
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(425)
        plt1 = sns.distplot(DSM_age_1992, rug=rug_flag, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(CBECS_age_1992, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='1992 CBECS')
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(426)
        plt1 = sns.distplot(DSM_age_1986, rug=rug_flag, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(CBECS_age_1986, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='1986 CBECS')
        plt.legend(fontsize='x-small');
        plt.xlabel(None)

        plt.subplot(427)
        plt1 = sns.distplot(DSM_age_1983, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(CBECS_age_1983, kde=kde_flag, color="black", bins=n_bins, label='1983 CBECS')
        plt.legend(fontsize='x-small');
        plt.xlabel('Age')

        plt.subplot(428)
        plt1 = sns.distplot(DSM_age_1979, rug=rug_flag, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation')
        plt2 = sns.distplot(CBECS_age_1979, rug=rug_flag, kde=kde_flag, color="black", bins=n_bins, label='1979 CBECS')
        plt.legend(fontsize='x-small');
        plt.xlabel('Age')
        plt.show();
    else:
        compare_CBECS_and_plot(year=2012, index=193, CBECS_series=CBECS_Weights.Com_Weight_2012, plot=True, n_bins=n_bins)

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
plot_MFA_all_scenarios = False
if plot_MFA_all_scenarios==True:
    plot_dsm(SSP1_dsm_res, 'SSP1 Residential')
    plot_dsm(SSP1_dsm_com, 'SSP1 Commercial')
    plot_dsm(SSP1_dsm_pub, 'SSP1 Public')
    plot_dsm(SSP2_dsm_res, 'SSP2 Residential')
    plot_dsm(SSP2_dsm_com, 'SSP2 Commercial')
    plot_dsm(SSP2_dsm_pub, 'SSP2 Public')
    plot_dsm(SSP3_dsm_res, 'SSP3 Residential')
    plot_dsm(SSP3_dsm_com, 'SSP3 Commercial')
    plot_dsm(SSP3_dsm_pub, 'SSP3 Public')
    plot_dsm(SSP4_dsm_res, 'SSP4 Residential')
    plot_dsm(SSP4_dsm_com, 'SSP4 Commercial')
    plot_dsm(SSP4_dsm_pub, 'SSP4 Public')
    plot_dsm(SSP5_dsm_res, 'SSP5 Residential')
    plot_dsm(SSP5_dsm_com, 'SSP5 Commercial')
    plot_dsm(SSP5_dsm_pub, 'SSP5 Public')
else:
    plot_dsm(SSP1_dsm_res, 'SSP1 Residential')

# # ----------------------------------------------------------------------------------------------------------------------
# # Plot all scenarios together for residential buildings
plot_MFA_all_same_graph = False
no_SSP5 = True      # True for ignoring SSP5, False for including SSP5
if plot_MFA_all_same_graph == True:
    plt.subplot(211)
    plt1, = plt.plot(SSP1_dsm_res.t, SSP1_dsm_res.s)
    plt2, = plt.plot(SSP2_dsm_res.t, SSP2_dsm_res.s)
    plt3, = plt.plot(SSP3_dsm_res.t, SSP3_dsm_res.s)
    plt4, = plt.plot(SSP4_dsm_res.t, SSP4_dsm_res.s)
    plt16, = plt.plot([base_year, base_year], [0, 100000], color='k', LineStyle='--')
    if no_SSP5 == True:
        temp = 'bleh'
    else:
        plt5, = plt.plot(SSP5_dsm_res.t, SSP5_dsm_res.s)
    if no_SSP5 == True:
        plt.legend([plt1, plt2, plt3, plt4], ['SSP1', 'SSP2', 'SSP3', 'SSP4'], loc=(1.05, 0.5))
    else:
        plt.legend([plt1, plt2, plt3, plt4, plt5], ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], loc=(1.05, 0.5))
    # plt.legend([plt1, plt2, plt3, plt4, plt5], ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], loc=(1.05, 0.5))
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
    if no_SSP5 == True:
        temp = 'bleh'
    else:
        plt9, = plt.plot(SSP5_dsm_res.t, SSP5_dsm_res.i, LineStyle='dashed')
        plt0, = plt.plot(SSP5_dsm_res.t, SSP5_dsm_res.o)

    plt11, = plt.plot([base_year, base_year], [0, 4000], color='k', LineStyle='--')

    if no_SSP5 == True:
        plt.legend([plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8],
                   ['Inflow SSP1', 'Outflow SSP1',
                    'Inflow SSP2', 'Outflow SSP2',
                    'Inflow SSP3', 'Outflow SSP3',
                    'Inflow SSP4', 'Outflow SSP4'], loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend([plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9, plt0],
                   ['Inflow SSP1', 'Outflow SSP1',
                    'Inflow SSP2', 'Outflow SSP2',
                    'Inflow SSP3', 'Outflow SSP3',
                    'Inflow SSP4', 'Outflow SSP4',
                    'Inflow SSP5', 'Outflow SSP5'], loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.ylim(top=5000)
    plt.xlim(left=SSP1_dsm_res.t[0] + 5)
    plt.xlabel('Year')
    plt.ylabel('Floor Area per year')
    plt.title('Residential Floor Area flows')
    plt.show();


# ----------------------------------------------------------------------------------------------------------------------
# Derive distributions from CBECS and RECS data (compare against other studies distributions)
# THIS CODE DOESN"T TELL US ANYTHING USEFUL!
fit_distribution_to_age_structure = False
if plot_all==True:
    if fit_distribution_to_age_structure==True:
        # Residential
        # function to estimate the parameters for different lifetime distributions
        def determine_lt_params_chi2(y=RECS_age_2015, n_bins=100, p_q_plots=True):
            # plot initial histogram:
            plt.hist(y, bins=n_bins)
            plt.show();

            x = np.arange(len(y))
            size = len(y)
            # center the data
            y = np.array(y)
            sc = StandardScaler()
            yy = y.reshape(-1, 1)
            sc.fit(yy)
            y_std = sc.transform(yy)
            y_std = y_std.flatten()
            y_std
            del yy

            dist_names = ['weibull_min']
            # dist_names = ['expon','gamma','lognorm','norm','weibull_max', 'exponweib']
            # dist_names = [x for x in dist_names_all if x not in distributions_excluded]

            # Set up empty lists to store results
            chi_square = []
            p_values = []

            # Set up 50 bins for chi-square test
            # Observed data will be approximately evenly distrubuted aross all bins
            percentile_bins = np.linspace(0, 100, 51)
            percentile_cutoffs = np.percentile(y_std, percentile_bins)
            observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
            cum_observed_frequency = np.cumsum(observed_frequency)

            # Loop through candidate distributions
            for distribution in dist_names:
                # Set up distribution and get fitted distribution parameters
                dist = getattr(stats, distribution)
                param = dist.fit(y_std)

                # Obtain the KS test P statistic, round it to 5 decimal places
                p = stats.kstest(y_std, distribution, args=param)[1]
                # p = np.around(p, 5)
                p_values.append(p)

                # Get expected counts in percentile bins
                # This is based on a 'cumulative distrubution function' (cdf)
                cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2],
                                      scale=param[-1])
                expected_frequency = []
                for bin in range(len(percentile_bins) - 1):
                    expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
                    expected_frequency.append(expected_cdf_area)

                # calculate chi-squared
                expected_frequency = np.array(expected_frequency) * (size)
                cum_expected_frequency = np.cumsum(expected_frequency)
                ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
                chi_square.append(ss)

            # Collate results and sort by goodness of fit (best at top)
            results = pd.DataFrame()
            results['Distribution'] = dist_names
            results['chi_square'] = chi_square
            results['p_value'] = p_values
            results.sort_values(['chi_square'], inplace=True)

            # Report results
            print('\nDistributions sorted by goodness of fit:')
            print('----------------------------------------')
            print(results)

            # Divide the observed data into 100 bins for plotting (this can be changed)
            number_of_bins = n_bins
            # bin_cutoffs = np.linspace(np.percentile(y, 0), np.percentile(y, 99), number_of_bins)

            # Create the plot
            h = plt.hist(y, bins=n_bins, color='0.75')

            # Get the top three distributions from the previous phase
            number_distributions_to_plot = 5
            dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]

            # Create an empty list to stroe fitted distribution parameters
            parameters = []

            # Loop through the distributions ot get line fit and paraemters
            for dist_name in dist_names:
                # Set up distribution and store distribution paraemters
                dist = getattr(stats, dist_name)
                param = dist.fit(y)
                parameters.append(param)

                # Get line for each distribution (and scale to match observed data)
                pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
                scale_pdf = np.trapz(h[0], h[1][:-1]) / np.trapz(pdf_fitted, x)
                pdf_fitted *= scale_pdf

                # Add the line to the plot
                plt.plot(pdf_fitted, label=dist_name)

                # Set the plot x axis to contain 99% of the data
                # This can be removed, but sometimes outlier data makes the plot less clear
                plt.xlim(0, np.percentile(y, 99))

            # Add legend and display plot

            plt.legend()
            plt.show()

            # Store distribution paraemters in a dataframe (this could also be saved)
            dist_parameters = pd.DataFrame()
            dist_parameters['Distribution'] = (
                results['Distribution'].iloc[0:number_distributions_to_plot])
            dist_parameters['Distribution parameters'] = parameters

            # Print parameter results
            print('\nDistribution parameters:')
            print('------------------------')

            for index, row in dist_parameters.iterrows():
                print('\nDistribution:', row[0])
                print('Parameters:', row[1])
            if p_q_plots==True:
                ## qq and pp plots
                data = y_std.copy()
                data.sort()
                # Loop through selected distributions (as previously selected)
                for distribution in dist_names:
                    # Set up distribution
                    dist = getattr(stats, distribution)
                    param = dist.fit(y_std)

                    # Get random numbers from distribution
                    norm = dist.rvs(*param[0:-2], loc=param[-2], scale=param[-1], size=size)
                    norm.sort()

                    # Create figure
                    fig = plt.figure(figsize=(8, 5))

                    # qq plot
                    ax1 = fig.add_subplot(121)  # Grid of 2x2, this is suplot 1
                    ax1.plot(norm, data, "o")
                    min_value = np.floor(min(min(norm), min(data)))
                    max_value = np.ceil(max(max(norm), max(data)))
                    ax1.plot([min_value, max_value], [min_value, max_value], 'r--')
                    ax1.set_xlim(min_value, max_value)
                    ax1.set_xlabel('Theoretical quantiles')
                    ax1.set_ylabel('Observed quantiles')
                    title = 'qq plot for ' + distribution + ' distribution'
                    ax1.set_title(title)

                    # pp plot
                    ax2 = fig.add_subplot(122)

                    # Calculate cumulative distributions
                    bins = np.percentile(norm, range(0, 101))
                    data_counts, bins = np.histogram(data, bins)
                    norm_counts, bins = np.histogram(norm, bins)
                    cum_data = np.cumsum(data_counts)
                    cum_norm = np.cumsum(norm_counts)
                    cum_data = cum_data / max(cum_data)
                    cum_norm = cum_norm / max(cum_norm)

                    # plot
                    ax2.plot(cum_norm, cum_data, "o")
                    min_value = np.floor(min(min(cum_norm), min(cum_data)))
                    max_value = np.ceil(max(max(cum_norm), max(cum_data)))
                    ax2.plot([min_value, max_value], [min_value, max_value], 'r--')
                    ax2.set_xlim(min_value, max_value)
                    ax2.set_xlabel('Theoretical cumulative distribution')
                    ax2.set_ylabel('Observed cumulative distribution')
                    title = 'pp plot for ' + distribution + ' distribution'
                    ax2.set_title(title)

                    # Display plot
                    plt.tight_layout(pad=4)
                    plt.show()

        determine_lt_params_chi2(y=RECS_age_2015, n_bins=100, p_q_plots=False)

        def determine_age_distribution(data=RECS_age_2015, plot=True, name='RECS 2015', return_normal=True):
            data = np.array(data)
            plt.hist(data, bins=50, density=True, alpha=0.5)
            shape, loc, scale = stats.weibull_min.fit(data, floc=0)
            mean, stddev = stats.norm.fit(data)

            if plot==True:
                plt.plot(x, stats.weibull_min(shape, loc, scale).pdf(x), label='Weibull_min '+ name)
                plt.plot(x, stats.norm(mean, stddev).pdf(x), label='Normal ' + name)
                plt.legend();
                plt.xlabel("Age")
                plt.show();
                print('Weibull lifetime parameters are: ')
                print('shape = ', shape, 'loc = ', loc, 'scale = ', scale)
                print('Normal lifetime parameters are: ')
                print('mean = ', mean, 'std dev = ', stddev)

                # Plot the CDF
                ecdf = statsmodels.distributions.ECDF(data)
                plt.plot(x, ecdf(x), label='Empirical CDF')
                plt.plot(x, stats.weibull_min(shape, loc, scale).cdf(x), label='Weibull_min ' + name)
                plt.plot(x, stats.norm(mean, stddev).cdf(x), label='Normal ' + name)
                plt.xlabel('Age')
                plt.title('CDF comparison')
                plt.legend()
                plt.show();

                plt.subplot(211)
                stats.probplot(data, dist=stats.weibull_min(shape, loc, scale), plot=plt)
                plt.title('Weibull QQ-plot')
                plt.xlim(right=100)
                # plt.show();
                plt.subplot(212)
                stats.probplot(data, dist=stats.norm(mean, stddev), plot=plt)
                plt.title('Normal QQ-plot')
                plt.xlim(right=100, left=0)
                plt.show();
            if return_normal==True:
                return shape, loc, scale, mean, stddev
            else:
                return shape, scale


        determine_age_distribution(data=RECS_age_2015, plot=True, return_normal=True)

        shape_r2015, scale_r2015 = determine_age_distribution(data=RECS_age_2015, plot=False, return_normal=False)
        shape_r2009, scale_r2009 = determine_age_distribution(data=RECS_age_2009, plot=False, return_normal=False)
        shape_r2005, scale_r2005 = determine_age_distribution(data=RECS_age_2005, plot=False, return_normal=False)
        shape_r2001, scale_r2001 = determine_age_distribution(data=RECS_age_2001, plot=False, return_normal=False)
        shape_r1997, scale_r1997 = determine_age_distribution(data=RECS_age_1997, plot=False, return_normal=False)
        shape_r1993, scale_r1993 = determine_age_distribution(data=RECS_age_1993, plot=False, return_normal=False)
        shape_r1987, scale_r1987 = determine_age_distribution(data=RECS_age_1987, plot=False, return_normal=False)
        shape_r1980, scale_r1980 = determine_age_distribution(data=RECS_age_1980, plot=False, return_normal=False)

        np.mean([shape_r2015, shape_r2009, shape_r2005, shape_r2001, shape_r1997, shape_r1993, shape_r1987, shape_r1980])
        np.std([shape_r2015, shape_r2009, shape_r2005, shape_r2001, shape_r1997, shape_r1993, shape_r1987, shape_r1980])
        np.mean([scale_r2015, scale_r2009, scale_r2005, scale_r2001, scale_r1997, scale_r1993, scale_r1987, scale_r1980])
        np.std([scale_r2015, scale_r2009, scale_r2005, scale_r2001, scale_r1997, scale_r1993, scale_r1987, scale_r1980])
