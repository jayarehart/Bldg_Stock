# Dynamic stock model (DSM) for US building stock

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from odym import dynamic_stock_model as dsm
from scipy import stats

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
                       base_year=2016,FA_base_year=246, Area_country=9.14759e6, gamma=-0.03,
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
                       base_year=2016,FA_base_year=246, Area_country=8081867, gamma=-0.03,
                       plot=True)      # area of continguous 48 = 8081867, area of all = 9833517


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
    plt.title(plot_name + ' Floor Space - Flow')
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

# A function to generate lifetime distributions for use in the ODYM dsm package.
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

lt_res = generate_lt('Weibull',par1=5.5, par2=85.8)
lt_com = generate_lt('Weibull',par1=4.8, par2=75.1)
lt_pub = generate_lt('Weibull',par1=6.1, par2=95.6)

# Debugging
# lt_res = generate_lt('Weibull',par1=5, par2=100)
# lt_com = generate_lt('Weibull',par1=5, par2=100)
# lt_pub = generate_lt('Weibull',par1=5, par2=100)

# Plot lifetime distributions:
plot_lifetime_distr=False
if plot_lifetime_distr==True:
    x = np.arange(1,200)
    def weib(x,n,a):
        return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

    # count, bins, ignored = plt.hist(np.random.weibull(5.5,1000))
    # scale = count.max()/weib(x, 85.8, 5.5).max()
    plt1, = plt.plot(x, weib(x, lt_res['Scale'][0], lt_res['Shape'][0])*lt_res['Scale'][0], label='Weibull Residential')
    plt2, = plt.plot(x, weib(x, lt_com['Scale'][0], lt_com['Shape'][0])*lt_com['Scale'][0], label='Weibull Commercial')
    plt3, = plt.plot(x, weib(x, lt_pub['Scale'][0], lt_pub['Shape'][0])*lt_pub['Scale'][0], label='Weibull Public')
    plt.legend(loc=2)
    plt.title('Input Weibull Distributions')
    plt.xlabel('Building lifespan')
    plt.show()

    # mess around
    lt1 = generate_lt('Weibull', par1=8, par2=100)
    lt2 = generate_lt('Weibull', par1=6, par2=100)
    lt3 = generate_lt('Weibull', par1=3, par2=100)
    lt4 = generate_lt('Weibull', par1=5, par2=60)
    lt5 = generate_lt('Weibull', par1=5, par2=80)
    lt6 = generate_lt('Weibull', par1=5, par2=100)
    lt7 = generate_lt('Weibull', par1=5, par2=120)

    plt1, = plt.plot(x, weib(x, lt1['Scale'][0], lt1['Shape'][0]) * lt1['Scale'][0], label='weibull(8, 100)', linestyle='--')
    plt2, = plt.plot(x, weib(x, lt2['Scale'][0], lt2['Shape'][0]) * lt2['Scale'][0], label='weibull(6, 100)', linestyle='--')
    plt3, = plt.plot(x, weib(x, lt3['Scale'][0], lt3['Shape'][0]) * lt3['Scale'][0], label='weibull(3, 100)', linestyle='--')
    plt4, = plt.plot(x, weib(x, lt4['Scale'][0], lt4['Shape'][0]) * lt4['Scale'][0], label='weibull(5, 60)')
    plt5, = plt.plot(x, weib(x, lt5['Scale'][0], lt5['Shape'][0]) * lt5['Scale'][0], label='weibull(5, 80)')
    plt6, = plt.plot(x, weib(x, lt6['Scale'][0], lt6['Shape'][0]) * lt6['Scale'][0], label='weibull(5, 100)')
    plt7, = plt.plot(x, weib(x, lt7['Scale'][0], lt7['Shape'][0]) * lt7['Scale'][0], label='weibull(5, 120)')

    plt.legend(loc=2)
    plt.title('Input Weibull Distributions')
    plt.xlabel('Building lifespan')
    plt.show()

# function to calculate the dynamic stock.
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
# Inputs for plotting the results of the dynamic stock model simulations and the DOE data
n_bins = 15
kde_flag = True
rug_flag = False
RECS_comparison = True
CBECS_comparison = True
plot_all = True

# Plot all RECS data against the DSM simulation distribution
if RECS_comparison==True:
    # number of bins for histogram comparisons
    # function to compute the density function of simulation and RECS data

    def compare_RECS_and_plot(year=2015, index_end=196, RECS_series=RECS_Weights.Res_Weight_2015, plot=False, n_bins=20):
        index_start = 100    # year to start comparison of histograms
        age_in_year = np.full(index_end-index_start, year) - SSP1_dsm_res.t[index_start:index_end]

        # Create density for dsm values in 2015 (dsm_for_histogram)
        dsm_summary_df = pd.DataFrame({'year': SSP1_dsm_res.t[index_start:index_end],
                                       'area': SSP1_dsm_res.s_c[index_end][index_start:index_end].astype(int),
                                       'age': age_in_year})
        dsm_summary_df['year'] = dsm_summary_df['year'].astype(int)
        dsm_summary_df['area'] = dsm_summary_df['area'].astype(int)
        dsm_summary_df['age'] = dsm_summary_df['age'].astype(int)
        dsm_for_histogram = []
        for index, row in dsm_summary_df.iterrows():
            age_reps = list(np.full(row['area'], row['age']))
            dsm_for_histogram.extend(age_reps)

        # Create desnity for RECS valeus in 2015
        RECS_summary_df = pd.DataFrame({'year': SSP1_dsm_res.t[index_start:index_end],
                                        'area': RECS_series[index_start:index_end] * SSP1_dsm_res.s_c[index_end][index_start:index_end].astype(int).sum(),
                                        'age': age_in_year})
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

        # ks test
        ks = (stats.ks_2samp(dsm_for_histogram, RECS_for_histogram))
        print(ks)

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

        if plot == True:
            sns.distplot(dsm_for_histogram, kde=kde_flag, rug=rug_flag, color="r", bins=n_bins, label='DSM Simulation')
            sns.distplot(RECS_for_histogram, kde=kde_flag, rug=rug_flag, color="black", bins=n_bins,
                         label=str(year) + ' RECS')
            plt.legend();
            # plt.title('Residential Floor Age Structure ' + str(year))
            # plt.xlabel('Age')
            plt.show();

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

        return dsm_for_histogram, RECS_for_histogram, quantiles1, quantiles2, ks

    if plot_all==True:
        # Compute the density functions for each
        DSM_age_2015, RECS_age_2015, q1_2015, q2_2015, ks_2015 = compare_RECS_and_plot(year=2015, index_end=196, RECS_series=RECS_Weights.Res_Weight_2015)
        DSM_age_2009, RECS_age_2009, q1_2009, q2_2009, ks_2009 = compare_RECS_and_plot(year=2009, index_end=190, RECS_series=RECS_Weights.Res_Weight_2009)
        DSM_age_2005, RECS_age_2005, q1_2005, q2_2005, ks_2005 = compare_RECS_and_plot(year=2005, index_end=186, RECS_series=RECS_Weights.Res_Weight_2005)
        DSM_age_2001, RECS_age_2001, q1_2001, q2_2001, ks_2001 = compare_RECS_and_plot(year=2001, index_end=181, RECS_series=RECS_Weights.Res_Weight_2001)
        DSM_age_1997, RECS_age_1997, q1_1997, q2_1997, ks_1997 = compare_RECS_and_plot(year=1997, index_end=178, RECS_series=RECS_Weights.Res_Weight_1997)
        DSM_age_1993, RECS_age_1993, q1_1993, q2_1993, ks_1993 = compare_RECS_and_plot(year=1993, index_end=174, RECS_series=RECS_Weights.Res_Weight_1993)
        DSM_age_1987, RECS_age_1987, q1_1987, q2_1987, ks_1987 = compare_RECS_and_plot(year=1987, index_end=168, RECS_series=RECS_Weights.Res_Weight_1987)
        DSM_age_1980, RECS_age_1980, q1_1980, q2_1980, ks_1980 = compare_RECS_and_plot(year=1980, index_end=161, RECS_series=RECS_Weights.Res_Weight_1980)

        print('mean ks values for each is = ', str(np.round(np.mean([ks_2015, ks_2009, ks_2005, ks_2001, ks_1997, ks_1993, ks_1987, ks_1980]),5)))
        # multiplot of each RECS year data against the data for that year in the DSM simulation
        fig = plt.figure(figsize=(16,8))
        axes1 = fig.add_subplot(241)
        sns.distplot(DSM_age_2015, kde=kde_flag, color="red", bins=n_bins, label='DSM Simulation'),
        sns.distplot(RECS_age_2015, kde=kde_flag, color="black", bins=n_bins, label='2015 RECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_2015, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation', color='gray')
        plt.hist(q2_2015, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='2015 RECS',color='darksalmon')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_2015[0],4)), ha='right', va='top', transform=axes1.transAxes, fontsize='x-small')
        # plt.show()

        axes1 = fig.add_subplot(242)
        sns.distplot(DSM_age_2009, kde=kde_flag, color="red", bins=n_bins, label='DSM Simulation'),
        sns.distplot(RECS_age_2009, kde=kde_flag, color="black", bins=n_bins, label='2009 RECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_2009, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation', color='gray')
        plt.hist(q2_2009, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='2009 RECS',color='darksalmon')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_2009[0],4)), ha='right', va='top', transform=axes1.transAxes, fontsize='x-small')


        axes1 = fig.add_subplot(243)
        sns.distplot(DSM_age_2005, kde=kde_flag, color="red", bins=n_bins, label='DSM Simulation'),
        sns.distplot(RECS_age_2005, kde=kde_flag, color="black", bins=n_bins, label='2005 RECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_2005, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation', color='gray')
        plt.hist(q2_2005, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='2005 RECS',color='darksalmon')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_2005[0],4)), ha='right', va='top', transform=axes1.transAxes, fontsize='x-small')



        axes1 = fig.add_subplot(244)
        sns.distplot(DSM_age_2001, kde=kde_flag, color="red", bins=n_bins, label='DSM Simulation'),
        sns.distplot(RECS_age_2001, kde=kde_flag, color="black", bins=n_bins, label='2001 RECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_2001, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation', color='gray')
        plt.hist(q2_2001, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='2001 RECS',color='darksalmon')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_2001[0],4)), ha='right', va='top', transform=axes1.transAxes, fontsize='x-small')


        axes1 = fig.add_subplot(245)
        sns.distplot(DSM_age_1997, kde=kde_flag, color="red", bins=n_bins, label='DSM Simulation'),
        sns.distplot(RECS_age_1997, kde=kde_flag, color="black", bins=n_bins, label='1997 RECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_1997, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation', color='gray')
        plt.hist(q2_1997, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='1997 RECS',color='darksalmon')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_1997[0],4)), ha='right', va='top', transform=axes1.transAxes, fontsize='x-small')

        axes1 = fig.add_subplot(246)
        sns.distplot(DSM_age_1993, kde=kde_flag, color="red", bins=n_bins, label='DSM Simulation'),
        sns.distplot(RECS_age_1993, kde=kde_flag, color="black", bins=n_bins, label='1993 RECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_1993, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation', color='gray')
        plt.hist(q2_1993, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='1993 RECS',color='darksalmon')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_1993[0],4)), ha='right', va='top', transform=axes1.transAxes, fontsize='x-small')

        axes1 = fig.add_subplot(247)
        sns.distplot(DSM_age_1987, kde=kde_flag, color="red", bins=n_bins, label='DSM Simulation'),
        sns.distplot(RECS_age_1987, kde=kde_flag, color="black", bins=n_bins, label='1987 RECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_1987, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation', color='gray')
        plt.hist(q2_1987, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='1987 RECS',color='darksalmon')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel('Age')
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_1987[0],4)), ha='right', va='top', transform=axes1.transAxes, fontsize='x-small')

        axes1 = fig.add_subplot(248)
        sns.distplot(DSM_age_1980, kde=kde_flag, color="red", bins=n_bins, label='DSM Simulation'),
        sns.distplot(RECS_age_1980, kde=kde_flag, color="black", bins=n_bins, label='1980 RECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_1980, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation', color='gray')
        plt.hist(q2_1980, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='1980 RECS',color='darksalmon')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel('Age')
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_1980[0],4)), ha='right', va='top', transform=axes1.transAxes, fontsize='x-small')
        plt.show();

    else:
        compare_RECS_and_plot(year=2015, index_end=196, RECS_series=RECS_Weights.Res_Weight_2015, plot=True, n_bins=n_bins)

# Plot all CBECS data against the DSM simulation distribution
if CBECS_comparison == True:

    # function to comput the desnity function of simualtion and CBECS data
    def compare_CBECS_and_plot(year=2012, index_end=193, CBECS_series=CBECS_Weights.Com_Weight_2012, plot=False, n_bins=20):
        index_start = 100    # year to start comparison of histograms
        age_in_year = np.full(index_end-index_start, year) - SSP1_dsm_com.t[index_start:index_end]

        # Create density for dsm values in 2015 (dsm_for_histogram)
        dsm_summary_df = pd.DataFrame({'year': SSP1_dsm_com.t[index_start:index_end],
                                       'area': SSP1_dsm_com.s_c[index_end][index_start:index_end].astype(int),
                                       'age': age_in_year})
        dsm_summary_df['year'] = dsm_summary_df['year'].astype(int)
        dsm_summary_df['area'] = dsm_summary_df['area'].astype(int)
        dsm_summary_df['age'] = dsm_summary_df['age'].astype(int)

        dsm_for_histogram = []
        for index, row in dsm_summary_df.iterrows():
            age_reps = list(np.full(row['area'], row['age']))
            dsm_for_histogram.extend(age_reps)

        # Create desnity for CBECS valeus in 2015
        CBECS_summary_df = pd.DataFrame({'year': SSP1_dsm_com.t[index_start:index_end],
                                        'area': CBECS_series[index_start:index_end] * SSP1_dsm_com.s_c[index_end][index_start:index_end].sum(),
                                        'age': age_in_year})
        CBECS_summary_df['year'] = CBECS_summary_df['year'].astype(int)
        CBECS_summary_df['area'] = CBECS_summary_df['area'].astype(int)
        CBECS_summary_df['age'] = CBECS_summary_df['age'].astype(int)
        CBECS_for_histogram = []
        for index, row in CBECS_summary_df.iterrows():
            age_reps = list(np.full(row['area'], row['age']))
            CBECS_for_histogram.extend(age_reps)

        dsm_for_histogram = np.array(dsm_for_histogram)
        CBECS_for_histogram = np.array(CBECS_for_histogram)
        print(stats.describe(dsm_for_histogram))
        print(stats.describe(CBECS_for_histogram))

        # ks test
        ks = (stats.ks_2samp(dsm_for_histogram, CBECS_for_histogram))
        print(ks)

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

        if plot == True:
            sns.distplot(dsm_for_histogram, kde=kde_flag, rug=rug_flag, color="blue", bins=n_bins, label='DSM Simulation')
            sns.distplot(CBECS_for_histogram, kde=kde_flag, rug=rug_flag, color="black", bins=n_bins,
                         label=str(year) + ' CBECS')
            plt.legend();
            # plt.title('Residential Floor Age Structure ' + str(year))
            # plt.xlabel('Age')
            plt.show();

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

        return dsm_for_histogram, CBECS_for_histogram, quantiles1, quantiles2, ks

    if plot_all==True:
        # Compute the density functions for each
        DSM_age_2012, CBECS_age_2012, q1_2012, q2_2012, ks_2012 = compare_CBECS_and_plot(year=2012, index_end=193, CBECS_series=CBECS_Weights.Com_Weight_2012)
        DSM_age_2003, CBECS_age_2003, q1_2003, q2_2003, ks_2003 = compare_CBECS_and_plot(year=2003, index_end=184, CBECS_series=CBECS_Weights.Com_Weight_2003)
        DSM_age_1999, CBECS_age_1999, q1_1999, q2_1999, ks_1999 = compare_CBECS_and_plot(year=1999, index_end=180, CBECS_series=CBECS_Weights.Com_Weight_1999)
        DSM_age_1995, CBECS_age_1995, q1_1995, q2_1995, ks_1995 = compare_CBECS_and_plot(year=1995, index_end=176, CBECS_series=CBECS_Weights.Com_Weight_1995)
        DSM_age_1992, CBECS_age_1992, q1_1992, q2_1992, ks_1992 = compare_CBECS_and_plot(year=1992, index_end=173, CBECS_series=CBECS_Weights.Com_Weight_1992)
        DSM_age_1986, CBECS_age_1986, q1_1986, q2_1986, ks_1986 = compare_CBECS_and_plot(year=1986, index_end=167, CBECS_series=CBECS_Weights.Com_Weight_1986)
        DSM_age_1983, CBECS_age_1983, q1_1983, q2_1983, ks_1983 = compare_CBECS_and_plot(year=1983, index_end=164, CBECS_series=CBECS_Weights.Com_Weight_1983)
        DSM_age_1979, CBECS_age_1979, q1_1979, q2_1979, ks_1979 = compare_CBECS_and_plot(year=1979, index_end=160, CBECS_series=CBECS_Weights.Com_Weight_1979)

        # multiplot of each RECS year data against the data for that year in the DSM simulation
        print('mean ks values for each is = ',
              str(np.round(np.mean([ks_2012[0], ks_2003[0], ks_1999[0], ks_1995[0], ks_1992[0], ks_1986[0], ks_1983[0], ks_1979[0]]), 5)))

        # multiplot of each RECS year data against the data for that year in the DSM simulation
        fig = plt.figure(figsize=(16, 8))
        axes1 = fig.add_subplot(241)
        sns.distplot(DSM_age_2012, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation'),
        sns.distplot(CBECS_age_2012, kde=kde_flag, color="black", bins=n_bins, label='2012 CBECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_2012, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation',
                 color='gray')
        plt.hist(q2_2012, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='2012 CBECS',
                 color='lightblue')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_2012[0], 4)), ha='right', va='top', transform=axes1.transAxes,
                 fontsize='x-small')
        # plt.show()

        axes1 = fig.add_subplot(242)
        sns.distplot(DSM_age_2003, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation'),
        sns.distplot(CBECS_age_2003, kde=kde_flag, color="black", bins=n_bins, label='2003 CBECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_2003, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation',
                 color='gray')
        plt.hist(q2_2003, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='2003 CBECS',
                 color='lightblue')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_2003[0], 4)), ha='right', va='top', transform=axes1.transAxes,
                 fontsize='x-small')

        axes1 = fig.add_subplot(243)
        sns.distplot(DSM_age_1999, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation'),
        sns.distplot(CBECS_age_1999, kde=kde_flag, color="black", bins=n_bins, label='1999 CECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_1999, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation',
                 color='gray')
        plt.hist(q2_1999, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='1999 CBECS',
                 color='lightblue')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_1999[0], 4)), ha='right', va='top', transform=axes1.transAxes,
                 fontsize='x-small')

        axes1 = fig.add_subplot(244)
        sns.distplot(DSM_age_1995, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation'),
        sns.distplot(CBECS_age_1995, kde=kde_flag, color="black", bins=n_bins, label='1995 CBECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_1995, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation',
                 color='gray')
        plt.hist(q2_1995, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='1995 CBECS',
                 color='lightblue')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_1995[0], 4)), ha='right', va='top', transform=axes1.transAxes,
                 fontsize='x-small')

        axes1 = fig.add_subplot(245)
        sns.distplot(DSM_age_1992, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation'),
        sns.distplot(CBECS_age_1992, kde=kde_flag, color="black", bins=n_bins, label='1992 CBECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_1992, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation',
                 color='gray')
        plt.hist(q2_1992, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='1992 CBECS',
                 color='lightblue')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_1992[0], 4)), ha='right', va='top', transform=axes1.transAxes,
                 fontsize='x-small')

        axes1 = fig.add_subplot(246)
        sns.distplot(DSM_age_1986, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation'),
        sns.distplot(CBECS_age_1986, kde=kde_flag, color="black", bins=n_bins, label='1986 CBECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_1986, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation',
                 color='gray')
        plt.hist(q2_1986, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='1986 CBECS',
                 color='lightblue')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel(None)
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_1986[0], 4)), ha='right', va='top', transform=axes1.transAxes,
                 fontsize='x-small')

        axes1 = fig.add_subplot(247)
        sns.distplot(DSM_age_1983, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation'),
        sns.distplot(CBECS_age_1983, kde=kde_flag, color="black", bins=n_bins, label='1983 CBECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_1983, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation',
                 color='gray')
        plt.hist(q2_1983, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='1983 CBECS',
                 color='lightblue')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel('Age')
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_1983[0], 4)), ha='right', va='top', transform=axes1.transAxes,
                 fontsize='x-small')

        axes1 = fig.add_subplot(248)
        sns.distplot(DSM_age_1979, kde=kde_flag, color="blue", bins=n_bins, label='DSM Simulation'),
        sns.distplot(CBECS_age_1979, kde=kde_flag, color="black", bins=n_bins, label='1979 CBECS'),
        # plot cdfs next to one another
        axes2 = axes1.twinx()
        plt.hist(q1_1979, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='Simulation',
                 color='gray')
        plt.hist(q2_1979, density=True, bins=100, cumulative=True, alpha=1.0, histtype='step', label='1979 CBECS',
                 color='lightblue')
        axes2.set_ylabel('CDF')
        axes1.set_ylabel('Density')
        plt.legend(loc=9, fontsize='x-small');
        plt.xlabel('Age')
        plt.text(1.0, 0.99, 'ks = ' + str(np.round(ks_1979[0], 4)), ha='right', va='top', transform=axes1.transAxes,
                 fontsize='x-small')
        plt.show();
    else:
        compare_CBECS_and_plot(year=2012, index_end=193, CBECS_series=CBECS_Weights.Com_Weight_2012, plot=True, n_bins=n_bins)

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

SSP1_sc_df = pd.DataFrame(SSP1_dsm_res.s_c + SSP1_dsm_com.s_c + SSP1_dsm_pub.s_c)
SSP2_sc_df = pd.DataFrame(SSP2_dsm_res.s_c + SSP2_dsm_com.s_c + SSP2_dsm_pub.s_c)
SSP3_sc_df = pd.DataFrame(SSP3_dsm_res.s_c + SSP3_dsm_com.s_c + SSP3_dsm_pub.s_c)
SSP4_sc_df = pd.DataFrame(SSP4_dsm_res.s_c + SSP4_dsm_com.s_c + SSP4_dsm_pub.s_c)
SSP5_sc_df = pd.DataFrame(SSP5_dsm_res.s_c + SSP5_dsm_com.s_c + SSP5_dsm_pub.s_c)


# write to excel
writer = pd.ExcelWriter('./Results/SSP_dsm.xlsx', engine='xlsxwriter')
SSP1_dsm_df.to_excel(writer, sheet_name='SSP1')
SSP2_dsm_df.to_excel(writer, sheet_name='SSP2')
SSP3_dsm_df.to_excel(writer, sheet_name='SSP3')
SSP4_dsm_df.to_excel(writer, sheet_name='SSP4')
SSP5_dsm_df.to_excel(writer, sheet_name='SSP5')
SSP1_sc_df.to_excel(writer, sheet_name='SSP1_sc', index=False)
SSP2_sc_df.to_excel(writer, sheet_name='SSP2_sc', index=False)
SSP3_sc_df.to_excel(writer, sheet_name='SSP3_sc', index=False)
SSP4_sc_df.to_excel(writer, sheet_name='SSP4_sc', index=False)
SSP5_sc_df.to_excel(writer, sheet_name='SSP5_sc', index=False)
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
    print('')
    # plot_dsm(SSP1_dsm_res, 'SSP1 Residential')

# # ----------------------------------------------------------------------------------------------------------------------
# # Plot all scenarios together for all buildings
plot_MFA_all_same_graph = True
no_SSP5 = True      # True for ignoring SSP5, False for including SSP5
if plot_MFA_all_same_graph == True:
    plt.subplot(211)
    plt1, = plt.plot(SSP1_dsm_res.t, SSP1_dsm_res.s + SSP1_dsm_com.s + SSP1_dsm_pub.s)
    plt2, = plt.plot(SSP2_dsm_res.t, SSP2_dsm_res.s + SSP2_dsm_com.s + SSP2_dsm_pub.s)
    plt3, = plt.plot(SSP3_dsm_res.t, SSP3_dsm_res.s + SSP3_dsm_com.s + SSP3_dsm_pub.s)
    plt4, = plt.plot(SSP4_dsm_res.t, SSP4_dsm_res.s + SSP4_dsm_com.s + SSP4_dsm_pub.s)
    plt16, = plt.plot([base_year, base_year], [0, 200000], color='k', LineStyle='--')

    plt.legend([plt1, plt2, plt3, plt4], ['SSP1', 'SSP2', 'SSP3', 'SSP4'], loc=(1.05, 0.5))
    # plt.legend([plt1, plt2, plt3, plt4, plt5], ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], loc=(1.05, 0.5))
    # plt.legend(loc=(1.05, 0.5))
    plt.tight_layout()
    plt.xlabel('Year')
    plt.xlim(left=1980)
    plt.ylabel('million $m^2$')
    plt.title('Total Floor Space - Stock')
    # plt.show();

    plt.subplot(212)
    plt1, = plt.plot(SSP1_dsm_res.t, SSP1_dsm_res.i + SSP1_dsm_com.i + SSP1_dsm_pub.i, LineStyle='dashed', color='#1f77b4')
    plt2, = plt.plot(SSP1_dsm_res.t, SSP1_dsm_res.o + SSP1_dsm_com.o + SSP1_dsm_pub.o, color = '#1f77b4')
    plt3, = plt.plot(SSP2_dsm_res.t, SSP2_dsm_res.i + SSP2_dsm_com.i + SSP2_dsm_pub.i, LineStyle='dashed', color='#ff7f0e' )
    plt4, = plt.plot(SSP2_dsm_res.t, SSP2_dsm_res.o + SSP2_dsm_com.o + SSP2_dsm_pub.o, color='#ff7f0e')
    plt5, = plt.plot(SSP3_dsm_res.t, SSP3_dsm_res.i + SSP3_dsm_com.i + SSP3_dsm_pub.i, LineStyle='dashed', color='#2ca02c')
    plt6, = plt.plot(SSP3_dsm_res.t, SSP3_dsm_res.o + SSP3_dsm_com.o + SSP3_dsm_pub.o, color='#2ca02c')
    plt7, = plt.plot(SSP4_dsm_res.t, SSP4_dsm_res.i + SSP4_dsm_com.i + SSP4_dsm_pub.i, LineStyle='dashed', color='#d62728')
    plt8, = plt.plot(SSP4_dsm_res.t, SSP4_dsm_res.o + SSP4_dsm_com.o + SSP4_dsm_pub.o, color='#d62728')

    plt11, = plt.plot([base_year, base_year], [0, 3000], color='k', LineStyle='--')

    plt.legend([plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8],
               ['Inflow SSP1', 'Outflow SSP1',
                'Inflow SSP2', 'Outflow SSP2',
                'Inflow SSP3', 'Outflow SSP3',
                'Inflow SSP4', 'Outflow SSP4'], loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.ylim(top=5000)
    # plt.xlim(left=SSP1_dsm_res.t[0] + 5)
    plt.xlim(left=1980)
    plt.xlabel('Year')
    plt.ylabel('million m$^2/year$')
    plt.title('Total Floor Space - Flows')
    plt.show();


# # Plot all scenarios together for residential buildings
plot_MFA_all_same_graph = True
no_SSP5 = True      # True for ignoring SSP5, False for including SSP5
if plot_MFA_all_same_graph == True:
    plt.subplot(211)
    plt1, = plt.plot(SSP1_dsm_res.t, SSP1_dsm_res.s)
    plt2, = plt.plot(SSP2_dsm_res.t, SSP2_dsm_res.s)
    plt3, = plt.plot(SSP3_dsm_res.t, SSP3_dsm_res.s)
    plt4, = plt.plot(SSP4_dsm_res.t, SSP4_dsm_res.s)
    plt16, = plt.plot([base_year, base_year], [0, 175000], color='k', LineStyle='--')
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
    plt.xlim(left=1980)
    plt.ylabel('million $m^2$')
    plt.title('Residential Floor Space - Stock')
    # plt.show();

    plt.subplot(212)
    plt1, = plt.plot(SSP1_dsm_res.t, SSP1_dsm_res.i, LineStyle='dashed', color='#1f77b4')
    plt2, = plt.plot(SSP1_dsm_res.t, SSP1_dsm_res.o, color='#1f77b4')
    plt3, = plt.plot(SSP2_dsm_res.t, SSP2_dsm_res.i, LineStyle='dashed', color='#ff7f0e' )
    plt4, = plt.plot(SSP2_dsm_res.t, SSP2_dsm_res.o, color='#ff7f0e' )
    plt5, = plt.plot(SSP3_dsm_res.t, SSP3_dsm_res.i, LineStyle='dashed', color='#2ca02c')
    plt6, = plt.plot(SSP3_dsm_res.t, SSP3_dsm_res.o, color='#2ca02c')
    plt7, = plt.plot(SSP4_dsm_res.t, SSP4_dsm_res.i, LineStyle='dashed', color='#d62728')
    plt8, = plt.plot(SSP4_dsm_res.t, SSP4_dsm_res.o, color='#d62728')
    if no_SSP5 == True:
        temp = 'bleh'
    else:
        plt9, = plt.plot(SSP5_dsm_res.t, SSP5_dsm_res.i, LineStyle='dashed')
        plt0, = plt.plot(SSP5_dsm_res.t, SSP5_dsm_res.o)

    plt11, = plt.plot([base_year, base_year], [0, 2500], color='k', LineStyle='--')

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
    # plt.xlim(left=SSP1_dsm_res.t[0] + 5)
    plt.xlim(left=1980)
    plt.xlabel('Year')
    plt.ylabel('million m$^2/year$')
    plt.title('Residential Floor Space - Flows')
    plt.show();


# # Plot all scenarios together for commercial buildings
plot_MFA_all_same_graph = True
no_SSP5 = True      # True for ignoring SSP5, False for including SSP5
if plot_MFA_all_same_graph == True:
    plt.subplot(211)
    plt1, = plt.plot(SSP1_dsm_com.t, SSP1_dsm_com.s)
    plt2, = plt.plot(SSP2_dsm_com.t, SSP2_dsm_com.s)
    plt3, = plt.plot(SSP3_dsm_com.t, SSP3_dsm_com.s)
    plt4, = plt.plot(SSP4_dsm_com.t, SSP4_dsm_com.s)
    plt16, = plt.plot([base_year, base_year], [0, 35000], color='k', LineStyle='--')
    if no_SSP5 == True:
        temp = 'bleh'
    else:
        plt5, = plt.plot(SSP5_dsm_com.t, SSP5_dsm_com.s)
    if no_SSP5 == True:
        plt.legend([plt1, plt2, plt3, plt4], ['SSP1', 'SSP2', 'SSP3', 'SSP4'], loc=(1.05, 0.5))
    else:
        plt.legend([plt1, plt2, plt3, plt4, plt5], ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], loc=(1.05, 0.5))
    # plt.legend([plt1, plt2, plt3, plt4, plt5], ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], loc=(1.05, 0.5))
    # plt.legend(loc=(1.05, 0.5))
    plt.tight_layout()
    plt.xlim(left=1980)
    plt.xlabel('Year')
    plt.ylabel('million $m^2$')
    plt.title('Commercial Floor Space - Stock')
    # plt.show();

    plt.subplot(212)
    plt1, = plt.plot(SSP1_dsm_com.t, SSP1_dsm_com.i, LineStyle='dashed', color='#1f77b4')
    plt2, = plt.plot(SSP1_dsm_com.t, SSP1_dsm_com.o, color='#1f77b4')
    plt3, = plt.plot(SSP2_dsm_com.t, SSP2_dsm_com.i, LineStyle='dashed', color='#ff7f0e' )
    plt4, = plt.plot(SSP2_dsm_com.t, SSP2_dsm_com.o, color='#ff7f0e' )
    plt5, = plt.plot(SSP3_dsm_com.t, SSP3_dsm_com.i, LineStyle='dashed',color='#2ca02c')
    plt6, = plt.plot(SSP3_dsm_com.t, SSP3_dsm_com.o, color='#2ca02c')
    plt7, = plt.plot(SSP4_dsm_com.t, SSP4_dsm_com.i, LineStyle='dashed', color='#d62728')
    plt8, = plt.plot(SSP4_dsm_com.t, SSP4_dsm_com.o, color='#d62728')
    if no_SSP5 == True:
        temp = 'bleh'
    else:
        plt9, = plt.plot(SSP5_dsm_com.t, SSP5_dsm_com.i, LineStyle='dashed')
        plt0, = plt.plot(SSP5_dsm_com.t, SSP5_dsm_com.o)

    plt11, = plt.plot([base_year, base_year], [0, 600], color='k', LineStyle='--')

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
    # plt.xlim(left=SSP1_dsm_com.t[0] + 5)
    plt.xlim(left=1980)
    plt.xlabel('Year')
    plt.ylabel('million m$^2/year$')
    plt.title('Commercial Floor Space - Flows')
    plt.show();


# # Plot all scenarios together for public buildings
plot_MFA_all_same_graph = True
no_SSP5 = True      # True for ignoring SSP5, False for including SSP5
if plot_MFA_all_same_graph == True:
    plt.subplot(211)
    plt1, = plt.plot(SSP1_dsm_pub.t, SSP1_dsm_pub.s)
    plt2, = plt.plot(SSP2_dsm_pub.t, SSP2_dsm_pub.s)
    plt3, = plt.plot(SSP3_dsm_pub.t, SSP3_dsm_pub.s)
    plt4, = plt.plot(SSP4_dsm_pub.t, SSP4_dsm_pub.s)
    plt16, = plt.plot([base_year, base_year], [0, 6500], color='k', LineStyle='--')
    if no_SSP5 == True:
        temp = 'bleh'
    else:
        plt5, = plt.plot(SSP5_dsm_pub.t, SSP5_dsm_pub.s)
    if no_SSP5 == True:
        plt.legend([plt1, plt2, plt3, plt4], ['SSP1', 'SSP2', 'SSP3', 'SSP4'], loc=(1.05, 0.5))
    else:
        plt.legend([plt1, plt2, plt3, plt4, plt5], ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], loc=(1.05, 0.5))
    # plt.legend([plt1, plt2, plt3, plt4, plt5], ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], loc=(1.05, 0.5))
    # plt.legend(loc=(1.05, 0.5))
    plt.tight_layout()
    plt.xlabel('Year')
    plt.xlim(left=1980)
    plt.ylabel('million $m^2$ ')
    plt.title('Public Floor Space - Stock')
    # plt.show();

    plt.subplot(212)
    plt1, = plt.plot(SSP1_dsm_pub.t, SSP1_dsm_pub.i, LineStyle='dashed', color='#1f77b4')
    plt2, = plt.plot(SSP1_dsm_pub.t, SSP1_dsm_pub.o, color='#1f77b4')
    plt3, = plt.plot(SSP2_dsm_pub.t, SSP2_dsm_pub.i, LineStyle='dashed', color='#ff7f0e' )
    plt4, = plt.plot(SSP2_dsm_pub.t, SSP2_dsm_pub.o, color='#ff7f0e' )
    plt5, = plt.plot(SSP3_dsm_pub.t, SSP3_dsm_pub.i, LineStyle='dashed', color='#2ca02c')
    plt6, = plt.plot(SSP3_dsm_pub.t, SSP3_dsm_pub.o, color='#2ca02c')
    plt7, = plt.plot(SSP4_dsm_pub.t, SSP4_dsm_pub.i, LineStyle='dashed', color='#d62728')
    plt8, = plt.plot(SSP4_dsm_pub.t, SSP4_dsm_pub.o, color='#d62728')
    if no_SSP5 == True:
        temp = 'bleh'
    else:
        plt9, = plt.plot(SSP5_dsm_pub.t, SSP5_dsm_pub.i, LineStyle='dashed')
        plt0, = plt.plot(SSP5_dsm_pub.t, SSP5_dsm_pub.o)

    plt11, = plt.plot([base_year, base_year], [0, 100], color='k', LineStyle='--')

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
    # plt.xlim(left=SSP1_dsm_pub.t[0] + 5)
    plt.xlim(left=1980)
    plt.xlabel('Year')
    plt.ylabel('million m$^2/year$')
    plt.title('Public Floor Space - Flows')
    plt.show();
