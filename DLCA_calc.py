# Functions for dynamic LCA

# Methods adapted from v2.0 of cooper et a. 2020 (https://doi.org/10.15125/BATH-00923)
# The details of the methods are described by the paper: https://doi.org/10.1016/j.biombioe.2020.105778

# Load in appropriate libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate a dynamic GHG methods for a given input file.
def DLCA_calc(input_data=None, start_yr=0, end_yr=500, time_horizon=100, fossil_CH4=False, plot=True, Name='test'):
    ## Items to add:
    # - additional gases, currently only handles CO2 and CH4

    ## Inputs
    # fossil_CH4 = False
    CH4_to_CO2 = 0.5 * (12 + 16 * 2) / (12 + 1 * 4)

    Other_forcing_efficacy = 1

    # climate sensitivity [K (W m-2)-1]
    clim_sens1 = 0.631
    clim_sens2 = 0.429
    # climate time response [years]
    clim_time1 = 8.4
    clim_time2 = 409.5

    mol_mass_air = 28.97  # g/mol
    earth_area = 5.10E+14  # m2
    mass_atm = 5.14E+18  # kg

    # If wanting to include more than just CO2 and Methane, these inputs will need to be incorporated.
    # gas1_name =
    # gas2_name =
    # gas3_name =
    # life_gas1 =
    # life_gas2 =
    # life_gas3 =
    # A_gas1 =
    # A_gas2 =
    # A_gas3 =
    # GTP_gas1 =
    # GTP_gas2 =
    # GTP_gas3 =
    # GWP_CH4 =
    # GWP_gas1 =
    # GWP_gas2 =
    # GWP_gas3 =

    # Lifetime Parameters
    life1_CO2 = 394.4
    life2_CO2 = 36.54
    life3_CO2 = 4.304
    life_CH4 = 12.4
    # Proportion lifetime applies to
    p_0 = 0.2173
    p_1 = 0.224
    p_2 = 0.2824
    p_3 = 0.2763

    # Radiative efficiency [W m-2 ppb-1]
    CO2_RE = 1.37E-05
    CH4_RE = 0.000363
    # molar mass of CO2
    mol_mass_CO2 = 44.01
    mol_mass_CH4 = 16.05

    # adjustment factor
    adj_factor_CO2 = 0.99746898064295
    adj_factor_CH4 = 1 + 0.15 + 0.5

    # Specific radiative forcing [W m-2 kg-1]
    # Forcing parameters from [ref 5]
    A_CO2 = CO2_RE * adj_factor_CO2 * mol_mass_air * 1000000000 / mass_atm / mol_mass_CO2
    A_CH4 = CH4_RE * adj_factor_CH4 * mol_mass_air * 1000000000 / mass_atm / mol_mass_CH4

    AGWP_CO2 = A_CO2 * (p_0 * time_horizon
                        + p_1 * life1_CO2 * (1 - np.exp(-time_horizon / life1_CO2))
                        + p_2 * life2_CO2 * (1 - np.exp(-time_horizon / life2_CO2))
                        + p_3 * life3_CO2 * (1 - np.exp(-time_horizon / life3_CO2)))

    AGTP_CO2 = A_CO2 * (p_0 * (clim_sens1 * (1 - np.exp(-time_horizon / clim_time1)) + clim_sens2 * (
                1 - np.exp(-time_horizon / clim_time2)))
                        + p_1 * life1_CO2 * (clim_sens1 / (life1_CO2 - clim_time1) * (
                        np.exp(-time_horizon / life1_CO2) - np.exp(-time_horizon / clim_time1)) + clim_sens2 / (
                                                         life1_CO2 - clim_time2) * (
                                                         np.exp(-time_horizon / life1_CO2) - np.exp(
                                                     -time_horizon / clim_time2)))
                        + p_2 * life2_CO2 * (clim_sens1 / (life2_CO2 - clim_time1) * (
                        np.exp(-time_horizon / life2_CO2) - np.exp(-time_horizon / clim_time1)) + clim_sens2 / (
                                                         life2_CO2 - clim_time2) * (
                                                         np.exp(-time_horizon / life2_CO2) - np.exp(
                                                     -time_horizon / clim_time2)))
                        + p_1 * life3_CO2 * (clim_sens1 / (life3_CO2 - clim_time1) * (
                        np.exp(-time_horizon / life3_CO2) - np.exp(-time_horizon / clim_time1)) + clim_sens2 / (
                                                         life3_CO2 - clim_time2) * (
                                                         np.exp(-time_horizon / life3_CO2) - np.exp(
                                                     -time_horizon / clim_time2))))

    AGWP_CH4 = A_CH4 * life_CH4 * (1 - np.exp(-time_horizon / life_CH4))

    AGTP_CH4 = A_CH4 * (life_CH4 * clim_sens1 / (life_CH4 - clim_time1) * (
                np.exp(-time_horizon / life_CH4) - np.exp(-time_horizon / clim_time1))
                        + life_CH4 * clim_sens2 / (life_CH4 - clim_time2) * (
                                    np.exp(-time_horizon / life_CH4) - np.exp(-time_horizon / clim_time2)))

    GWP_CH4 = AGWP_CH4 / AGWP_CO2
    GTP_CH4 = AGTP_CH4 / AGTP_CO2

    # years to consider in the analysis
    year_vec = np.linspace(start=start_yr, stop=end_yr, num=(end_yr - start_yr + 1)).astype(int)

    # flux of emissions input
    if input_data == None:
        # Load in 1 kg of both CO2 and CH4 in year 0 for testing sake.
        CO2_flux = np.linspace(start=0, stop=0, num=501)
        CO2_flux[0] = 1  # single pulse of 1 kg of CO2 in the first year

        CH4_flux = np.linspace(start=0, stop=0, num=501)
        CH4_flux[0] = 1  # single pulse of 1 kg of CH4 in the first year
    else:
        input_data_df = pd.read_excel(input_data, sheet_name='flux', index_col='year')
        CO2_flux = input_data_df['CO2_flux']
        CH4_flux = input_data_df['CH4_flux']

    # check to see if years are correct.
    if len(CO2_flux) != len(year_vec):
        print('ERROR:  Length of years is mismatched with flux input')
        exit()

    # Cumulative net emissions [kg]
    cum_net_emissions = pd.DataFrame(0, index=year_vec,
                                     columns=['CO2_net', 'CH4_net', 'gas1_net', 'gas2_net', 'gas3_net'])
    for index, row in cum_net_emissions.iterrows():
        if index == 0:
            cum_net_emissions.loc[index, 'CO2_net'] = CO2_flux[index]
            cum_net_emissions.loc[index, 'CH4_net'] = CH4_flux[index]
            # Add in new gases too here
        else:
            cum_net_emissions.loc[index, 'CO2_net'] = cum_net_emissions.loc[index - 1, 'CO2_net'] + CO2_flux[index]
            cum_net_emissions.loc[index, 'CH4_net'] = cum_net_emissions.loc[index - 1, 'CH4_net'] + CH4_flux[index]
            # Add in new gases too here

        # Initialize mass in atmosphere dataframe
    mass_in_atm_df = pd.DataFrame(0, index=year_vec, columns=['CO2', 'CH4'])
    # Initialize CO2 atmospheric decay components dataframe
    CO2_mass_df = pd.DataFrame(0, index=year_vec, columns=['mass0', 'mass1', 'mass2', 'mass3', 'CH4_to_CO2'])
    # Initialize radiative forcing dataframe
    irf_df = pd.DataFrame(0, index=year_vec, columns=['CO2_radfor', 'CH4_radfor', 'CO2_IRF', 'CH4_IRF', 'IRF_all'])

    # Calculate the Mass Decay
    for index, row in mass_in_atm_df.iterrows():
        # Calculation for the first timestep
        if index == 0:
            mass_in_atm_df.loc[index, 'CH4'] = CH4_flux[index]
            CO2_mass_df.loc[index, 'mass0'] = p_0 * CO2_flux[index]
            CO2_mass_df.loc[index, 'mass1'] = p_1 * CO2_flux[index]
            CO2_mass_df.loc[index, 'mass2'] = p_2 * CO2_flux[index]
            CO2_mass_df.loc[index, 'mass3'] = p_3 * CO2_flux[index]
            CO2_mass_df.loc[index, 'CH4_to_CO2'] = 0
        # Calculation for the start + 1 to end timesteps
        else:
            mass_in_atm_df.loc[index, 'CH4'] = CH4_flux[index] + mass_in_atm_df.loc[index - 1, 'CH4'] * np.exp(
                -1 / life_CH4)
            CO2_mass_df.loc[index, 'mass0'] = p_0 * (CO2_flux[index] + CO2_mass_df.loc[index, 'CH4_to_CO2']) + \
                                              CO2_mass_df.loc[index - 1, 'mass0']
            CO2_mass_df.loc[index, 'mass1'] = p_1 * (CO2_flux[index] + CO2_mass_df.loc[index, 'CH4_to_CO2']) + \
                                              CO2_mass_df.loc[index - 1, 'mass1'] * np.exp(-1 / life1_CO2)
            CO2_mass_df.loc[index, 'mass2'] = p_2 * (CO2_flux[index] + CO2_mass_df.loc[index, 'CH4_to_CO2']) + \
                                              CO2_mass_df.loc[index - 1, 'mass2'] * np.exp(-1 / life2_CO2)
            CO2_mass_df.loc[index, 'mass3'] = p_3 * (CO2_flux[index] + CO2_mass_df.loc[index, 'CH4_to_CO2']) + \
                                              CO2_mass_df.loc[index - 1, 'mass3'] * np.exp(-1 / life3_CO2)
            if fossil_CH4 == True:
                CO2_mass_df.loc[index, 'CH4_to_CO2'] = mass_in_atm_df.loc[index - 1, 'CH4'] * (
                            1 - np.exp(-1 / life_CH4)) * CH4_to_CO2
            elif fossil_CH4 == False:
                CO2_mass_df.loc[index, 'CH4_to_CO2'] = 0
        mass_in_atm_df.loc[index, 'CO2'] = CO2_mass_df.loc[index, 'mass0'] + CO2_mass_df.loc[index, 'mass1'] + \
                                           CO2_mass_df.loc[index, 'mass2'] + CO2_mass_df.loc[index, 'mass3']

    # Calculate the radiative forcing (instantaneous just after start of year), [W m^-2]
    irf_df['CO2_radfor'] = A_CO2 * mass_in_atm_df['CO2']
    irf_df['CH4_radfor'] = A_CH4 * mass_in_atm_df['CH4']

    # Calculate integrated radiative forcing [W-year m^-2]
    for index, row in irf_df.iterrows():
        if index == 0:
            irf_df.loc[index, 'CO2_IRF'] = 0
            irf_df.loc[index, 'CH4_IRF'] = 0
        else:
            irf_df.loc[index, 'CO2_IRF'] = irf_df.loc[index - 1, 'CO2_IRF'] + A_CO2 * (
                        CO2_mass_df.loc[index - 1, 'mass0'] +
                        CO2_mass_df.loc[index - 1, 'mass1'] * life1_CO2 * (1 - np.exp(-1 / life1_CO2)) +
                        CO2_mass_df.loc[index - 1, 'mass2'] * life2_CO2 * (1 - np.exp(-1 / life2_CO2)) +
                        CO2_mass_df.loc[index - 1, 'mass3'] * life3_CO2 * (1 - np.exp(-1 / life3_CO2))
                        )
            irf_df.loc[index, 'CH4_IRF'] = irf_df.loc[index - 1, 'CH4_IRF'] + mass_in_atm_df.loc[
                index - 1, 'CH4'] * A_CH4 * life_CH4 * (1 - np.exp(-1 / life_CH4))

    irf_df['IRF_all'] = irf_df['CO2_IRF'] + irf_df['CH4_IRF']

    # Temperature change components (two-part climate response model)
    # Initialize the temperature change dataframe
    temp_change_df = pd.DataFrame(0, index=year_vec,
                                  columns=['CO2_temp_s1', 'CO2_temp_s2', 'CH4_temp_s1', 'CH4_temp_s2', 'CO2_temp',
                                           'CH4_temp', 'temp_change'])
    for index, row in temp_change_df.iterrows():
        if index == 0:
            temp_change_df.loc[index, 'CO2_temp_s1'] = 0
            temp_change_df.loc[index, 'CO2_temp_s2'] = 0
            temp_change_df.loc[index, 'CH4_temp_s1'] = 0
            temp_change_df.loc[index, 'CH4_temp_s2'] = 0
            temp_change_df.loc[index, 'CO2_temp'] = 0
            temp_change_df.loc[index, 'CH4_temp'] = 0
        else:
            temp_change_df.loc[index, 'CO2_temp_s1'] = A_CO2 * (
                        CO2_mass_df.loc[index - 1, 'mass0'] * clim_sens1 * (1 - np.exp(-1 / clim_time1))
                        + CO2_mass_df.loc[index - 1, 'mass1'] * clim_sens1 * life1_CO2 / (life1_CO2 - clim_time1) * (
                                    np.exp(-1 / life1_CO2) - np.exp(-1 / clim_time1))
                        + CO2_mass_df.loc[index - 1, 'mass2'] * clim_sens1 * life2_CO2 / (life2_CO2 - clim_time1) * (
                                    np.exp(-1 / life2_CO2) - np.exp(-1 / clim_time1))
                        + CO2_mass_df.loc[index - 1, 'mass3'] * clim_sens1 * life3_CO2 / (life3_CO2 - clim_time1) * (
                                    np.exp(-1 / life3_CO2) - np.exp(-1 / clim_time1))) + \
                                                       temp_change_df.loc[index - 1, 'CO2_temp_s1'] * np.exp(
                -1 / clim_time1)
            temp_change_df.loc[index, 'CO2_temp_s2'] = A_CO2 * (
                        CO2_mass_df.loc[index - 1, 'mass0'] * clim_sens2 * (1 - np.exp(-1 / clim_time2))
                        + CO2_mass_df.loc[index - 1, 'mass1'] * clim_sens2 * life1_CO2 / (life1_CO2 - clim_time2) * (
                                    np.exp(-1 / life1_CO2) - np.exp(-1 / clim_time2))
                        + CO2_mass_df.loc[index - 1, 'mass2'] * clim_sens2 * life2_CO2 / (life2_CO2 - clim_time2) * (
                                    np.exp(-1 / life2_CO2) - np.exp(-1 / clim_time2))
                        + CO2_mass_df.loc[index - 1, 'mass3'] * clim_sens2 * life3_CO2 / (life3_CO2 - clim_time2) * (
                                    np.exp(-1 / life3_CO2) - np.exp(-1 / clim_time2))) + \
                                                       temp_change_df.loc[index - 1, 'CO2_temp_s2'] * np.exp(
                -1 / clim_time2)
            temp_change_df.loc[index, 'CH4_temp_s1'] = A_CH4 * mass_in_atm_df.loc[
                index - 1, 'CH4'] * clim_sens1 * life_CH4 / (life_CH4 - clim_time1) * (
                                                                   np.exp(-1 / life_CH4) - np.exp(-1 / clim_time1)) + \
                                                       temp_change_df.loc[index - 1, 'CH4_temp_s1'] * np.exp(
                -1 / clim_time1)
            temp_change_df.loc[index, 'CH4_temp_s2'] = A_CH4 * mass_in_atm_df.loc[
                index - 1, 'CH4'] * clim_sens2 * life_CH4 / (life_CH4 - clim_time2) * (
                                                                   np.exp(-1 / life_CH4) - np.exp(-1 / clim_time2)) + \
                                                       temp_change_df.loc[index - 1, 'CH4_temp_s2'] * np.exp(
                -1 / clim_time2)
            temp_change_df.loc[index, 'CO2_temp'] = temp_change_df.loc[index, 'CO2_temp_s1'] + temp_change_df.loc[
                index, 'CO2_temp_s2']
            temp_change_df.loc[index, 'CH4_temp'] = temp_change_df.loc[index, 'CH4_temp_s1'] + temp_change_df.loc[
                index, 'CH4_temp_s2']

    temp_change_df['temp_change'] = temp_change_df['CO2_temp'] + temp_change_df['CH4_temp'].fillna(0)

    # load in supporting data
    AGWP_df_sup = pd.read_excel('./DLCA_supp_data.xlsx', sheet_name='AGWP', index_col='year')  # [W yr m^-2]
    AGTP_df_sup = pd.read_excel('./DLCA_supp_data.xlsx', sheet_name='AGTP', index_col='year')  # [delta-K]

    # AGWP and AGTP in reverse-order from year X (the time-horizon)
    AGWP_AGTP_rev = pd.DataFrame(0, index=year_vec,
                                 columns=['CO2_AGWP_rev', 'CH4_AGWP_rev', 'CO2_AGTP_rev', 'CH4_AGTP_rev'])
    AGWP_AGTP_rev['CO2_AGWP_rev'] = pd.concat(
        (AGWP_df_sup.loc[0:time_horizon, 'CO2_AGWP'][::-1], pd.Series(year_vec[time_horizon + 1:] * 0))).reset_index(
        drop=True)
    AGWP_AGTP_rev['CH4_AGWP_rev'] = pd.concat(
        (AGWP_df_sup.loc[0:time_horizon, 'CH4_AGWP'][::-1], pd.Series(year_vec[time_horizon + 1:] * 0))).reset_index(
        drop=True)
    AGWP_AGTP_rev['CO2_AGTP_rev'] = pd.concat(
        (AGTP_df_sup.loc[0:time_horizon, 'CO2_AGTP'][::-1], pd.Series(year_vec[time_horizon + 1:] * 0))).reset_index(
        drop=True)
    AGWP_AGTP_rev['CH4_AGTP_rev'] = pd.concat(
        (AGTP_df_sup.loc[0:time_horizon, 'CH4_AGTP'][::-1], pd.Series(year_vec[time_horizon + 1:] * 0))).reset_index(
        drop=True)

    # Equivalency summary dataframe
    # equiv_df = pd.DataFrame(0, index=year_vec, columns=['irf_GWP_slidingTH', 'GTP_slidingTH', 'CO2_AGTP_rev', 'CH4_AGTP_rev'])

    equiv_df = pd.DataFrame(0, index=year_vec, columns=['LCA_dyn', 'LCA_static'])
    equiv_df['LCA_dyn'] = irf_df['IRF_all'] / AGWP_CO2
    equiv_df['LCA_static'] = cum_net_emissions['CO2_net'] + cum_net_emissions[
        'CH4_net'] * GWP_CH4  # add in more gasses here if desired

    if plot == True:
        fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(10, 8))
        # fig.tight_layout()
        sns.set_theme(style="ticks")
        # Plot the mass in the atmosphere
        sns.lineplot(ax=axes[0, 0], data=mass_in_atm_df)
        axes[0, 0].set(xlabel='Year', ylabel='kg', title='Mass in Atmosphere')

        # Plot the cumulative GWI (integrated radiative forcing)
        sns.lineplot(ax=axes[1, 0], data=irf_df.drop(['CO2_radfor', 'CH4_radfor'], axis=1))
        axes[1, 0].set(xlabel='Year', ylabel='$W-yr/m^{-2}$', title='Cumulative GWI (Integrated radiative forcing)')
        axes[1, 0].legend(labels=['CO2', 'CH4', 'Total'])

        # Plot the temperature change effect
        sns.lineplot(ax=axes[0, 1], data=temp_change_df, y='temp_change', x=temp_change_df.index)
        axes[0, 1].set(xlabel='Year', ylabel='$\Delta K$', title='Temperature Change Effect')

        # Plot the CO2 equivalences
        sns.lineplot(ax=axes[1, 1], data=equiv_df)
        axes[1, 1].set(xlabel='Year', ylabel='$kg CO_2e$', title='Carbon Dioxide-Equivalence')

        plt.savefig('./DLCA/DLCA_plot_' + Name + '.png')

    return mass_in_atm_df, irf_df, temp_change_df, equiv_df


S1_mass_in_atm, S1_irf_df, S1_temp_change_df, S1_equiv_df = DLCA_calc(
    input_data='/Users/josepharehart/PycharmProjects/Bldg_Stock/DLCA/flux_S1.xlsx',
    start_yr=0, end_yr=500, time_horizon=100, fossil_CH4=False, plot=True, Name='S1')

S2_mass_in_atm, S2_irf_df, S2_temp_change_df, S2_equiv_df = DLCA_calc(
    input_data='/Users/josepharehart/PycharmProjects/Bldg_Stock/DLCA/flux_S2.xlsx',
    start_yr=0, end_yr=500, time_horizon=100, fossil_CH4=False, plot=True, Name='S2')

S3_mass_in_atm, S3_irf_df, S3_temp_change_df, S3_equiv_df = DLCA_calc(
    input_data='/Users/josepharehart/PycharmProjects/Bldg_Stock/DLCA/flux_S3.xlsx',
    start_yr=0, end_yr=500, time_horizon=100, fossil_CH4=False, plot=True, Name='S3')

S4_mass_in_atm, S4_irf_df, S4_temp_change_df, S4_equiv_df = DLCA_calc(
    input_data='/Users/josepharehart/PycharmProjects/Bldg_Stock/DLCA/flux_S4.xlsx',
    start_yr=0, end_yr=500, time_horizon=100, fossil_CH4=False, plot=True, Name='S4')

S5_mass_in_atm, S5_irf_df, S5_temp_change_df, S5_equiv_df = DLCA_calc(
    input_data='/Users/josepharehart/PycharmProjects/Bldg_Stock/DLCA/flux_S5.xlsx',
    start_yr=0, end_yr=500, time_horizon=100, fossil_CH4=False, plot=True, Name='S5')

S6_mass_in_atm, S6_irf_df, S6_temp_change_df, S6_equiv_df = DLCA_calc(
    input_data='/Users/josepharehart/PycharmProjects/Bldg_Stock/DLCA/flux_S6.xlsx',
    start_yr=0, end_yr=500, time_horizon=100, fossil_CH4=False, plot=True, Name='S6')


# Plot all of the scenarios together
SUMMARY_mass = pd.DataFrame({'S1_CO2':S1_mass_in_atm['CO2'],
                             'S2_CO2':S1_mass_in_atm['CO2'],
                             'S3_CO2':S1_mass_in_atm['CO2'],
                             'S4_CO2':S1_mass_in_atm['CO2'],
                             'S5_CO2':S1_mass_in_atm['CO2'],
                             'S6_CO2':S1_mass_in_atm['CO2'],
                             'S1_CH4':S1_mass_in_atm['CH4'],
                             'S2_CH4':S1_mass_in_atm['CH4'],
                             'S3_CH4':S1_mass_in_atm['CH4'],
                             'S4_CH4':S1_mass_in_atm['CH4'],
                             'S5_CH4':S1_mass_in_atm['CH4'],
                             'S6_CH4':S1_mass_in_atm['CH4'],
                             })

SUMMARY_irf = pd.DataFrame({'S1_irf':S1_irf_df['IRF_all'],
                            'S2_irf':S2_irf_df['IRF_all'],
                            'S3_irf':S3_irf_df['IRF_all'],
                            'S4_irf':S4_irf_df['IRF_all'],
                            'S5_irf':S5_irf_df['IRF_all'],
                            'S6_irf':S6_irf_df['IRF_all'],
                             })

SUMMARY_temp = pd.DataFrame({'S1_temp':S1_temp_change_df['temp_change'],
                             'S2_temp':S2_temp_change_df['temp_change'],
                             'S3_temp':S3_temp_change_df['temp_change'],
                             'S4_temp':S4_temp_change_df['temp_change'],
                             'S5_temp':S5_temp_change_df['temp_change'],
                             'S6_temp':S6_temp_change_df['temp_change'],
                             })

SUMMARY_equiv = pd.DataFrame({'S1_LCA_dyn':S1_equiv_df['LCA_dyn'],
                             'S2_LCA_dyn':S2_equiv_df['LCA_dyn'],
                             'S3_LCA_dyn':S3_equiv_df['LCA_dyn'],
                             'S4_LCA_dyn':S4_equiv_df['LCA_dyn'],
                             'S5_LCA_dyn':S5_equiv_df['LCA_dyn'],
                             'S6_LCA_dyn':S6_equiv_df['LCA_dyn'],
                             'S1_LCA_static':S1_equiv_df['LCA_static'],
                             'S2_LCA_static':S2_equiv_df['LCA_static'],
                             'S3_LCA_static':S3_equiv_df['LCA_static'],
                             'S4_LCA_static':S4_equiv_df['LCA_static'],
                             'S5_LCA_static':S5_equiv_df['LCA_static'],
                             'S6_LCA_static':S6_equiv_df['LCA_static'],
                             })


fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(10, 8))
# fig.tight_layout()
sns.set_theme(style="ticks")
# Plot the mass in the atmosphere
sns.lineplot(ax=axes[0, 0], data=SUMMARY_mass)
axes[0, 0].set(xlabel='Year', ylabel='kg', title='Mass in Atmosphere')

# Plot the cumulative GWI (integrated radiative forcing)
sns.lineplot(ax=axes[1, 0], data=SUMMARY_irf)
axes[1, 0].set(xlabel='Year', ylabel='$W-yr/m^{-2}$', title='Cumulative GWI (Integrated radiative forcing)')

# Plot the temperature change effect
sns.lineplot(ax=axes[0, 1], data=SUMMARY_temp)
axes[0, 1].set(xlabel='Year', ylabel='$\Delta K$', title='Temperature Change Effect')

# Plot the CO2 equivalences
sns.lineplot(ax=axes[1, 1], data=SUMMARY_equiv)
axes[1, 1].set(xlabel='Year', ylabel='$kg CO_2e$', title='Carbon Dioxide-Equivalence')

plt.savefig('./DLCA/DLCA_plot_scenarios.png')
