# Calculate the emissison inventory for each scenario


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
import numpy as np
import seaborn as sns





# load in the inflow results of the MFA analysis (Mt of material)
S1_i_mean = pd.read_csv('./Results/MFA_results/S1_mat_i_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S2_i_mean = pd.read_csv('./Results/MFA_results/S2_mat_i_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S3_i_mean = pd.read_csv('./Results/MFA_results/S3_mat_i_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S4_i_mean = pd.read_csv('./Results/MFA_results/S4_mat_i_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S5_i_mean = pd.read_csv('./Results/MFA_results/S5_mat_i_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S6_i_mean = pd.read_csv('./Results/MFA_results/S6_mat_i_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S7_i_mean = pd.read_csv('./Results/MFA_results/S7_mat_i_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S1_o_mean = pd.read_csv('./Results/MFA_results/S1_mat_o_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S2_o_mean = pd.read_csv('./Results/MFA_results/S2_mat_o_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S3_o_mean = pd.read_csv('./Results/MFA_results/S3_mat_o_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S4_o_mean = pd.read_csv('./Results/MFA_results/S4_mat_o_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S5_o_mean = pd.read_csv('./Results/MFA_results/S5_mat_o_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S6_o_mean = pd.read_csv('./Results/MFA_results/S6_mat_o_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg
S7_o_mean = pd.read_csv('./Results/MFA_results/S7_mat_o_mean.csv').set_index('time', drop=True) * 1e9   # convert to kg


# debugging
# mat_flow_year = S1_i_mean[0:1].set_index('time', drop=True)
mat_flow_out = S1_o_mean
mat_flow_in = S1_i_mean
year=2017
scenario_name = 'S1'
include_carb = True

def create_dyn_inventory_scenario(mat_flow_in,mat_flow_out,scenario_name,include_carb=True,plot_emission=True):
    # notes about the function
    #       only including A1-A3 emissions for CO2 and CH4.
    #       assuming biomass will be long stored (end-of-life is outside the scope of the study)

    # Initialize time frame for dynamic LCA
    start_yr = 0
    end_yr = 500
    year_vec = np.linspace(start=start_yr, stop=end_yr, num=(end_yr - start_yr + 1)).astype(int)

    ### ------- BIOMASS REGROWTH MODEL -------
    # bio-based
    # Biogenic carbon content in wood
    ECC_bio_wood = 0.5 * 44/12 * (1-12/100)
    rot_period_glulam = 40   # years
    rot_period_dimlum = 75   # years

    # Function to calculate the Biomass regrowth for each cohort of materials
    def biomass_regrowth_i(years=year_vec, year_label=2017, rot_period=50, bio_CO2_coeff=ECC_bio_wood, curve='Normal'):
        # returns the rate at which the carbon is regrown in the forest in each subsequent year
        if curve == 'Normal':
            # Cherubini et al. 2011 generic method for biomass regrowth
            mu = rot_period / 2
            sigma = mu/2
            result = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((year_vec- mu)/ sigma) ** 2) * ECC_bio_wood
        if curve == 'Richards':
            # Richards growth curve: another model for biomass growth centered around half of the rotation period.
            growth_rate=0.2
            M = rot_period / 2
            A = year_vec[0]
            B = ECC_bio_wood
            # logistic growth rate:
            t = np.linspace(year_vec[0], year_vec[-1], num=(year_vec[-1] - year_vec[0] + 1), endpoint=True)
            y = A + (B - A) / (1 +  np.exp(-growth_rate * (t-M)))
            result_pre = pd.DataFrame({year_label:y}).diff()
            result_pre[year_label][0] = 0
            result = result_pre[year_label]
        return result

    # initialize output dataframes
    regrowth_dimlum_all = pd.DataFrame()
    regrowth_glulam_all = pd.DataFrame()
    counter=0
    for i in mat_flow_in.iterrows():
        # year = mat_flow_in.loc[i[0]]
        # compute the regrowth for the i_th year
        regrowth_dimlum_i = - mat_flow_in.loc[i[0]]['Sum_dimlum_inflow'] * biomass_regrowth_i(years=year_vec, year_label=(i[0]), rot_period=rot_period_dimlum, bio_CO2_coeff=ECC_bio_wood, curve='Richards')
        regrowth_glulam_i = - mat_flow_in.loc[i[0]]['Sum_engwood_inflow'] * biomass_regrowth_i(years=year_vec, year_label=(i[0]), rot_period=rot_period_glulam, bio_CO2_coeff=ECC_bio_wood, curve='Richards')
        # append to the large dataframe
        regrowth_dimlum_all = regrowth_dimlum_all.append(regrowth_dimlum_i)
        regrowth_glulam_all = regrowth_glulam_all.append(regrowth_glulam_i)

    # Transpose the dataframe
    regrowth_dimlum_all_T = regrowth_dimlum_all.T
    regrowth_glulam_all_T = regrowth_glulam_all.T

    # Shift cohort down to appropriate "carbon-year"
    counter = 0
    for column in regrowth_dimlum_all_T:
        regrowth_dimlum_all_T[column] = regrowth_dimlum_all_T[column].shift(counter)
        regrowth_glulam_all_T[column] = regrowth_glulam_all_T[column].shift(counter)
        counter = counter + 1

    # Sum across all cohorts to get GHG inventory (Mt biogenic carbon uptake) per year for 500 years.
    total_uptake_dimlum = regrowth_dimlum_all_T.sum(axis=1)
    total_uptake_glulam = regrowth_glulam_all_T.sum(axis=1)

    # mass balance check
    # sum(S1_i_mean['Sum_dimlum_inflow'] * ECC_bio_wood)
    # sum(total_uptake_dimlum)
    # sum(S1_i_mean['Sum_engwood_inflow'] * ECC_bio_wood)
    # sum(total_uptake_glulam)
    # --> All looks good from this perspective.

    ### ------- CARBONATION MODEL -------
    if include_carb==True:
        # debug
        mass_in=mat_flow_in['Sum_conc_inflow']
        mass_out=mat_flow_out['Sum_conc_outflow']
        service_life=60
        year_label=2017
        def carb_uptake(years=year_vec, year_label=2017,mass_in=mat_flow_in['Sum_conc_inflow'], mass_out=mat_flow_out['Sum_conc_outflow'], service_life=60):
            # carbonation input parameters
            alpha = 0.165
            Beta_classC = 0.27
            phi_c = 0.611   # Andrade 2020
            y = 0.2
            C_m = alpha - Beta_classC * y
            R = 0.0016*27.5 ** 3.106    # for 4000psi concrete, Type I/II cement
            k0 = 3.0
            k1 = 1.0
            k2 = 1.0
            n = 0
            atm_CO2_ppm = 400
            atm_CO2_kgm3 = 1.2e-4   # kg/m3
            mass_cement = 361       # kg/m3
            mass_flyash = 90        # kg/m3
            perc_surface_exposed = 0.8

            # Idealize everything as a wall
            wall_thickness = 0.6    # m
            dens_conc = 2400        # kg/m3
            SA_i = perc_surface_exposed * mass_in[year_label] / dens_conc / wall_thickness

            df_carb_service = pd.DataFrame({'time':np.linspace(0,service_life,service_life+1, dtype=int)}).set_index('time', drop=False)
            df_carb_service['x'] = np.sqrt(2*atm_CO2_kgm3*df_carb_service['time']/R) * np.sqrt(k0*k1*k2)*(1/df_carb_service['time'] ** n)   # in m
            df_carb_service['V_c'] = (df_carb_service['x'] * SA_i)        # in m^3
            df_carb_service = df_carb_service.assign(V_c = [SA_i*wall_thickness if V_c > SA_i*wall_thickness else V_c for V_c in df_carb_service['V_c']])
            # plt.plot(df_carb_service['V_c'])
            df_carb_service['C_s'] = phi_c * C_m * df_carb_service['V_c'] * mass_cement
            df_carb_service['C_s_[Mt]'] = df_carb_service['C_s']/ 1e9

            # end of life (eol) carbonation
            #   assume all is crushed to 10cm spheres
            diam_sphere = 0.1   # m
            volume_remain = SA_i*wall_thickness - int(df_carb_service['V_c'][-1:])
            V_sphere = 4/3 * np.pi * (diam_sphere/2) ** 3
            n_sphere = volume_remain / V_sphere
            SA_i_eol = 4 * np.pi * (diam_sphere/2) ** 2 * n_sphere
            # eol exposure condiitons
            k1_XC4 = 0.41
            n_XC4 = 0.085

            df_carb_eol = pd.DataFrame({'time':np.linspace(1,500-service_life,500-service_life, dtype=int)},
                                       index=np.linspace(service_life+1,500,500-service_life, dtype=int))
            df_carb_eol['x'] = np.sqrt(2*atm_CO2_kgm3*df_carb_eol['time']/R) * np.sqrt(k0*k1_XC4*k2)*(1/df_carb_eol['time'] ** n_XC4)   # in m
            df_carb_eol['V_c'] = (df_carb_eol['x'] * SA_i_eol)        # in m^3
            df_carb_eol = df_carb_eol.assign(V_c = [volume_remain if V_c > volume_remain else V_c for V_c in df_carb_eol['V_c']])
            # plt.plot(df_carb_eol['V_c'])
            df_carb_eol['C_s'] = phi_c * C_m * df_carb_eol['V_c'] * mass_cement
            df_carb_eol['C_s_[Mt]'] = df_carb_eol['C_s']/ 1e9

            # exisitng stock being retired
            V_exist = 0.8 * mass_out[year_label] / dens_conc
            n_sphere_exist = V_exist / V_sphere
            SA_exist = 4 * np.pi * (diam_sphere/2) ** 2 * n_sphere_exist
            df_carb_exist = pd.DataFrame({'time':np.linspace(1,500,500, dtype=int)},
                                         index=np.linspace(1,500,500, dtype=int))
            df_carb_exist['x'] = np.sqrt(2*atm_CO2_kgm3*df_carb_exist['time']/R) * np.sqrt(k0*k1_XC4*k2)*(1/df_carb_exist['time'] ** n_XC4)   # in m
            df_carb_exist['V_c'] = (df_carb_exist['x'] * SA_exist)        # in m^3
            df_carb_exist['C_s'] = phi_c * C_m * df_carb_exist['V_c'] * mass_cement
            df_carb_exist['C_s_[Mt]'] = df_carb_exist['C_s']/ 1e9

        # add them together
            result_carb_uptake_i = pd.concat([df_carb_service, df_carb_eol])
            result_carb_uptake_i[year_label] = result_carb_uptake_i['C_s'] + df_carb_exist['C_s']
            # plt.plot(result_carb_uptake_i['C_s_[Mt]'])

            return result_carb_uptake_i[year_label]


        # initialize output dataframes
        carb_concrete = pd.DataFrame()
        counter=0
        for i in mat_flow_in.iterrows():
            # year = mat_flow_in.loc[i[0]]
            # compute the regrowth for the i_th year
            carb_concrete_i = - carb_uptake(years=year_vec, year_label=i[0],mass_in=mat_flow_in['Sum_conc_inflow'], mass_out=mat_flow_out['Sum_conc_outflow'], service_life=60)
            counter = counter + 1
            # append to the large dataframe
            carb_concrete = carb_concrete.append(carb_concrete_i)

        carb_concerete_T = carb_concrete.T
        # Shift cohort down to appropriate "carbon-year"
        counter = 0
        for column in carb_concerete_T:
            carb_concerete_T[column] = carb_concerete_T[column].shift(counter)
            counter = counter + 1

        # plt.plot(carb_concerete_T)
        total_uptake_carb = carb_concerete_T.sum(axis=1)

    ### ------- CRADLE-TO-GATE EMISSIONS -------

    ### input parameters
    # Embodied carbon coefficient data
    # OLD DATA
    # ECC_CO2_steel = 1.22    # kg CO2e/kg    AISC EPD for hot rolled sections
    # ECC_CH4_steel = 0.01    # DUMMY DATA
    # ECC_CO2_conc = 446.27 / 0.76455 / 2400    # kg CO2e/kg  NRMCA 20% FA 4000psi concrete
    # ECC_CH4_conc = 0.01    # DUMMY DATA
    # ECC_CO2_engwood = 137.19 / 548       # kg CO2e/kg    AWC industry average EPD for glulam
    # ECC_CH4_engwood = 0.01    # DUMMY DATA
    # ECC_CO2_dimlum = 63.12 / 460       # kg CO2e/kg    AWC industry average EPD for softwood lumber
    # ECC_CH4_dimlum = 0.01    # DUMMY DATA
    # ECC_CO2_masonry = 264.16 / 0.76455 / 2400    # kg CO2e/kg  NRMCA 0% FA 3000psi concrete
    # ECC_CH4_masonry = 0.01    # DUMMY DATA
    ## NEW DATA FROM ATHENA
    ECC_CO2_steel_A1A4 = 1.077    # kg CO2e/kg
    ECC_CO2_steel_C1C4 = 0.036    # kg CO2e/kg
    ECC_CO2_conc_A1A4 = 0.161     # kg CO2e/kg
    ECC_CO2_conc_C1C4 = 0.011     # kg CO2e/kg
    ECC_CO2_engwood_A1A4 = 0.197  # kg CO2e/kg
    ECC_CO2_engwood_C1C4 = 0.031  # kg CO2e/kg
    ECC_CO2_dimlum_A1A4 = 0.121   # kg CO2e/kg
    ECC_CO2_dimlum_C1C4= 0.029    # kg CO2e/kg
    ECC_CO2_masonry_A1A4 = 0.138  # kg CO2e/kg
    ECC_CO2_masonry_C1C4 = 0.011  # kg CO2e/kg

    ECC_CH4_steel = 0    # not separating out methane
    ECC_CH4_conc = 0.    # not separating out methane
    ECC_CH4_engwood = 0    # not separating out methane
    ECC_CH4_dimlum = 0    # not separating out methane
    ECC_CH4_masonry = 0    # not separating out methane

    ECC_CO2_vec_A1A4 = [ECC_CO2_steel_A1A4, ECC_CO2_conc_A1A4, ECC_CO2_engwood_A1A4, ECC_CO2_dimlum_A1A4, ECC_CO2_masonry_A1A4]
    ECC_CH4_vec_A1A4 = [ECC_CH4_steel, ECC_CH4_conc, ECC_CH4_engwood, ECC_CH4_dimlum, ECC_CH4_masonry]

    CO2_flux_cradle_to_gate = mat_flow_in * ECC_CO2_vec_A1A4
    CH4_flux_cradle_to_gate = mat_flow_in * ECC_CH4_vec_A1A4

    CO2_flux_cradle_to_gate_total = CO2_flux_cradle_to_gate.sum(axis=1)
    CH4_flux_cradle_to_gate_total = CH4_flux_cradle_to_gate.sum(axis=1)

    # END OF LIFE ECC
    ECC_CO2_vec_C1C4 = [ECC_CO2_steel_C1C4, ECC_CO2_conc_C1C4, ECC_CO2_engwood_C1C4, ECC_CO2_dimlum_C1C4, ECC_CO2_masonry_C1C4]
    CO2_flux_eol = mat_flow_out * ECC_CO2_vec_C1C4
    CO2_flux_eol_total = CO2_flux_eol.sum(axis=1)

    CO2_flux_total = CO2_flux_cradle_to_gate_total + CO2_flux_eol_total

    # Extend flux to go to year 500
    CO2_flux_total_ext = CO2_flux_total.append(pd.DataFrame(np.zeros((417, 1)))).reset_index(drop=True)
    CH4_flux_cradle_to_gate_total_ext = CH4_flux_cradle_to_gate_total.append(pd.DataFrame(np.zeros((417, 1)))).reset_index(drop=True)

    # Add in carbon uptake
    if include_carb == True:
        CO2_flux_total = CO2_flux_total_ext.squeeze() + total_uptake_dimlum + total_uptake_glulam + total_uptake_carb
    else:
        CO2_flux_total = CO2_flux_total_ext.squeeze() + total_uptake_dimlum + total_uptake_glulam


    if plot_emission == True:
        # Plot of emisison vs. uptake for each scenario:

        fig = plt.figure(figsize=(8, 6), dpi=140)
        plt.rcParams.update({'font.size': 8})
        gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.25) # 2x2 grid
        ax0 = fig.add_subplot(gs[0, 0]) # first row, first col
        ax1 = fig.add_subplot(gs[0, 1]) # first row, second col
        ax2 = fig.add_subplot(gs[1, :]) # full second row
        # plot the cradle to gate CO2 emisisons
        ax0.plot(CO2_flux_cradle_to_gate.index, CO2_flux_cradle_to_gate['Sum_steel_inflow'] * 1e-9, label = 'Steel')
        ax0.plot(CO2_flux_cradle_to_gate.index, CO2_flux_cradle_to_gate['Sum_conc_inflow'] * 1e-9, label = 'Concrete')
        ax0.plot(CO2_flux_cradle_to_gate.index, CO2_flux_cradle_to_gate['Sum_engwood_inflow'] * 1e-9, label = 'Eng Wood')
        ax0.plot(CO2_flux_cradle_to_gate.index, CO2_flux_cradle_to_gate['Sum_dimlum_inflow'] * 1e-9, label = 'Dim Lumber')
        ax0.plot(CO2_flux_cradle_to_gate.index, CO2_flux_cradle_to_gate['Sum_masonry_inflow'] * 1e-9, label = 'Masonry')
        ax0.set_ylabel('A1-A4 Emissions (Mt $ CO_2 / year $)')
        # plot the end of life CO2 emisisons
        ax1.plot(CO2_flux_eol.index, CO2_flux_eol['Sum_steel_outflow'] * 1e-9, label = 'Steel')
        ax1.plot(CO2_flux_eol.index, CO2_flux_eol['Sum_conc_outflow'] * 1e-9, label = 'Concrete')
        ax1.plot(CO2_flux_eol.index, CO2_flux_eol['Sum_engwood_outflow'] * 1e-9, label = 'Eng Wood')
        ax1.plot(CO2_flux_eol.index, CO2_flux_eol['Sum_dimlum_outflow'] * 1e-9, label = 'Dim Lumber')
        ax1.plot(CO2_flux_eol.index, CO2_flux_eol['Sum_masonry_outflow'] * 1e-9, label = 'Masonry')
        ax1.set_ylabel('C1-C4 Emissions (Mt $ CO_2 / year $)')
        ax1.legend(loc = 'upper left', fontsize="small");
        # plot the CO2 uptake
        ax2.plot(np.linspace(start=2017,stop=2200,num=2200-2017+1), total_uptake_dimlum[0:184] * 1e-9, label = 'Dim Lumber', color='tab:red')
        ax2.plot(np.linspace(start=2017,stop=2200,num=2200-2017+1), total_uptake_glulam[0:184] * 1e-9, label = 'Eng Wood', color='tab:green')
        ax2.set_ylabel('Uptake (Mt $ CO_2 / year $)')
        if include_carb == True:
            ax2.plot(np.linspace(start=2017,stop=2200,num=2200-2017+1), total_uptake_carb[0:184] * 1e-9, label = 'Conc Carb', color='tab:orange')

        # ax2.legend(loc = 'lower right');
        fig.suptitle('Annual $ CO_2$ Emissions for Scenario: ' + scenario_name)

        plt.savefig('./Figures/DLCA/Emissions_inventory_'+scenario_name+'.png', dpi=240)





    total_flux_df = pd.DataFrame({'CO2_flux':CO2_flux_total,
                            'CH4_flux':CH4_flux_cradle_to_gate_total_ext[0]})

    return total_flux_df

S1_inventory = create_dyn_inventory_scenario(mat_flow_in=S1_i_mean,mat_flow_out=S1_o_mean,scenario_name='S1',include_carb=True,plot_emission=True)
S2_inventory = create_dyn_inventory_scenario(mat_flow_in=S2_i_mean,mat_flow_out=S2_o_mean,scenario_name='S2',include_carb=True,plot_emission=True)
S3_inventory = create_dyn_inventory_scenario(mat_flow_in=S3_i_mean,mat_flow_out=S3_o_mean,scenario_name='S3',include_carb=True,plot_emission=True)
S4_inventory = create_dyn_inventory_scenario(mat_flow_in=S4_i_mean,mat_flow_out=S4_o_mean,scenario_name='S4',include_carb=True,plot_emission=True)
S5_inventory = create_dyn_inventory_scenario(mat_flow_in=S5_i_mean,mat_flow_out=S5_o_mean,scenario_name='S5',include_carb=True,plot_emission=True)
S6_inventory = create_dyn_inventory_scenario(mat_flow_in=S6_i_mean,mat_flow_out=S6_o_mean,scenario_name='S6',include_carb=True,plot_emission=True)
S7_inventory = create_dyn_inventory_scenario(mat_flow_in=S7_i_mean,mat_flow_out=S7_o_mean,scenario_name='S7',include_carb=True,plot_emission=True)


def DLCA_calc(input_flux, time_horizon=100):

    ## Inputs
    fossil_CH4 = False
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
    year_vec = input_flux.index

    CO2_flux = input_flux['CO2_flux']
    CH4_flux = input_flux['CH4_flux']

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
    AGWP_df_sup = pd.read_csv('./AGWP_supp_data.csv')
    AGTP_df_sup = pd.read_csv('./AGTP_supp_data.csv')
    # AGWP_df_sup = pd.read_excel('./DLCA_supp_data.xlsx', sheet_name='AGWP', index_col='year')  # [W yr m^-2]
    # AGTP_df_sup = pd.read_excel('./DLCA_supp_data.xlsx', sheet_name='AGTP', index_col='year')  # [delta-K]

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

    return mass_in_atm_df, irf_df, temp_change_df, equiv_df

S1_mass_in_atm, S1_irf_df, S1_temp_change_df, S1_equiv_df = DLCA_calc(S1_inventory)
S2_mass_in_atm, S2_irf_df, S2_temp_change_df, S2_equiv_df = DLCA_calc(S2_inventory)
S3_mass_in_atm, S3_irf_df, S3_temp_change_df, S3_equiv_df = DLCA_calc(S3_inventory)
S4_mass_in_atm, S4_irf_df, S4_temp_change_df, S4_equiv_df = DLCA_calc(S4_inventory)
S5_mass_in_atm, S5_irf_df, S5_temp_change_df, S5_equiv_df = DLCA_calc(S5_inventory)
S6_mass_in_atm, S6_irf_df, S6_temp_change_df, S6_equiv_df = DLCA_calc(S6_inventory)
S7_mass_in_atm, S7_irf_df, S7_temp_change_df, S7_equiv_df = DLCA_calc(S7_inventory)


### Plot all of the scenarios together

# get the years vector
index_timeline = np.linspace(start=2017, stop=2017+500, num=501, dtype=int)
plot_val = 84
# in Mt
SUMMARY_mass = pd.DataFrame({'S1_CO2':S1_mass_in_atm['CO2'] * 1e-9,
                             'S2_CO2':S2_mass_in_atm['CO2'] * 1e-9,
                             'S3_CO2':S3_mass_in_atm['CO2'] * 1e-9,
                             'S4_CO2':S4_mass_in_atm['CO2'] * 1e-9,
                             'S5_CO2':S5_mass_in_atm['CO2'] * 1e-9,
                             'S6_CO2':S6_mass_in_atm['CO2'] * 1e-9,
                             'S7_CO2':S7_mass_in_atm['CO2'] * 1e-9,
                             'S1_CH4':S1_mass_in_atm['CH4'] * 1e-9,
                             'S2_CH4':S2_mass_in_atm['CH4'] * 1e-9,
                             'S3_CH4':S3_mass_in_atm['CH4'] * 1e-9,
                             'S4_CH4':S4_mass_in_atm['CH4'] * 1e-9,
                             'S5_CH4':S5_mass_in_atm['CH4'] * 1e-9,
                             'S6_CH4':S6_mass_in_atm['CH4'] * 1e-9,
                             'S7_CH4':S7_mass_in_atm['CH4'] * 1e-9,
                             }).set_index(index_timeline)[0:plot_val]

# in W-yr/m2
SUMMARY_irf = pd.DataFrame({'S1_irf':S1_irf_df['IRF_all'],
                            'S2_irf':S2_irf_df['IRF_all'],
                            'S3_irf':S3_irf_df['IRF_all'],
                            'S4_irf':S4_irf_df['IRF_all'],
                            'S5_irf':S5_irf_df['IRF_all'],
                            'S6_irf':S6_irf_df['IRF_all'],
                            'S7_irf':S7_irf_df['IRF_all'],
                            }).set_index(index_timeline)[0:plot_val]

# in delta-K
SUMMARY_temp = pd.DataFrame({'S1_temp':S1_temp_change_df['temp_change'],
                             'S2_temp':S2_temp_change_df['temp_change'],
                             'S3_temp':S3_temp_change_df['temp_change'],
                             'S4_temp':S4_temp_change_df['temp_change'],
                             'S5_temp':S5_temp_change_df['temp_change'],
                             'S6_temp':S6_temp_change_df['temp_change'],
                             'S7_temp':S7_temp_change_df['temp_change'],
                             }).set_index(index_timeline)[0:plot_val]
# in Mt
SUMMARY_equiv = pd.DataFrame({'S1_LCA_dyn':S1_equiv_df['LCA_dyn'] * 1e-9,
                              'S2_LCA_dyn':S2_equiv_df['LCA_dyn'] * 1e-9,
                              'S3_LCA_dyn':S3_equiv_df['LCA_dyn'] * 1e-9,
                              'S4_LCA_dyn':S4_equiv_df['LCA_dyn'] * 1e-9,
                              'S5_LCA_dyn':S5_equiv_df['LCA_dyn'] * 1e-9,
                              'S6_LCA_dyn':S6_equiv_df['LCA_dyn'] * 1e-9,
                              'S7_LCA_dyn':S7_equiv_df['LCA_dyn'] * 1e-9,
                              'S1_LCA_static':S1_equiv_df['LCA_static'] * 1e-9,
                              'S2_LCA_static':S2_equiv_df['LCA_static'] * 1e-9,
                              'S3_LCA_static':S3_equiv_df['LCA_static'] * 1e-9,
                              'S4_LCA_static':S4_equiv_df['LCA_static'] * 1e-9,
                              'S5_LCA_static':S5_equiv_df['LCA_static'] * 1e-9,
                              'S6_LCA_static':S6_equiv_df['LCA_static'] * 1e-9,
                              'S7_LCA_static':S7_equiv_df['LCA_static'] * 1e-9
                              }).set_index(index_timeline)[0:plot_val]

fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(10, 8))
# plot the CO2 emisisons
axes[0,0].axhline(y=0, color='black', linestyle='-')
axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S1_CO2'],color='tab:blue',  linestyle='solid', label = 'S1 (CO2)', )
axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S2_CO2'],color='tab:orange', linestyle='solid', label = 'S2 (CO2)')
axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S3_CO2'],color='tab:green', linestyle='solid', label = 'S3 (CO2)')
axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S4_CO2'],color='tab:red', linestyle='solid', label = 'S4 (CO2)')
axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S5_CO2'],color='tab:purple', linestyle='solid', label = 'S5 (CO2)')
axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S6_CO2'],color='tab:brown', linestyle='solid', label = 'S6 (CO2)')
axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S7_CO2'],color='tab:gray', linestyle='solid', label = 'S7 (CO2)')
# plot the CH4 emisisons
# axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S1_CH4'],color='tab:blue', linestyle='dashed', label = 'S1 (CH4)')
# axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S2_CH4'],color='tab:orange', linestyle='dashed', label = 'S2 (CH4)')
# axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S3_CH4'],color='tab:green', linestyle='dashed', label = 'S3 (CH4)')
# axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S4_CH4'],color='tab:red', linestyle='dashed', label = 'S4 (CH4)')
# axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S5_CH4'],color='tab:purple', linestyle='dashed', label = 'S5 (CH4)')
# axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S6_CH4'],color='tab:brown', linestyle='dashed', label = 'S6 (CH4)')
# axes[0,0].plot(SUMMARY_mass.index, SUMMARY_mass['S7_CH4'],color='tab:gray', linestyle='dashed', label = 'S7 (CH4)')
# Add labels
axes[0,0].set_ylabel('Mass in Atmosphere (Mt)')
axes[0,0].legend(loc = 'lower left', fontsize="small", ncol=1);

# Plot the cumulative GWI (integrated radiative forcing)
axes[0,1].axhline(y=0, color='black', linestyle='-')
axes[0,1].plot(SUMMARY_irf.index, SUMMARY_irf['S1_irf'],color='tab:blue',  linestyle='solid', label = 'S1', )
axes[0,1].plot(SUMMARY_irf.index, SUMMARY_irf['S2_irf'],color='tab:orange', linestyle='solid', label = 'S2')
axes[0,1].plot(SUMMARY_irf.index, SUMMARY_irf['S3_irf'],color='tab:green', linestyle='solid', label = 'S3')
axes[0,1].plot(SUMMARY_irf.index, SUMMARY_irf['S4_irf'],color='tab:red', linestyle='solid', label = 'S4')
axes[0,1].plot(SUMMARY_irf.index, SUMMARY_irf['S5_irf'],color='tab:purple', linestyle='solid', label = 'S5')
axes[0,1].plot(SUMMARY_irf.index, SUMMARY_irf['S6_irf'],color='tab:brown', linestyle='solid', label = 'S6')
axes[0,1].plot(SUMMARY_irf.index, SUMMARY_irf['S7_irf'],color='tab:gray', linestyle='solid', label = 'S7')
axes[0,1].set_ylabel('Cumulative GWI ($W-yr/m^{2}$)')
axes[0,1].legend(loc = 'upper left', fontsize="small", ncol=1);

# Plot the temperature change effect
axes[1,0].axhline(y=0, color='black', linestyle='-')
axes[1,0].plot(SUMMARY_temp.index, SUMMARY_temp['S1_temp'],color='tab:blue',  linestyle='solid', label = 'S1', )
axes[1,0].plot(SUMMARY_temp.index, SUMMARY_temp['S2_temp'],color='tab:orange', linestyle='solid', label = 'S2')
axes[1,0].plot(SUMMARY_temp.index, SUMMARY_temp['S3_temp'],color='tab:green', linestyle='solid', label = 'S3')
axes[1,0].plot(SUMMARY_temp.index, SUMMARY_temp['S4_temp'],color='tab:red', linestyle='solid', label = 'S4')
axes[1,0].plot(SUMMARY_temp.index, SUMMARY_temp['S5_temp'],color='tab:purple', linestyle='solid', label = 'S5')
axes[1,0].plot(SUMMARY_temp.index, SUMMARY_temp['S6_temp'],color='tab:brown', linestyle='solid', label = 'S6')
axes[1,0].plot(SUMMARY_temp.index, SUMMARY_temp['S7_temp'],color='tab:gray', linestyle='solid', label = 'S7')
axes[1,0].set_ylabel('Temperature Change Effect ($\Delta K$)')
axes[1,0].legend(loc = 'lower left', fontsize="small", ncol=1);

# Plot the CO2 equivalences
axes[1,1].axhline(y=0, color='black', linestyle='-')
axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S1_LCA_dyn'],color='tab:blue',  linestyle='solid', label = 'S1 Dynamic', )
axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S2_LCA_dyn'],color='tab:orange', linestyle='solid', label = 'S2 Dynamic')
axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S3_LCA_dyn'],color='tab:green', linestyle='solid', label = 'S3 Dynamic')
axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S4_LCA_dyn'],color='tab:red', linestyle='solid', label = 'S4 Dynamic')
axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S5_LCA_dyn'],color='tab:purple', linestyle='solid', label = 'S5 Dynamic')
axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S6_LCA_dyn'],color='tab:brown', linestyle='solid', label = 'S6 Dynamic')
axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S7_LCA_dyn'],color='tab:gray', linestyle='solid', label = 'S7 Dynamic')
# axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S1_LCA_static'],color='tab:blue',  linestyle='dashed', label = 'S1 Static', )
# axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S2_LCA_static'],color='tab:orange', linestyle='dashed', label = 'S2 Static')
# axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S3_LCA_static'],color='tab:green', linestyle='dashed', label = 'S3 Static')
# axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S4_LCA_static'],color='tab:red', linestyle='dashed', label = 'S4 Static')
# axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S5_LCA_static'],color='tab:purple', linestyle='dashed', label = 'S5 Static')
# axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S6_LCA_static'],color='tab:brown', linestyle='dashed', label = 'S6 Static')
# axes[1,1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S7_LCA_static'],color='tab:gray', linestyle='dashed', label = 'S7 Static')
axes[1,1].set_ylabel('$CO_2$ Equivalencies ($Mt CO_2e$)')
axes[1,1].legend(loc='lower left', fontsize="small", ncol=1);

plt.savefig('./Figures/DLCA/DLCA_output_all.png', dpi=240)


# Plot some of the figures
fig, axes = plt.subplots(1,2, constrained_layout=True, figsize=(10, 6))
plt.rcParams.update({'font.size': 12})
# Plot the temperature change effect
axes[0].axhline(y=0, color='black', linestyle='-')
axes[0].plot(SUMMARY_temp.index, SUMMARY_temp['S1_temp'],color='tab:blue',  linestyle='solid', label = 'S1')
axes[0].plot(SUMMARY_temp.index, SUMMARY_temp['S2_temp'],color='tab:orange', linestyle='solid', label = 'S2')
axes[0].plot(SUMMARY_temp.index, SUMMARY_temp['S3_temp'],color='tab:green', linestyle='solid', label = 'S3')
axes[0].plot(SUMMARY_temp.index, SUMMARY_temp['S4_temp'],color='tab:red', linestyle='solid', label = 'S4')
axes[0].plot(SUMMARY_temp.index, SUMMARY_temp['S5_temp'],color='tab:purple', linestyle='solid', label = 'S5')
axes[0].plot(SUMMARY_temp.index, SUMMARY_temp['S6_temp'],color='tab:brown', linestyle='solid', label = 'S6')
axes[0].plot(SUMMARY_temp.index, SUMMARY_temp['S7_temp'],color='tab:gray', linestyle='solid', label = 'S7')
axes[0].set_ylabel('Temperature Change Effect ($\Delta K$)')
axes[0].legend(loc = 'lower left', fontsize="small", ncol=1);

# Plot the CO2 equivalences
axes[1].axhline(y=0, color='black', linestyle='-')
axes[1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S1_LCA_dyn'],color='tab:blue',  linestyle='solid', label = 'S1 Dynamic', )
axes[1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S2_LCA_dyn'],color='tab:orange', linestyle='solid', label = 'S2 Dynamic')
axes[1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S3_LCA_dyn'],color='tab:green', linestyle='solid', label = 'S3 Dynamic')
axes[1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S4_LCA_dyn'],color='tab:red', linestyle='solid', label = 'S4 Dynamic')
axes[1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S5_LCA_dyn'],color='tab:purple', linestyle='solid', label = 'S5 Dynamic')
axes[1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S6_LCA_dyn'],color='tab:brown', linestyle='solid', label = 'S6 Dynamic')
axes[1].plot(SUMMARY_equiv.index, SUMMARY_equiv['S7_LCA_dyn'],color='tab:gray', linestyle='solid', label = 'S7 Dynamic')
axes[1].set_ylabel('$CO_2$ Equivalencies ($Mt CO_2e$)')

plt.savefig('./Figures/DLCA/DLCA_output_temp_dlca.png', dpi=240)





