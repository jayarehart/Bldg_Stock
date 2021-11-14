

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
import numpy as np
import seaborn as sns


plot_emission=True
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
        result = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((years- mu)/ sigma) ** 2) * ECC_bio_wood
    if curve == 'Richards':
        # Richards growth curve: another model for biomass growth centered around half of the rotation period.
        growth_rate=0.2
        M = rot_period / 2
        A = years[0]
        B = ECC_bio_wood
        # logistic growth rate:
        t = np.linspace(years[0], years[-1], num=(years[-1] - years[0] + 1), endpoint=True)
        y = A + (B - A) / (1 +  np.exp(-growth_rate * (t-M)))
        result_pre = pd.DataFrame({year_label:y}).diff()
        result_pre[year_label][0] = 0
        result = result_pre[year_label]
    return result

total_uptake_dimlum = biomass_regrowth_i(years=year_vec, year_label=2017, rot_period=rot_period_glulam, bio_CO2_coeff=ECC_bio_wood, curve='Normal')
total_uptake_glulam = biomass_regrowth_i(years=year_vec, year_label=2017, rot_period=rot_period_dimlum, bio_CO2_coeff=ECC_bio_wood, curve='Normal')


### ------- CARBONATION MODEL -------

# debug
service_life=60
year_label=2017
mass_in=1

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
dens_conc = 2400        # kg/m3
vol_conc = mass_in / dens_conc
SA_i = vol_conc*6    # m2

df_carb_service = pd.DataFrame({'time':np.linspace(0,service_life,service_life+1, dtype=int)}).set_index('time', drop=False)
df_carb_service['x'] = np.sqrt(2*atm_CO2_kgm3*df_carb_service['time']/R) * np.sqrt(k0*k1*k2)*(1/df_carb_service['time'] ** n)   # in m
df_carb_service['V_c'] = (df_carb_service['x'] * SA_i)        # in m^3
df_carb_service = df_carb_service.assign(V_c = [vol_conc if V_c > vol_conc else V_c for V_c in df_carb_service['V_c']])
# plt.plot(df_carb_service['V_c'])
df_carb_service['C_s'] = phi_c * C_m * df_carb_service['V_c'] * mass_cement

# end of life (eol) carbonation
#   assume all is crushed to 5cm spheres
diam_sphere = 0.05   # m
volume_remain = vol_conc - float(df_carb_service['V_c'][-1:])
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

# add them together
result_carb_uptake_i = pd.concat([df_carb_service, df_carb_eol])
result_carb_uptake_i[year_label] = result_carb_uptake_i['C_s']
total_uptake_carb = result_carb_uptake_i['C_s']
# plt.plot(result_carb_uptake_i[2017])

### ------- CRADLE-TO-GATE EMISSIONS -------
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

CO2_flux_cradle_to_gate = mass_in * ECC_CO2_vec_A1A4

length_df = 150
CO2_flux_long = pd.DataFrame({'steel':np.zeros(length_df),'conc':np.zeros(length_df),'engwood':np.zeros(length_df),'dimlum':np.zeros(length_df),'masonry':np.zeros(length_df)}, index=np.linspace(0,length_df-1,length_df, dtype=int))
CO2_flux_long.iloc[0]= CO2_flux_long.iloc[0] + ECC_CO2_vec_A1A4
CO2_flux_long.loc[1:,'conc'] = CO2_flux_long.iloc[1:]['conc'] - total_uptake_carb[1:length_df]
CO2_flux_long.loc[1:,'engwood'] = CO2_flux_long.iloc[1:]['conc'] - total_uptake_glulam[1:length_df]
CO2_flux_long.loc[1:,'dimlum'] = CO2_flux_long.iloc[1:]['conc'] - total_uptake_dimlum[1:length_df]

CO2_flux_long.cumsum().plot()

##############################################################################################################
######################                                                                  ######################
######################       START HERE FOR PICKING UP THIS IF WE FEEL IT'S NECESSARY   ######################
######################       NEED TO add in end of life elements too, something wrong with uptake calcs too...
##############################################################################################################

# END OF LIFE ECC
ECC_CO2_vec_C1C4 = [ECC_CO2_steel_C1C4, ECC_CO2_conc_C1C4, ECC_CO2_engwood_C1C4, ECC_CO2_dimlum_C1C4, ECC_CO2_masonry_C1C4]
CO2_flux_eol = mass_in * ECC_CO2_vec_C1C4

CO2_flux_total = CO2_flux_cradle_to_gate_total + CO2_flux_eol_total

# Extend flux to go to year 500
CO2_flux_total_ext = CO2_flux_total.append(pd.DataFrame(np.zeros((417, 1)))).reset_index(drop=True)

# Add in carbon uptake
CO2_flux_total = CO2_flux_total_ext.squeeze() + total_uptake_dimlum + total_uptake_glulam + total_uptake_carb


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

]