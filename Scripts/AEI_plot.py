# First attempt at converting material flows into embodied carbon demands



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import seaborn as sns


# Embodied carbon coefficient data
ECC_steel = 1.22    # kg CO2e/kg    AISC EPD for hot rolled sections
ECC_conc = 446.27 / 0.76455 / 2400    # kg CO2e/kg  NRMCA 20% FA 4000psi concrete
ECC_engwood = 137.19 / 548       # kg CO2e/kg    AWC industry average EPD for glulam
ECC_dimlum = 63.12 / 460       # kg CO2e/kg    AWC industry average EPD for softwood lumber
ECC_masonry = 264.16 / 0.76455 / 2400    # kg CO2e/kg  NRMCA 0% FA 3000psi concrete

# ECC_bio_dimlum = 2042.32 / 460  # kg CO2
ECC_bio_dimlum = 0.39  # kg CO2         BATH v2.0 ICE number
# ECC_bio_engwood = 2120.41 / 544     # kg CO2
ECC_bio_engwood = 0.45     # kg CO2

ECC_vec = [ECC_steel, ECC_conc, ECC_engwood, ECC_dimlum, ECC_masonry, ECC_bio_dimlum, ECC_bio_engwood]

# load in the inflow results of the MFA analysis
S1_i_mean = pd.read_csv('./Results/MFA_results/S1_mat_i_mean.csv')
S2_i_mean = pd.read_csv('./Results/MFA_results/S2_mat_i_mean.csv')
S3_i_mean = pd.read_csv('./Results/MFA_results/S3_mat_i_mean.csv')
S4_i_mean = pd.read_csv('./Results/MFA_results/S4_mat_i_mean.csv')
S5_i_mean = pd.read_csv('./Results/MFA_results/S5_mat_i_mean.csv')
S6_i_mean = pd.read_csv('./Results/MFA_results/S6_mat_i_mean.csv')
S7_i_mean = pd.read_csv('./Results/MFA_results/S7_mat_i_mean.csv')

# duplicate a column for storage in wood
S1_i_mean['Sum_dimlum_inflow_2'] = S1_i_mean['Sum_dimlum_inflow']
S1_i_mean['Sum_engwood_inflow_2'] = S1_i_mean['Sum_engwood_inflow']
S1_i_mean = S1_i_mean.set_index('time', drop=True)
S2_i_mean['Sum_dimlum_inflow_2'] = S2_i_mean['Sum_dimlum_inflow']
S2_i_mean['Sum_engwood_inflow_2'] = S2_i_mean['Sum_engwood_inflow']
S2_i_mean = S2_i_mean.set_index('time', drop=True)
S3_i_mean['Sum_dimlum_inflow_2'] = S3_i_mean['Sum_dimlum_inflow']
S3_i_mean['Sum_engwood_inflow_2'] = S3_i_mean['Sum_engwood_inflow']
S3_i_mean = S3_i_mean.set_index('time', drop=True)
S4_i_mean['Sum_dimlum_inflow_2'] = S4_i_mean['Sum_dimlum_inflow']
S4_i_mean['Sum_engwood_inflow_2'] = S4_i_mean['Sum_engwood_inflow']
S4_i_mean = S4_i_mean.set_index('time', drop=True)
S5_i_mean['Sum_dimlum_inflow_2'] = S5_i_mean['Sum_dimlum_inflow']
S5_i_mean['Sum_engwood_inflow_2'] = S5_i_mean['Sum_engwood_inflow']
S5_i_mean = S5_i_mean.set_index('time', drop=True)
S6_i_mean['Sum_dimlum_inflow_2'] = S6_i_mean['Sum_dimlum_inflow']
S6_i_mean['Sum_engwood_inflow_2'] = S6_i_mean['Sum_engwood_inflow']
S6_i_mean = S6_i_mean.set_index('time', drop=True)
S7_i_mean['Sum_dimlum_inflow_2'] = S7_i_mean['Sum_dimlum_inflow']
S7_i_mean['Sum_engwood_inflow_2'] = S7_i_mean['Sum_engwood_inflow']
S7_i_mean = S7_i_mean.set_index('time', drop=True)


# scale mass by ECC (Mt of CO2e)
S1_EC = S1_i_mean * ECC_vec
S2_EC = S2_i_mean * ECC_vec
S3_EC = S3_i_mean * ECC_vec
S4_EC = S4_i_mean * ECC_vec
S5_EC = S5_i_mean * ECC_vec
S6_EC = S6_i_mean * ECC_vec
S7_EC = S7_i_mean * ECC_vec


# plot the ECC without storage
fig, axs = plt.subplots(2, 4, figsize=(14, 8), sharey=True)
# S2_EC[['Sum_steel_inflow', 'Sum_conc_inflow', 'Sum_engwood_inflow', 'Sum_dimlum_inflow', 'Sum_masonry_inflow']].plot.area()
axs[0, 0].stackplot(S1_EC.index, S1_EC['Sum_steel_inflow'], S1_EC['Sum_conc_inflow'], S1_EC['Sum_engwood_inflow'], S1_EC['Sum_dimlum_inflow'], S1_EC['Sum_masonry_inflow'])
axs[1, 0].stackplot(S2_EC.index, S2_EC['Sum_steel_inflow'], S2_EC['Sum_conc_inflow'], S2_EC['Sum_engwood_inflow'], S2_EC['Sum_dimlum_inflow'], S2_EC['Sum_masonry_inflow'])
axs[0, 1].stackplot(S3_EC.index, S3_EC['Sum_steel_inflow'], S3_EC['Sum_conc_inflow'], S3_EC['Sum_engwood_inflow'], S3_EC['Sum_dimlum_inflow'], S3_EC['Sum_masonry_inflow'])
axs[1, 1].stackplot(S4_EC.index, S4_EC['Sum_steel_inflow'], S4_EC['Sum_conc_inflow'], S4_EC['Sum_engwood_inflow'], S4_EC['Sum_dimlum_inflow'], S4_EC['Sum_masonry_inflow'])
axs[0, 2].stackplot(S5_EC.index, S5_EC['Sum_steel_inflow'], S5_EC['Sum_conc_inflow'], S5_EC['Sum_engwood_inflow'], S5_EC['Sum_dimlum_inflow'], S5_EC['Sum_masonry_inflow'])
axs[1, 2].stackplot(S6_EC.index, S6_EC['Sum_steel_inflow'], S6_EC['Sum_conc_inflow'], S6_EC['Sum_engwood_inflow'], S6_EC['Sum_dimlum_inflow'], S6_EC['Sum_masonry_inflow'])
axs[0, 3].stackplot(S7_EC.index, S7_EC['Sum_steel_inflow'], S7_EC['Sum_conc_inflow'], S7_EC['Sum_engwood_inflow'], S7_EC['Sum_dimlum_inflow'], S7_EC['Sum_masonry_inflow'])
axs[1, 3].axis('off')
legend_text=['Steel', 'Concrete', 'Eng. Wood', 'Dim. Lumber', 'Masonry']
fig.legend(legend_text, title='Materials', loc='lower right', bbox_to_anchor=(0.9, 0.1), fancybox=True)

axs[0, 0].set_title('SSP1 + Low Density')
axs[1, 0].set_title('SSP1 + Medium Density')
axs[0, 1].set_title('SSP1 + High Density')
axs[1, 1].set_title('SSP3 + No Mass Timber')
axs[0, 2].set_title('SSP3 + Moderate Mass Timber')
axs[1, 2].set_title('SSP3 + Low Mass Timber')
axs[0, 3].set_title('SSP2 + Low Mass Timber')

axs[0, 0].set(ylabel='$ Mt CO_2 / year $')
axs[1, 0].set(ylabel='$ Mt CO_2 / year $')


# plot the carbon storage
figure(figsize=(8, 6), dpi=80)
plt.plot(S1_EC.index, S1_EC['Sum_dimlum_inflow_2'] + S1_EC['Sum_engwood_inflow_2'], label = 'SSP1 + Low Density')
plt.plot(S2_EC.index, S2_EC['Sum_dimlum_inflow_2'] + S2_EC['Sum_engwood_inflow_2'], label = 'SSP1 + Medium Density')
plt.plot(S3_EC.index, S3_EC['Sum_dimlum_inflow_2'] + S3_EC['Sum_engwood_inflow_2'], label = 'SSP1 + High Density')
plt.plot(S4_EC.index, S4_EC['Sum_dimlum_inflow_2'] + S4_EC['Sum_engwood_inflow_2'], label = 'SSP3 + No Mass Timber')
plt.plot(S5_EC.index, S5_EC['Sum_dimlum_inflow_2'] + S5_EC['Sum_engwood_inflow_2'], label = 'SSP3 + Moderate Mass Timber')
plt.plot(S6_EC.index, S6_EC['Sum_dimlum_inflow_2'] + S6_EC['Sum_engwood_inflow_2'], label = 'SSP3 + Low Mass Timber')
plt.plot(S7_EC.index, S7_EC['Sum_dimlum_inflow_2'] + S7_EC['Sum_engwood_inflow_2'], label = 'SSP2 + Low Mass Timber')
plt.legend(loc = 'lower left');
plt.ylabel('$ Mt CO_2 / year $')
plt.ylim(0)
plt.title('Annual Carbon Storage of Gravity Structural Systems')



# Sum of the results over the analysis period
sum(S1_EC['Sum_dimlum_inflow_2'] + S1_EC['Sum_engwood_inflow_2'])/1e3
sum(S2_EC['Sum_dimlum_inflow_2'] + S2_EC['Sum_engwood_inflow_2'])/1e3
sum(S3_EC['Sum_dimlum_inflow_2'] + S3_EC['Sum_engwood_inflow_2'])/1e3
sum(S4_EC['Sum_dimlum_inflow_2'] + S4_EC['Sum_engwood_inflow_2'])/1e3
sum(S5_EC['Sum_dimlum_inflow_2'] + S5_EC['Sum_engwood_inflow_2'])/1e3
sum(S6_EC['Sum_dimlum_inflow_2'] + S6_EC['Sum_engwood_inflow_2'])/1e3
sum(S7_EC['Sum_dimlum_inflow_2'] + S7_EC['Sum_engwood_inflow_2'])/1e3




