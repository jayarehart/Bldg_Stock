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
ECC_bio_dimlum = 0.5*44/12*(1/12)  # kg biogenic CO2         simple conversion
# ECC_bio_dimlum = -0.42           # guest et al. 2013, rotation = 70 years, storage = 80 years
# ECC_bio_engwood = 2120.41 / 544     # kg CO2
ECC_bio_engwood = 0.5*44/12*(1/12)     # biogenic kg CO2     simple conversion
# ECC_bio_engwood = -0.55          #guest et al.  2013, rotation = 40 years, storage = 80 years

ECC_vec = [ECC_steel, ECC_conc, ECC_engwood, ECC_dimlum, ECC_masonry, ECC_bio_dimlum, ECC_bio_engwood]

# load in the inflow results of the MFA analysis
S1_i_mean = pd.read_csv('./Results/MFA_results/S1_mat_i_mean.csv')
S2_i_mean = pd.read_csv('./Results/MFA_results/S2_mat_i_mean.csv')
S3_i_mean = pd.read_csv('./Results/MFA_results/S3_mat_i_mean.csv')
S4_i_mean = pd.read_csv('./Results/MFA_results/S4_mat_i_mean.csv')
S5_i_mean = pd.read_csv('./Results/MFA_results/S5_mat_i_mean.csv')
S6_i_mean = pd.read_csv('./Results/MFA_results/S6_mat_i_mean.csv')
S7_i_mean = pd.read_csv('./Results/MFA_results/S7_mat_i_mean.csv')

# duplicate a column for storage in wood (denoted as '2')
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

# Summary of all EC for each scenario
list_to_sum = ['Sum_steel_inflow', 'Sum_conc_inflow', 'Sum_engwood_inflow', 'Sum_dimlum_inflow', 'Sum_masonry_inflow']
EC_df = pd.concat([S1_EC.loc[:,list_to_sum].sum(axis=1), S2_EC.loc[:,list_to_sum].sum(axis=1), S3_EC.loc[:,list_to_sum].sum(axis=1), S4_EC.loc[:,list_to_sum].sum(axis=1), S5_EC.loc[:,list_to_sum].sum(axis=1), S6_EC.loc[:,list_to_sum].sum(axis=1), S7_EC.loc[:,list_to_sum].sum(axis=1)], axis=1).rename(columns={0:'S1',1:'S2',2:'S3',3:'S4',4:'S5',5:'S6',6:'S7'})
EC_df.to_csv('./Results/EC_out.csv')

# Summary of all EC for each scenario
list_to_sum = ['Sum_steel_inflow', 'Sum_conc_inflow', 'Sum_engwood_inflow', 'Sum_dimlum_inflow', 'Sum_masonry_inflow']
EC_df = pd.concat([S1_EC.loc[:,list_to_sum].sum(axis=1), S2_EC.loc[:,list_to_sum].sum(axis=1), S3_EC.loc[:,list_to_sum].sum(axis=1), S4_EC.loc[:,list_to_sum].sum(axis=1), S5_EC.loc[:,list_to_sum].sum(axis=1), S6_EC.loc[:,list_to_sum].sum(axis=1), S7_EC.loc[:,list_to_sum].sum(axis=1)], axis=1).rename(columns={0:'S1',1:'S2',2:'S3',3:'S4',4:'S5',5:'S6',6:'S7'})
EC_df.to_csv('./Results/EC_storage_out.csv')

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
axs[1, 2].set_title('SSP2 + Moderate Mass Timber')
axs[0, 3].set_title('SSP2 + Low Mass Timber')

axs[0, 0].set(ylabel='$ Mt CO_2 e/ year $')
axs[1, 0].set(ylabel='$ Mt CO_2 e/ year $')

# plot with secondary y-axis

# plot the ECC without storage
fig, axs = plt.subplots(2, 4, figsize=(14, 8), sharey=True)

axs[0, 0].stackplot(S1_EC.index, S1_EC['Sum_steel_inflow'], S1_EC['Sum_conc_inflow'], S1_EC['Sum_engwood_inflow'], S1_EC['Sum_dimlum_inflow'], S1_EC['Sum_masonry_inflow'], alpha=0.8)
ax2 = axs[0, 0].twinx()
ax2.set_ylim(0,14)
ax2.set_yticklabels([])
ax2.plot(S1_EC.index, np.cumsum(S1_EC['Sum_steel_inflow']+S1_EC['Sum_conc_inflow']+S1_EC['Sum_engwood_inflow']+S1_EC['Sum_dimlum_inflow']+S1_EC['Sum_masonry_inflow'])*1e-3, color='dimgray')

axs[1, 0].stackplot(S2_EC.index, S2_EC['Sum_steel_inflow'], S2_EC['Sum_conc_inflow'], S2_EC['Sum_engwood_inflow'], S2_EC['Sum_dimlum_inflow'], S2_EC['Sum_masonry_inflow'], alpha=0.8)
ax2 = axs[1, 0].twinx()
ax2.set_ylim(0,14)
ax2.set_yticklabels([])
ax2.plot(S2_EC.index, np.cumsum(S2_EC['Sum_steel_inflow']+S2_EC['Sum_conc_inflow']+S2_EC['Sum_engwood_inflow']+S2_EC['Sum_dimlum_inflow']+S2_EC['Sum_masonry_inflow'])*1e-3, color='dimgray')

axs[0, 1].stackplot(S3_EC.index, S3_EC['Sum_steel_inflow'], S3_EC['Sum_conc_inflow'], S3_EC['Sum_engwood_inflow'], S3_EC['Sum_dimlum_inflow'], S3_EC['Sum_masonry_inflow'], alpha=0.8)
ax2 = axs[0, 1].twinx()
ax2.set_ylim(0,14)
ax2.set_yticklabels([])
ax2.plot(S3_EC.index, np.cumsum(S3_EC['Sum_steel_inflow']+S3_EC['Sum_conc_inflow']+S3_EC['Sum_engwood_inflow']+S3_EC['Sum_dimlum_inflow']+S3_EC['Sum_masonry_inflow'])*1e-3, color='dimgray')

axs[1, 1].stackplot(S4_EC.index, S4_EC['Sum_steel_inflow'], S4_EC['Sum_conc_inflow'], S4_EC['Sum_engwood_inflow'], S4_EC['Sum_dimlum_inflow'], S4_EC['Sum_masonry_inflow'], alpha=0.8)
ax2 = axs[1, 1].twinx()
ax2.set_ylim(0,14)
ax2.set_yticklabels([])
ax2.plot(S4_EC.index, np.cumsum(S4_EC['Sum_steel_inflow']+S4_EC['Sum_conc_inflow']+S4_EC['Sum_engwood_inflow']+S4_EC['Sum_dimlum_inflow']+S4_EC['Sum_masonry_inflow'])*1e-3, color='dimgray')

axs[0, 2].stackplot(S5_EC.index, S5_EC['Sum_steel_inflow'], S5_EC['Sum_conc_inflow'], S5_EC['Sum_engwood_inflow'], S5_EC['Sum_dimlum_inflow'], S5_EC['Sum_masonry_inflow'], alpha=0.8)
ax2 = axs[0, 2].twinx()
ax2.set_ylim(0,14)
ax2.set_yticklabels([])
ax2.plot(S5_EC.index, np.cumsum(S5_EC['Sum_steel_inflow']+S5_EC['Sum_conc_inflow']+S5_EC['Sum_engwood_inflow']+S5_EC['Sum_dimlum_inflow']+S5_EC['Sum_masonry_inflow'])*1e-3, color='dimgray')

axs[1, 2].stackplot(S6_EC.index, S6_EC['Sum_steel_inflow'], S6_EC['Sum_conc_inflow'], S6_EC['Sum_engwood_inflow'], S6_EC['Sum_dimlum_inflow'], S6_EC['Sum_masonry_inflow'], alpha=0.8)
ax2 = axs[1, 2].twinx()
ax2.set_ylim(0,14)
ax2.set(ylabel='Cumulative Emissions (Gt $CO_2e$)')
ax2.plot(S6_EC.index, np.cumsum(S6_EC['Sum_steel_inflow']+S6_EC['Sum_conc_inflow']+S6_EC['Sum_engwood_inflow']+S6_EC['Sum_dimlum_inflow']+S6_EC['Sum_masonry_inflow'])*1e-3, color='dimgray')

axs[0, 3].stackplot(S7_EC.index, S7_EC['Sum_steel_inflow'], S7_EC['Sum_conc_inflow'], S7_EC['Sum_engwood_inflow'], S7_EC['Sum_dimlum_inflow'], S7_EC['Sum_masonry_inflow'], alpha=0.8)
ax2 = axs[0, 3].twinx()
ax2.set_ylim(0,14)
ax2.set(ylabel='Cumulative Emissions (Gt $CO_2e$)')
ax2.plot(S7_EC.index, np.cumsum(S7_EC['Sum_steel_inflow']+S7_EC['Sum_conc_inflow']+S7_EC['Sum_engwood_inflow']+S7_EC['Sum_dimlum_inflow']+S7_EC['Sum_masonry_inflow'])*1e-3, color='dimgray')
axs[1, 3].axis('off')

legend_text=['Steel', 'Concrete', 'Eng. Wood', 'Dim. Lumber', 'Masonry']
fig.legend(legend_text, title='Materials', loc='lower right', bbox_to_anchor=(0.9, 0.1), fancybox=True)

axs[0, 0].set_title('S1: SSP1 + Low Density')
axs[1, 0].set_title('S2: SSP1 + Medium Density')
axs[0, 1].set_title('S3: SSP1 + High Density')
axs[1, 1].set_title('S4: SSP3 + No Mass Timber')
axs[0, 2].set_title('S5: SSP3 + Moderate Mass Timber')
axs[1, 2].set_title('S6: SSP2 + Moderate Mass Timber')
axs[0, 3].set_title('S7: SSP2 + Low Mass Timber')

axs[0, 0].set(ylabel='$ Mt CO_2 e/ year $')
axs[1, 0].set(ylabel='$ Mt CO_2 e/ year $')
# plt.savefig('./Figures/EC_demand.png', dpi=440)

# Normalize embodied carbon by unit floor space
S1_FA = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP1')[['time','inflow_total']].rename(columns={"inflow_total": "i_million_m2"})
S1_FA = S1_FA.set_index('time', drop=True).loc[2017:]
S2_FA = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP1')[['time','inflow_total']].rename(columns={"inflow_total": "i_million_m2"})
S2_FA = S2_FA.set_index('time', drop=True).loc[2017:]
S3_FA = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP1')[['time','inflow_total']].rename(columns={"inflow_total": "i_million_m2"})
S3_FA = S3_FA.set_index('time', drop=True).loc[2017:]
S4_FA = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP3')[['time','inflow_total']].rename(columns={"inflow_total": "i_million_m2"})
S4_FA = S4_FA.set_index('time', drop=True).loc[2017:]
S5_FA = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP3')[['time','inflow_total']].rename(columns={"inflow_total": "i_million_m2"})
S5_FA = S5_FA.set_index('time', drop=True).loc[2017:]
S6_FA = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP2')[['time','inflow_total']].rename(columns={"inflow_total": "i_million_m2"})
S6_FA = S6_FA.set_index('time', drop=True).loc[2017:]
S7_FA = pd.read_excel('./Results/SSP_dsm.xlsx', sheet_name='SSP2')[['time','inflow_total']].rename(columns={"inflow_total": "i_million_m2"})
S7_FA = S7_FA.set_index('time', drop=True).loc[2017:]

# S1_i_mean = pd.concat([S1_i_mean, S1_FA], axis=1)
# S2_i_mean = pd.concat([S2_i_mean, S2_FA], axis=1)
# S3_i_mean = pd.concat([S3_i_mean, S3_FA], axis=1)
# S4_i_mean = pd.concat([S4_i_mean, S4_FA], axis=1)
# S5_i_mean = pd.concat([S5_i_mean, S5_FA], axis=1)
# S6_i_mean = pd.concat([S6_i_mean, S6_FA], axis=1)
# S7_i_mean = pd.concat([S7_i_mean, S7_FA], axis=1)

# scale results by floor space
S1_norm = S1_EC[['Sum_steel_inflow', 'Sum_conc_inflow', 'Sum_engwood_inflow',
                 'Sum_dimlum_inflow', 'Sum_masonry_inflow']].sum(axis=1) / (S1_FA['i_million_m2']) * 1000
S2_norm = S2_EC[['Sum_steel_inflow', 'Sum_conc_inflow', 'Sum_engwood_inflow',
                 'Sum_dimlum_inflow', 'Sum_masonry_inflow']].sum(axis=1) / (S2_FA['i_million_m2']) * 1000
S3_norm = S3_EC[['Sum_steel_inflow', 'Sum_conc_inflow', 'Sum_engwood_inflow',
                 'Sum_dimlum_inflow', 'Sum_masonry_inflow']].sum(axis=1) / (S3_FA['i_million_m2']) * 1000
S4_norm = S4_EC[['Sum_steel_inflow', 'Sum_conc_inflow', 'Sum_engwood_inflow',
                 'Sum_dimlum_inflow', 'Sum_masonry_inflow']].sum(axis=1) / (S4_FA['i_million_m2']) * 1000
S5_norm = S5_EC[['Sum_steel_inflow', 'Sum_conc_inflow', 'Sum_engwood_inflow',
                 'Sum_dimlum_inflow', 'Sum_masonry_inflow']].sum(axis=1) / (S5_FA['i_million_m2']) * 1000
S6_norm = S6_EC[['Sum_steel_inflow', 'Sum_conc_inflow', 'Sum_engwood_inflow',
                 'Sum_dimlum_inflow', 'Sum_masonry_inflow']].sum(axis=1) / (S6_FA['i_million_m2']) * 1000
S7_norm = S7_EC[['Sum_steel_inflow', 'Sum_conc_inflow', 'Sum_engwood_inflow',
                 'Sum_dimlum_inflow', 'Sum_masonry_inflow']].sum(axis=1) / (S7_FA['i_million_m2']) * 1000

S_norm = pd.concat([S1_norm, S2_norm, S3_norm, S4_norm, S5_norm, S6_norm, S7_norm], axis=1).rename(columns={0:'S1',1:'S2',2:'S3',3:'S4',4:'S5',5:'S6',6:'S7'})
S_norm.to_csv('./Results/EC_intensity_out.csv')
# S_norm.plot()

# plot the carbon storage
fig, axs = plt.subplots(2, figsize=(8, 8))
axs[0].plot(S1_EC.index, S1_EC['Sum_dimlum_inflow_2'] + S1_EC['Sum_engwood_inflow_2'], label = 'S1: SSP1 + Low Density')
axs[0].plot(S2_EC.index, S2_EC['Sum_dimlum_inflow_2'] + S2_EC['Sum_engwood_inflow_2'], label = 'S2: SSP1 + Medium Density')
axs[0].plot(S3_EC.index, S3_EC['Sum_dimlum_inflow_2'] + S3_EC['Sum_engwood_inflow_2'], label = 'S3: SSP1 + High Density')
axs[0].plot(S4_EC.index, S4_EC['Sum_dimlum_inflow_2'] + S4_EC['Sum_engwood_inflow_2'], label = 'S4: SSP3 + No Mass Timber')
axs[0].plot(S5_EC.index, S5_EC['Sum_dimlum_inflow_2'] + S5_EC['Sum_engwood_inflow_2'], label = 'S5: SSP3 + Moderate Mass Timber')
axs[0].plot(S6_EC.index, S6_EC['Sum_dimlum_inflow_2'] + S6_EC['Sum_engwood_inflow_2'], label = 'S6: SSP2 + Moderate Mass Timber')
axs[0].plot(S7_EC.index, S7_EC['Sum_dimlum_inflow_2'] + S7_EC['Sum_engwood_inflow_2'], label = 'S7: SSP2 + Low Mass Timber')
axs[0].set_ylabel('Mt Biogenic $ CO_2 / year $')
# axs[0].set_ylim(0)
axs[0].set_title('Annual Carbon Storage')

axs[1].plot(S1_EC.index, np.cumsum(S1_EC['Sum_dimlum_inflow_2'] + S1_EC['Sum_engwood_inflow_2'])*1e-3, label = 'SSP1 + Low Density')
axs[1].plot(S2_EC.index, np.cumsum(S2_EC['Sum_dimlum_inflow_2'] + S2_EC['Sum_engwood_inflow_2'])*1e-3, label = 'SSP1 + Medium Density')
axs[1].plot(S3_EC.index, np.cumsum(S3_EC['Sum_dimlum_inflow_2'] + S3_EC['Sum_engwood_inflow_2'])*1e-3, label = 'SSP1 + High Density')
axs[1].plot(S4_EC.index, np.cumsum(S4_EC['Sum_dimlum_inflow_2'] + S4_EC['Sum_engwood_inflow_2'])*1e-3, label = 'SSP3 + No Mass Timber')
axs[1].plot(S5_EC.index, np.cumsum(S5_EC['Sum_dimlum_inflow_2'] + S5_EC['Sum_engwood_inflow_2'])*1e-3, label = 'SSP3 + Moderate Mass Timber')
axs[1].plot(S6_EC.index, np.cumsum(S6_EC['Sum_dimlum_inflow_2'] + S6_EC['Sum_engwood_inflow_2'])*1e-3, label = 'SSP2 + Moderate Mass Timber')
axs[1].plot(S7_EC.index, np.cumsum(S7_EC['Sum_dimlum_inflow_2'] + S7_EC['Sum_engwood_inflow_2'])*1e-3, label = 'SSP2 + Low Mass Timber')
axs[1].set_ylabel('Gt Biogenic $ CO_2$')
axs[1].legend(loc = 'upper left', fontsize='small');
axs[1].set_title('Cumulative Carbon Storage')
# plt.savefig('./Figures/Carbon_Storage_simple.png', dpi=440)



# Sum of the results over the analysis period
sum(S1_EC['Sum_dimlum_inflow_2'] + S1_EC['Sum_engwood_inflow_2'])/1e3
sum(S2_EC['Sum_dimlum_inflow_2'] + S2_EC['Sum_engwood_inflow_2'])/1e3
sum(S3_EC['Sum_dimlum_inflow_2'] + S3_EC['Sum_engwood_inflow_2'])/1e3
sum(S4_EC['Sum_dimlum_inflow_2'] + S4_EC['Sum_engwood_inflow_2'])/1e3
sum(S5_EC['Sum_dimlum_inflow_2'] + S5_EC['Sum_engwood_inflow_2'])/1e3
sum(S6_EC['Sum_dimlum_inflow_2'] + S6_EC['Sum_engwood_inflow_2'])/1e3
sum(S7_EC['Sum_dimlum_inflow_2'] + S7_EC['Sum_engwood_inflow_2'])/1e3




