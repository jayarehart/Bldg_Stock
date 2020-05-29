# Bldg_Stock
Creating a dynamic building stock model for the US.

## Script Workflow
2. dsm_scenario.py
3. HAZUS_ratio_occupancy.py
4. HAZUS_ratio_type.py
5. material_demand.py
### Archived:
* dsm_initial.py


### dsm_scenario.py
**Aim**: Produce a dynamic stock model of floor area for a particular scenario, given a set of population,
GDP, and floor area elasticity inputs. This stock-driven model is derived between the years of 1820 and 2100
 for the US building stock and is calibrated against measured ages of buildings from the US Department of
 Energy (DOE) in various years using the RECS and CBECS studies. The resulting output of the script is a 
 total stock of floor area at each year over the analysis period divided into residential, commercial, and 
 public building space.

**Details**:
* A stock driven model is derived between the years of 1820 and 2100 for the US buildng stock.
* Population and GDP are intepolated based upon historical and forecasted data from IIASA global energy assessment
(GEA) [ref] and the Maddison Project [ref].
* Lifetime distributions for the buildings are assumed to be a Weibull distribution. These distributions are then
input into the dynamic stock model (__do_stock_driven_model__). A DSM is computed for each type of buildings.
* The results of the dynamic stock model output, showing the inflow, outflow, and stock of buildings over time
are compared against the US DOE data fro, the RECS and CBECS data. A Kolmogorov–Smirnov test is used to evaluate
the difference between the two distributions for building age at different times.
* When the lifetime distribution parameters align with the historical data measured through RECS and CBECS,
the future scenarios for development (shared socioeconomic pathways, SSPs) are computed, saved as .xlsx files, and plotted.
* SSP_dsm.xlsx has all the results from the DSM model for all scenarios, given an input for the floor area elasticity.

### HAZUS_ratio_occupancy.py and HAZUS_ratio_type.py
**Aim**: These two scripts manipulate the HAZUS data extracted from the databases to compute the weight of 
each building occupancy (residential, commercial, public) and each building sturcutral type (LF_wood, Mass_Timber,
Steel, Reinforced Concrete, Reinforced Masonry, Unreinforced Masonry, and Mobile Homes).

**Details**: for details of which HAZUS building definitions are used in the analysis, see the table in the
manuscript Supplementary Information (SI).

### material_demand.py
**Aim**: The results from the dsm_scenario and HAZUS scripts are used alongside material intensity data collected
for each structural system to compute the future demand of structural materials under different scenarios. 

**Details**: 
* The total building stock floor area is divided by its structure type.
* The material intensities of each structural system type, aggregated from literature values, is multiplied by the
total inflow of floor area to compute the total demand for a material in a given year.

 
 