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
input into the dynamic stock model (__do_stock_driven_model__)
* The results of the dynamic stock model output, showing the inflow, outflow, and stock of buildings over time
are compared against the US DOE data fro, the RECS and CBECS data. A Kolmogorov–Smirnov test is used to evaluate
the difference between the two distributions for building age at different times.
* When the lifetime distribution parameters align with the historical data measured through RECS and CBECS,
the future scenarios for development (SSPs) are computed, saved as .xlsx files and plotted.
 
 