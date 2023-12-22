import os
import sys
sys.path.append('')
import pandas as pd
import numpy as np

def grid_price_interpolation(grid_prices,site_name,technology_year,project_life,price_units):

    if price_units == 'kWh':
        mwh_to_kwh = 0.001
    else:
        mwh_to_kwh = 1

    operational_year = technology_year + 5
    EOL_year = operational_year + project_life

    grid_prices_site = grid_prices[['Year',site_name]]

    price_interpolated = {}
    for year in range(operational_year,EOL_year):
        if year <= max(grid_prices_site['Year']):
            price_interpolated[year]=np.interp(year,grid_prices_site['Year'],grid_prices_site[site_name])*mwh_to_kwh
        else:
            price_interpolated[year]=grid_prices_site[site_name].values[-1:][0]*mwh_to_kwh

    return price_interpolated

if __name__=="__main__":

    technology_year = 2035
    project_path = os.path.abspath('')
    grid_prices = pd.read_csv(os.path.join(project_path, "H2_Analysis", "annual_average_retail_prices.csv"),index_col = None,header = 0)
    site_name = 'TX'
    project_life = 30

    grid_prices_interpolated = grid_price_interpolation(grid_prices,site_name,technology_year,project_life)

    []

