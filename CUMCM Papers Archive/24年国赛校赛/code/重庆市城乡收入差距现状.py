import pandas as pd

urban_rural_data = pd.read_csv('重庆市各年城乡收入差距.csv')

import numpy as np

def calculate_theil(row):
    I1 = row['城镇居民可支配收入']
    I2 = row['农村居民可支配收入']
    P1 = row['城镇常住人口']
    P2 = row['农村常住人口']

    I_total = I1 * P1 + I2 * P2
    P_total = P1 + P2

    I1_ratio = (I1 * P1) / I_total
    I2_ratio = (I2 * P2) / I_total
    P1_ratio = P1 / P_total
    P2_ratio = P2 / P_total

    Theil_1 = I1_ratio * np.log(I1_ratio / P1_ratio)
    Theil_2 = I2_ratio * np.log(I2_ratio / P2_ratio)

    return Theil_1 + Theil_2

urban_rural_data['Theil'] = urban_rural_data.apply(calculate_theil, axis=1) 
urban_rural_data['常住人口城镇化率'] = urban_rural_data['城镇常住人口'] / urban_rural_data['重庆市总人口']

print(urban_rural_data)

