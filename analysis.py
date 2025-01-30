import numpy as np
import pandas as pd
df = pd.read_csv('training_data_vt2025.csv')
print(sum(df['increase_stock']=='high_bike_demand'))
