import numpy as np
import pandas as pd
from pathlib import Path

region_no = '18'
basins_file = f'/data2/wzx/basin_set/GNN-RNN/data/area{region_no}.txt'
forcing_cols = ["Year", "Mnth", "Day", "Hr", "dayl(s)", "prcp(mm/day)", "srad(W/m2)",
                "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
features = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"]
discharge_cols = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]

camels_root = '/data2/wzx/basin_set/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet'
camels_root = Path(camels_root)
camels_discharge = Path('/data2/wzx/basin_set/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/usgs_streamflow')
basin_list = pd.read_csv(basins_file,header=None,dtype=str)[0].values.tolist()
# 加载数据
region_data = None
flag = 0
for basin_no in basin_list:
    files = list(camels_root.glob(f'**/{basin_no}_*.txt'))
    files_discharge = list(camels_discharge.glob(f'**/{basin_no}_*.txt'))
    file_path = files[0]
    files_discharge_path = files_discharge[0]
    df_hydro = pd.read_csv(file_path,sep=r"\s+",header=3)
    # discharge 数据
    df_discharge = pd.read_csv(files_discharge_path,sep=r'\s+',header=None,names=discharge_cols)
    df_fea = df_hydro[features]
    df_discharge = df_discharge["QObs"]
    df = pd.concat([df_fea,df_discharge],axis=1)
    df = df.dropna()
    # print(df)
    df = df.drop((df[df["QObs"] < 0]).index)
    df_mean = df.mean(axis=0)
    region_data = pd.concat((region_data,df_mean),axis=1)
    # region_data = pd.concat((region_data,df),axis=0,join='outer',ignore_index=True)
print('-----------------mean-------------------')
print(region_data.mean(axis=1))
print('-----------------std---------------------')
print(region_data.std(axis=1))


