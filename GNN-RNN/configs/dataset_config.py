import pandas as pd
from pathlib import Path


class DataSetConfig:
    # Camels dataset absolute path
    camels_root = Path("basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2")
    forcing_type = "daymet" # TODO: Daymet
    basin_mark = "673"
    basins_file = 'file_path'
    global_basin_list = pd.read_csv(basins_file,header=None,dtype=str)[0].values.tolist()


    # TODO:Daymet
    train_start = pd.to_datetime("1980-10-01",format="%Y-%m-%d")
    train_end = pd.to_datetime("1995-09-30",format="%Y-%m-%d")
    valid_start = pd.to_datetime("1995-10-01",format="%Y-%m-%d")
    valid_end = pd.to_datetime("2000-09-30",format="%Y-%m-%d")
    test_start = pd.to_datetime("2000-10-01",format="%Y-%m-%d")
    test_end = pd.to_datetime("2014-09-30",format="%Y-%m-%d")
    dataset_info = f"{forcing_type}{basin_mark}_{train_start.year}-{train_end.year}_{valid_start.year}" \
                f"-{valid_end.year}_{test_start.year}-{test_end.year}"


class DataShapeConfig:
    # data shape
    past_len = 15  # data for the past 15 days
    pred_len = 1   # data for the future 1 day
    use_feature = True    # whether to use future data
    if use_feature:
        src_len = past_len + pred_len
    else:
        src_len = past_len
    tgt_len = past_len + pred_len

    dynamic_size = 5
    static_size = 27
    # whether to use static data
    use_static = False
    if use_static:
        src_size = dynamic_size + static_size    # 5+27
    else:
        src_size = dynamic_size    # 5
    tgt_size = 1

    data_shape_info = f"{src_len}+{past_len}+{pred_len}[{src_size}+{tgt_size}]"