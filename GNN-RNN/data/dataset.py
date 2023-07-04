import os.path
from bisect import bisect_right

import numpy as np
import pandas as pd
import torch
from abc import ABCMeta, abstractmethod, ABC
from torch.utils.data import Dataset
from pathlib import Path
from configs.pretrain_config import PretrainConfig


class AbstractReader(metaclass=ABCMeta):
    '''
    abstract data reading class
    abstract methods are implemented in subclasses
    '''

    @abstractmethod
    def _load_data(self, *args, **kwargs):
        # load data
        pass

    @abstractmethod
    def _process_invalid_data(self, *args, **kwargs):
        # process invalid data
        pass

    @abstractmethod
    def get_df_x(self):
        pass

    @abstractmethod
    def get_df_y(self):
        pass


# 读取静态数据抽象类
class AbstractStaticReader:
    '''
    read file from static property
    select the attributes to use for normalization
    '''

    @abstractmethod
    def get_df_static(self, basin):
        pass


# read daymet hydrological data class
class DaymetHydroReader(AbstractReader):
    camels_root = None
    forcing_root = None
    discharge_root = None
    forcing_cols = ["Year", "Mnth", "Day", "Hr", "dayl(s)", "prcp(mm/day)", "srad(W/m2)",
                    "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
    features = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"]
    discharge_cols = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
    target = ["QObs(mm/d)"]

    def __init__(self, basin: str):
        self.basin = basin
        self.area = None
        df = self._load_data()
        df = self._process_invalid_data(df)
        self.df_x = df[self.features]
        self.df_y = df[self.target]

    @classmethod
    def init_root(cls, camels_root):
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "daymet"
        cls.discharge_root = cls.camels_root / "usgs_streamflow"

    def _load_data(self, *args, **kwargs):
        df_forcing = self.load_forcing()
        df_discharge = self.load_discharge()
        df = pd.concat([df_forcing, df_discharge], axis=1)
        return df

    def get_df_x(self):
        return self.df_x

    def get_df_y(self):
        return self.df_y

    # load weather data
    def load_forcing(self):
        files = list(self.forcing_root.glob(f"**/{self.basin}_*.txt"))
        if len(files) == 0:
            raise RuntimeError(f"No forcing file found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant forcing files found for Basin {self.basin}")
        else:
            file_path = files[0]

        df = pd.read_csv(file_path, sep=r"\s+", header=3)
        dates = df.Year.map(str) + '/' + df.Mnth.map(str) + '/' + df.Day.map(str)
        df_index = pd.to_datetime(dates, format="%Y/%m/%d")

        df = df.set_index(df_index)

        with open(file_path) as fp:
            fp.readline()
            fp.readline()
            content = fp.readline().strip()
        self.area = int(content)

        return df[self.features]

    def load_discharge(self):
        files = list(self.discharge_root.glob(f"**/{self.basin}_*.txt"))
        if len(files) == 0:
            raise RuntimeError(f"No discharge found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant discharge files found for Basin {self.basin}")
        else:
            file_path = files[0]

        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=self.discharge_cols)
        dates = df.Year.map(str) + '/' + df.Mnth.map(str) + '/' + df.Day.map(str)
        df_index = pd.to_datetime(dates, format="%Y/%m/%d")

        df = df.set_index(df_index)

        assert len(self.target) == 1
        df[self.target[0]] = 28316846.592 * df["QObs"] * 86400 / (self.area * 10 ** 6)
        return df[self.target]

    def _process_invalid_data(self, df: pd.DataFrame):
        len_raw = len(df)
        print(len_raw)
        df = df.dropna()
        len_drop_nan = len(df)
        print(len_drop_nan)
        if len_raw > len_drop_nan:
            print(f"Delete {len_raw - len_drop_nan} rows because of nan {self.basin}")
        df = df.drop((df[df["QObs(mm/d)"] < 0]).index)
        return df


# maurer
class MaurerHydroReader(DaymetHydroReader):
    camels_root = None
    forcing_root = None
    discharge_root = None

    @classmethod
    def init_root(cls, camels_root):
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "maurer"
        cls.discharge_root = cls.forcing_root / "usgs_streamflow"

    def __init__(self, basin: str):
        super().__init__(basin)


# nldas
class NldasHydroReader(DaymetHydroReader):
    camels_root = None
    forcing_root = None
    discharge_root = None

    @classmethod
    def init_root(cls, camels_root):
        cls.camels_root = Path(camels_root)
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "nldas"
        cls.discharge_root = cls.forcing_root / "usgs_streamflow"

    def __init__(self, basin: str):
        super().__init__(basin)


class HydroReaderFactory:
    @staticmethod
    def hydro_reader(camels_root, forcing_type, basin):
        if forcing_type == "daymet":
            DaymetHydroReader.init_root(camels_root)
            reader = DaymetHydroReader(basin)
        elif forcing_type == "maurer_extended":
            MaurerHydroReader.init_root(camels_root)
            reader = MaurerHydroReader(basin)
        elif forcing_type == "nldas_extended":
            NldasHydroReader.init_root(camels_root)
            reader = NldasHydroReader(basin)
        else:
            raise RuntimeError(f"No such hydro type:{basin}")
        return reader



class CamelsDataset(Dataset):
    '''
    Attributes：
        camels_root:str
            the path to the dataset camels
        basin_list:list of str
            contains 8-digit codes for all catchment areas
        past_len:int
            the length of the past time step discharge data
        pred_len:int
            the length of the future time step discharge data
        stage:str
            [train,val,test]
        dates:list of pd.DateTimes
            [start_time,end_time]
        x_dict:dict as {basin:np.ndarray}
            meteorological data
        y_dict:dict as {basin:np.array}
            discharge data
        length_ls:list of int
        index_ls:list of int
        x_mean:np.ndarray
        y_mean:np.ndarray
        x_std:np.ndarray
        y_std:np.ndarray
    '''

    def __init__(self, camels_root: str, forcing_type: str, basin_list: list, past_len: int, pred_len: int,
                 stage: str, dates: list,
                 x_mean: np.ndarray, y_mean: np.ndarray, x_std: np.ndarray, y_std: np.ndarray, y_stds_dict=None):
        self.camels_root = camels_root
        self.forcing_type = forcing_type
        self.basin_list = basin_list
        self.past_len = past_len
        self.pred_len = pred_len
        self.stage = stage
        self.dates = dates
        self.x_dict = dict()
        self.y_dict = dict()
        self.length_ls = list()
        self.date_index_dict = dict()

        if y_stds_dict is None:
            self.y_stds_dict = dict()
        else:
            self.y_stds_dict = y_stds_dict

        # normalize
        self._load_data(forcing_type)

        # calc mean and std
        if self.stage == 'train':
            self.x_mean, self.x_std = self.calc_mean_std(self.x_dict)
            self.y_mean, self.y_std = self.calc_mean_std(self.y_dict)
        else:
            self.x_mean = x_mean
            self.x_std = x_std
            self.y_mean = y_mean
            self.y_std = y_std

        # normalize
        self.normalize_data()

        # test
        self.a = self.length_ls[0]
        self.num_samples = self.length_ls[0]

        self.index_ls = [0]
        for i in range(len(self.length_ls)):
            v = self.index_ls[i] + self.length_ls[i]
            self.index_ls.append(v)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        '''
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basin_list[basin_idx]
        # x_seq:(512,22,5)
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]
        '''

        # basin_idx = bisect_right(self.index_ls, idx) - 1

        local_idx = idx
        # x_seq_dic = dict()
        # y_seq_past_dic = dict()
        # y_seq_future_dic = dict()
        x_seq_dic = torch.tensor([])
        y_seq_past_dic = torch.tensor([])
        y_seq_future_dic = torch.tensor([])
        y_std_dic = dict
        loop_len = len(self.basin_list)

        for i in range(loop_len):
            basin = self.basin_list[i]
            # basin = '07056000'
            # basin = self.basin_list[basin_idx]
            # x_seq:(16,5)
            x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
            x_seq = torch.tensor(x_seq).unsqueeze(dim=0)
            y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
            y_seq_past = torch.tensor(y_seq_past).unsqueeze(dim=0)
            y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]
            y_seq_future = torch.tensor(y_seq_future).unsqueeze(dim=0)
            # x_seq = x_seq.squeeze
            x_seq_dic = torch.cat((x_seq_dic,x_seq),dim=0)
            y_seq_past_dic = torch.cat((y_seq_past_dic,y_seq_past),dim=0)
            y_seq_future_dic = torch.cat((y_seq_future_dic,y_seq_future),dim=0)
            '''
            x_seq_dic[i] = x_seq
            y_seq_past_dic[i] = y_seq_past
            y_seq_future_dic[i] = y_seq_future
            # y_std_dic[i] = self.y_stds_dict[i]
            '''
        return x_seq_dic, y_seq_past_dic, y_seq_future_dic

    @staticmethod
    def calc_mean_std(data_dict: dict):
        data_all = np.concatenate(list(data_dict.values()), axis=0)
        nan_mean = np.nanmean(data_all, axis=0)
        nan_std = np.nanstd(data_all, axis=0)
        return nan_mean, nan_std

    def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'input':
            feature = (feature - self.x_mean) / self.x_std
        elif variable == 'output':
            feature = (feature - self.y_mean) / self.y_std
        else:
            raise RuntimeError(f"No such variable:{variable}")
        return feature

    def normalize_data(self):
        for idx, basin in enumerate(self.basin_list):
            print(self.stage, f"Normalizing %4f" % (idx / len(self.basin_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            x_norm = self._local_normalization(x, 'input')
            y_norm = self._local_normalization(y, 'output')
            self.x_dict[basin] = x_norm
            self.y_dict[basin] = y_norm

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'input':
            feature = (feature * self.x_std) + self.x_mean
        elif variable == 'output':
            feature = (feature * self.y_std) + self.y_mean
        else:
            raise RuntimeError(f"No such variable:{variable}")
        return feature

    def get_means(self):
        return self.x_mean, self.y_mean

    def get_stds(self):
        return self.x_std, self.y_std

    def _load_data(self, forcing_type):
        basin_number = len(self.basin_list)
        for idx, basin in enumerate(self.basin_list):
            print(self.stage, f"{basin} Loading data %.4f" % (idx / basin_number))
            reader = HydroReaderFactory.hydro_reader(self.camels_root, forcing_type, basin)
            df_x = reader.get_df_x()
            df_y = reader.get_df_y()

            df_x = df_x[self.dates[0]:self.dates[1]]
            df_y = df_y[self.dates[0]:self.dates[1]]
            assert len(df_x) == len(df_y)
            self.date_index_dict[basin] = df_x.index

            x = df_x.values.astype('float32')
            y = df_y.values.astype('float32')
            self.x_dict[basin] = x
            self.y_dict[basin] = y


            self.length_ls.append(len(x) - self.past_len - self.pred_len + 1)
            if self.stage == 'train':
                self.y_stds_dict[basin] = y.std(axis=0).item()

    @classmethod
    def get_instance(cls, past_len: int, pred_len: int, stage: str, specific_cfg: dict, x_mean=None, y_mean=None,
                     x_std=None, y_std=None, y_stds_dic=None):
        final_data_path = specific_cfg['final_data_path']
        camels_root = specific_cfg['camels_root']
        basin_list = specific_cfg['basin_list']
        forcing_type = specific_cfg['forcing_type']
        start_date = specific_cfg['start_date']
        end_date = specific_cfg['end_date']
        if final_data_path is None:
            dates = [start_date, end_date]
            instance = cls(camels_root, forcing_type, basin_list, past_len, pred_len, stage, dates, x_mean, y_mean,
                           x_std, y_std, y_stds_dic)
            return instance
        else:
            if final_data_path.exists():
                instance = torch.load(final_data_path)
                return instance
            else:
                dates = [start_date, end_date]
                instance = cls(camels_root, forcing_type, basin_list, past_len, pred_len, stage, dates, x_mean, y_mean,
                               x_std, y_std, y_stds_dic)
                final_data_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(instance, final_data_path)
                return instance


# read static data
class StaticReader(AbstractStaticReader):
    '''
    normalize attribute
    '''

    def __init__(self, camels_root):
        self.camels_root = camels_root
        self.static_file_path = Path(
            "CAMELS/CAMELS-US") / "camels_attributes_v2.0" / "selected_norm_static_attributes.csv"
        self.df_static = pd.read_csv(self.static_file_path, header=0, dtype={"gauge_id": str}).set_index("gauge_id")
        self.df_static = self.df_static.astype(dtype='float16')

    def get_df_static(self, basin):
        return self.df_static.loc[[basin]].values


class CamelsDatasetWithStatic(CamelsDataset):
    # add static data
    def __init__(self, camels_root: str, forcing_type: str, basins_list: list, past_len: int, pred_len: int, stage: str,
                 dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        # instantiate
        self.static_reader = StaticReader(camels_root)
        self.norm_static_fea = dict()
        super().__init__(camels_root, forcing_type, basins_list, past_len, pred_len, stage, dates, x_mean, y_mean,
                         x_std, y_std, y_stds_dict)

    def _load_data(self, forcing_type):
        basin_number = len(self.basin_list)
        for idx, basin in enumerate(self.basin_list):
            print(self.stage, f"{basin} Loading data %.4f" % (idx / basin_number))
            reader = HydroReaderFactory.hydro_reader(self.camels_root, forcing_type, basin)
            df_x = reader.get_df_x()
            df_y = reader.get_df_y()

            df_x = df_x[self.dates[0]:self.dates[1]]
            df_y = df_y[self.dates[0]:self.dates[1]]
            assert len(df_x) == len(df_y)
            self.date_index_dict[basin] = df_x.index

            x = df_x.values.astype('float32')
            y = df_y.values.astype('float32')
            self.x_dict[basin] = x
            self.y_dict[basin] = y

            self.length_ls.append(len(x) - self.past_len - self.pred_len + 1)
            self.norm_static_fea[basin] = self.static_reader.get_df_static(basin)
            if self.stage == 'train':
                self.y_stds_dict[basin] = y.std(axis=0).item()

    def normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basin_list):
            print(self.stage, "Normalizing %.4f" % (idx / len(self.basin_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            # Normalize data
            x_norm = self._local_normalization(x, variable='input')
            y_norm = self._local_normalization(y, variable='output')
            norm_static_fea = self.norm_static_fea[basin].repeat(x_norm.shape[0], axis=0)
            x_norm_static = np.concatenate([x_norm, norm_static_fea], axis=1)
            self.x_dict[basin] = x_norm_static
            self.y_dict[basin] = y_norm

    def __getitem__(self, idx: int):
        local_idx = idx
        x_seq_dic = torch.tensor([])
        y_seq_past_dic = torch.tensor([])
        y_seq_future_dic = torch.tensor([])
        y_std_dic = dict
        loop_len = len(self.basin_list)

        for i in range(loop_len):
            basin = self.basin_list[i]
            x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
            x_seq = torch.tensor(x_seq).unsqueeze(dim=0)
            y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
            y_seq_past = torch.tensor(y_seq_past).unsqueeze(dim=0)
            y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]
            y_seq_future = torch.tensor(y_seq_future).unsqueeze(dim=0)
            # x_seq = x_seq.squeeze
            x_seq_dic = torch.cat((x_seq_dic,x_seq),dim=0)
            y_seq_past_dic = torch.cat((y_seq_past_dic,y_seq_past),dim=0)
            y_seq_future_dic = torch.cat((y_seq_future_dic,y_seq_future),dim=0)
            '''
            x_seq_dic[i] = x_seq
            y_seq_past_dic[i] = y_seq_past
            y_seq_future_dic[i] = y_seq_future
            # y_std_dic[i] = self.y_stds_dict[i]
            '''
        return x_seq_dic, y_seq_past_dic, y_seq_future_dic

class CamelsDatasetLimited(CamelsDataset):
    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basin_list[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]


class DatasetFactory:
    @staticmethod
    def get_dataset_type(use_future_fea, use_static):
        if (not use_future_fea) and use_static:
            raise RuntimeError("No implemented yet.")
        elif not use_future_fea:
            ds = CamelsDatasetLimited
        elif use_static:
            ds = CamelsDatasetWithStatic
        else:
            ds = CamelsDataset
        return ds

