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
    抽象数据读取类
    确保将元数据读取为pd.DataFrame
    抽象方法要在子类中实现
    '''

    @abstractmethod
    def _load_data(self, *args, **kwargs):
        # 加载数据抽象方法
        pass

    @abstractmethod
    def _process_invalid_data(self, *args, **kwargs):
        # 处理无效的数据
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
    从静态属性文件中读取文件（csv）
    选择使用的属性 做归一化处理
    将属性转为pd.DataFrame
    '''

    @abstractmethod
    def get_df_static(self, basin):
        # 返回特定流域的静态属性pd.DataFrame
        pass


# 读取Daymet水文数据类
class DaymetHydroReader(AbstractReader):
    camels_root = None
    forcing_root = None
    discharge_root = None
    # 数据集中的列名
    forcing_cols = ["Year", "Mnth", "Day", "Hr", "dayl(s)", "prcp(mm/day)", "srad(W/m2)",
                    "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
    features = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"]
    discharge_cols = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
    # 目标结果
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
        # 对应的输入
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "daymet"
        # cls.discharge_root = cls.forcing_root / "usgs_streamflow"
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

    # 加载气象数据
    def load_forcing(self):
        files = list(self.forcing_root.glob(f"**/{self.basin}_*.txt"))
        if len(files) == 0:
            raise RuntimeError(f"No forcing file found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant forcing files found for Basin {self.basin}")
        else:
            # 获取对应集水区的文件路径
            file_path = files[0]

        # 读取数据并将datetime作为数据集的索引，将第四行作为列属性
        df = pd.read_csv(file_path, sep=r"\s+", header=3)
        # 将日期转为字符串进行拼接
        dates = df.Year.map(str) + '/' + df.Mnth.map(str) + '/' + df.Day.map(str)
        # 将参数转化为日期时间对象
        df_index = pd.to_datetime(dates, format="%Y/%m/%d")

        # 将日期设置为索引
        df = df.set_index(df_index)

        # 读取流域代码
        with open(file_path) as fp:
            fp.readline()
            fp.readline()
            content = fp.readline().strip()
        self.area = int(content)

        # 以上述的df_index为索引，返回self.features列的数据，返回类型为df.DataFrame
        return df[self.features]

    # 加载径流数据
    def load_discharge(self):
        files = list(self.discharge_root.glob(f"**/{self.basin}_*.txt"))
        if len(files) == 0:
            raise RuntimeError(f"No discharge found for Basin {self.basin}")
        elif len(files) >= 2:
            raise RuntimeError(f"Redundant discharge files found for Basin {self.basin}")
        else:
            file_path = files[0]

        # 读取数据，将datetime作为索引，转为DataFrame类型
        # 设置names = self.discharge_cols作为表头
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=self.discharge_cols)
        # 将日期转为字符串进行拼接
        dates = df.Year.map(str) + '/' + df.Mnth.map(str) + '/' + df.Day.map(str)
        # 将参数转化为日期时间对象
        df_index = pd.to_datetime(dates, format="%Y/%m/%d")

        # 将日期时间设置为索引
        df = df.set_index(df_index)

        assert len(self.target) == 1
        # 将流量从每秒多少立方英尺转换为每天多少毫米
        # 1立方英尺 = 28316846.592立方毫米
        df[self.target[0]] = 28316846.592 * df["QObs"] * 86400 / (self.area * 10 ** 6)
        return df[self.target]

    # 处理无效的数据
    def _process_invalid_data(self, df: pd.DataFrame):
        # 删除数据集discharge中存在Nan的行
        len_raw = len(df)
        print(len_raw)
        df = df.dropna()
        len_drop_nan = len(df)
        print(len_drop_nan)
        if len_raw > len_drop_nan:
            print(f"Delete {len_raw - len_drop_nan} rows because of nan {self.basin}")
        df = df.drop((df[df["QObs(mm/d)"] < 0]).index)
        return df


# 读取其它类型的数据，只是数据存放的路径不同，方法和上述相同，可以用过继承上述类实现相同的操作
# maurer
# 当使用maurer数据集时调用此类，和daymet中相同
class MaurerHydroReader(DaymetHydroReader):
    camels_root = None
    forcing_root = None
    discharge_root = None

    @classmethod
    def init_root(cls, camels_root):
        cls.camels_root = Path(camels_root)
        # 对应的输入
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "maurer"
        # 仍然使用这个观测的数据集进行处理
        cls.discharge_root = cls.forcing_root / "usgs_streamflow"

    # 在该类中重写构造方法，在构造方法中调用DaymetHydroReader中的构造方法，进行相同的初始化过程
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
        # 对应的输入
        cls.forcing_root = cls.camels_root / "basin_mean_forcing" / "nldas"
        # 仍然使用这个观测的数据集进行处理
        cls.discharge_root = cls.forcing_root / "usgs_streamflow"

    # 在该类中重写构造方法，在构造方法中调用DaymetHydroReader中的构造方法，进行相同的初始化过程
    def __init__(self, basin: str):
        super().__init__(basin)


# 水文数据读取类
# 依据不同的标签读取不同的数据集
class HydroReaderFactory:
    @staticmethod
    def hydro_reader(camels_root, forcing_type, basin):
        # 返回的是用DaymetHydroReader实例化出的一个对象
        if forcing_type == "daymet":
            DaymetHydroReader.init_root(camels_root)
            # 实例化出一个对象返回，后续通过这个对象调用函数获取数据
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


# test
'''
camels_root = Path("F:/DataSet/basin_set/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2")
forcing_type = "daymet"
basin_list = Path("E:/lab/paper/RR-Former/RR-Former/data/3basins_list.txt")
basin_NO = pd.read_csv(basin_list, header=None, dtype=str)
for i in range(len(basin_NO)):
    basin = basin_NO[0][i]
    reader = HydroReaderFactory.hydro_reader(camels_root, forcing_type, basin)
    final_data = reader._load_data()
    # print(final_data)
    df_x = reader.get_df_x()
    df_y = reader.get_df_y()
    df_x = df_x[0:10]
    df_y = df_y[0:10]
    x = df_x.values.astype('float16')
    y = df_y.values.astype('float16')
    # print(x)
    # print(y)
    # s = x.std(axis=0)
    # item()函数将结果转为一个标量
    sy = y.std(axis = 0).item()
    # print(s)
    ''
    [2.176 25.2    2.95   4.3   50.12]
    [0.615 21.12   3.55   3.059 60.]
    [0.8584 16.11    2.89    3.004  47.88]
    '
    print(sy)
    # print(df_x)
    # print(df_x.values.astype('float16'))
'''


class CamelsDataset(Dataset):
    '''
    通过列表的方式进行工作，将basin_list中所有的盆地进行训练、验证和测试

    Attributes：
        camels_root:str
            数据集camels的路径
        basin_list:list of str
            包含所有用到的集水区的8位代码
        past_len:int
            过去时间步长discharge data的长度
        pred_len:int
            预测时间步长discharge data的长度
            气象数据的长度位为（past_len + pred_len）
        stage:str
            选择范围[train,val,test]
            决定是否计算均值和标准差
            在训练阶段计算均值和标准差
        dates:list of pd.DateTimes
            表示使用的日期范围，包括两个元素，开始日期和结束日期
        x_dict:dict as {basin:np.ndarray}
            将basin映射到相应的气象数据
        y_dict:dict as {basin:np.array}
            将basin映射到相应的discharge数据
        length_ls:list of int
            包含与basin_list对应每个盆地的序列化序列数
        index_ls:list of int
            通过length_ls经过__item__方法构造
        x_mean:np.ndarray
        y_mean:np.ndarray
        x_std:np.ndarray
        y_std:np.ndarray
    '''

    def __init__(self, camels_root: str, forcing_type: str, basin_list: list, past_len: int, pred_len: int,
                 stage: str, dates: list,
                 x_mean: np.ndarray, y_mean: np.ndarray, x_std: np.ndarray, y_std: np.ndarray, y_stds_dict=None):
        # 如果stage != 'train',x_mean,x_std,y_mean,y_std应该被提供
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

        # 初始化过程中加载数据
        self._load_data(forcing_type)

        # 计算均值mean和方差std
        if self.stage == 'train':
            # 为train时，需要计算mean和std
            self.x_mean, self.x_std = self.calc_mean_std(self.x_dict)
            self.y_mean, self.y_std = self.calc_mean_std(self.y_dict)
        else:
            # 不为train时，直接提供mean和std
            self.x_mean = x_mean
            self.x_std = x_std
            self.y_mean = y_mean
            self.y_std = y_std

        # 归一化数据
        self.normalize_data()

        # 设置样本的数量
        self.num_samples = 0
        for item in self.length_ls:
            self.num_samples += item

        self.index_ls = [0]
        for i in range(len(self.length_ls)):
            v = self.index_ls[i] + self.length_ls[i]
            self.index_ls.append(v)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        '''
        # 在数据集放入到DataLoader中后，通过该方法获得每个样本的数据
        # 这里的self.index_ls中存储从第一个站点到当前站点的数据总量
        # 通过bisect_right函数确定当前的idx在哪一个站点中
        # 接下来的local_idx用来确定当前的idx在确定的一个站点中的位置，从而确定下一个样本的开始位置
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basin_list[basin_idx]
        # x_seq:(512,22,5)
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]
        '''

        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basin_list[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]

    @staticmethod
    def calc_mean_std(data_dict: dict):
        # 计算均值和方差函数
        # 对所有的数据进行拼接
        data_all = np.concatenate(list(data_dict.values()), axis=0)
        nan_mean = np.nanmean(data_all, axis=0)
        nan_std = np.nanstd(data_all, axis=0)
        return nan_mean, nan_std

    # 归一化函数，用在归一化操作中，直接调用
    def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
        # variable用来判断输入还是输出['input','output']
        # 对输入数据和输出数据都可以进行归一化操作
        if variable == 'input':
            feature = (feature - self.x_mean) / self.x_std
        elif variable == 'output':
            feature = (feature - self.y_mean) / self.y_std
        else:
            raise RuntimeError(f"No such variable:{variable}")
        return feature

    # 对数据进行归一化，通过调用_local_normalization()函数
    def normalize_data(self):
        # 遍历所有集水区
        for idx, basin in enumerate(self.basin_list):
            # 打印当前归一化的进度
            print(self.stage, f"Normalizing %4f" % (idx / len(self.basin_list)))
            # 获得之前保存在字典中对应的集水区的数据
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            x_norm = self._local_normalization(x, 'input')
            y_norm = self._local_normalization(y, 'output')
            # 用归一化后的数据替换掉之前的数据
            self.x_dict[basin] = x_norm
            self.y_dict[basin] = y_norm

    # 重新缩放数据，逆归一化操作
    # 用于计算的数据重新缩放为原本尺度下的数据
    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'input':
            feature = (feature * self.x_std) + self.x_mean
        elif variable == 'output':
            feature = (feature * self.y_std) + self.y_mean
        else:
            raise RuntimeError(f"No such variable:{variable}")
        return feature

    # 用于获取均值
    def get_means(self):
        return self.x_mean, self.y_mean

    # 用于获取方法
    def get_stds(self):
        return self.x_std, self.y_std

    # 加载forcing_data
    def _load_data(self, forcing_type):
        # 集水区数量等于basin_list中集水区代码的数量
        basin_number = len(self.basin_list)
        # 从basin_list中依次读取集水区的代码
        for idx, basin in enumerate(self.basin_list):
            print(self.stage, f"{basin} Loading data %.4f" % (idx / basin_number))
            # 调用HydroReaderFactory类，传入当前集水区的代码，获取相对应的数据
            # reader:DaymetHydroReader实例化出的对象
            reader = HydroReaderFactory.hydro_reader(self.camels_root, forcing_type, basin)
            df_x = reader.get_df_x()
            df_y = reader.get_df_y()

            # 选择相应日期的数据
            # dates:表示使用的日期范围，开始日期和结束日期，pd.DateTimes
            df_x = df_x[self.dates[0]:self.dates[1]]
            df_y = df_y[self.dates[0]:self.dates[1]]
            # 两者数据长度不等时执行断言机制
            assert len(df_x) == len(df_y)
            # 将用到的df_x索引存入到对应的字典中，键值为basin
            self.date_index_dict[basin] = df_x.index

            # 选择使用到的features和discharge
            x = df_x.values.astype('float32')
            y = df_y.values.astype('float32')
            # 将获取到的features和discharge值存入对应字典中，键值为basin
            self.x_dict[basin] = x
            self.y_dict[basin] = y

            # length_ls:包含basin_list中每个basin的序列化数据的序列数
            # 相当于一个basin中包含的数据可以构成的样本数量
            self.length_ls.append(len(x) - self.past_len - self.pred_len + 1)
            # 计算均值方差
            # item()将得到的结果转为一个标量
            if self.stage == 'train':
                self.y_stds_dict[basin] = y.std(axis=0).item()

    @classmethod
    def get_instance(cls, past_len: int, pred_len: int, stage: str, specific_cfg: dict, x_mean=None, y_mean=None,
                     x_std=None, y_std=None, y_stds_dic=None):
        # 最后数据的存储路径
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
            # if final_data_path.exist():
            if final_data_path.exists():
                instance = torch.load(final_data_path)
                return instance
            else:
                dates = [start_date, end_date]
                instance = cls(camels_root, forcing_type, basin_list, past_len, pred_len, stage, dates, x_mean, y_mean,
                               x_std, y_std, y_stds_dic)
                # 在父目录上创建一个新文件夹    ????????
                final_data_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(instance, final_data_path)
                return instance


# 静态径流数据读取
class StaticReader(AbstractStaticReader):
    '''
    选择特定basin的静态属性，并进行归一化处理
    '''

    # 构造方法进行初始化
    def __init__(self, camels_root):
        self.camels_root = camels_root
        # 设置路径
        self.static_file_path = Path('')
        self.df_static = pd.read_csv(self.static_file_path, header=0, dtype={"gauge_id": str}).set_index("gauge_id")
        self.df_static = self.df_static.astype(dtype='float16')

    # 实现抽象类中的抽象方法
    def get_df_static(self, basin):
        return self.df_static.loc[[basin]].values


class CamelsDatasetWithStatic(CamelsDataset):
    # 将静态属性加入到数据序列中
    def __init__(self, camels_root: str, forcing_type: str, basins_list: list, past_len: int, pred_len: int, stage: str,
                 dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        # 获取一个读取静态数据的对象
        self.static_reader = StaticReader(camels_root)
        self.norm_static_fea = dict()
        super().__init__(camels_root, forcing_type, basins_list, past_len, pred_len, stage, dates, x_mean, y_mean,
                         x_std, y_std, y_stds_dict)

    # 重写_load_data方法
    def _load_data(self, forcing_type):
        # 集水区数量等于basin_list中集水区代码的数量
        basin_number = len(self.basin_list)
        # 从basin_list中依次读取集水区的代码
        for idx, basin in enumerate(self.basin_list):
            print(self.stage, f"{basin} Loading data %.4f" % (idx / basin_number))
            # 调用HydroReaderFactory类，传入当前集水区的代码，获取相对应的数据
            # reader:DaymetHydroReader实例化出的对象
            reader = HydroReaderFactory.hydro_reader(self.camels_root, forcing_type, basin)
            df_x = reader.get_df_x()
            df_y = reader.get_df_y()

            # 选择相应日期的数据
            # dates:表示使用的日期范围，开始日期和结束日期，pd.DateTimes
            df_x = df_x[self.dates[0]:self.dates[1]]
            df_y = df_y[self.dates[0]:self.dates[1]]
            # 两者数据长度不等时执行断言机制
            assert len(df_x) == len(df_y)
            # 将用到的df_x索引存入到对应的字典中，键值为basin
            self.date_index_dict[basin] = df_x.index

            # 选择使用到的features和discharge
            x = df_x.values.astype('float32')
            y = df_y.values.astype('float32')
            # 将获取到的features和discharge值存入对应字典中，键值为basin
            self.x_dict[basin] = x
            self.y_dict[basin] = y

            # length_ls:包含basin_list中每个basin的序列化数据的序列数
            # 每个basin中能够生成的样本数量
            self.length_ls.append(len(x) - self.past_len - self.pred_len + 1)
            # 添加静态属性
            self.norm_static_fea[basin] = self.static_reader.get_df_static(basin)
            # 计算均值方差
            # item()将得到的结果转为一个标量
            if self.stage == 'train':
                self.y_stds_dict[basin] = y.std(axis=0).item()

    def normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, "Normalizing %.4f" % (idx / len(self.basins_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            # Normalize data
            x_norm = self._local_normalization(x, variable='inputs')
            y_norm = self._local_normalization(y, variable='output')
            norm_static_fea = self.norm_static_fea[basin].repeat(x_norm.shape[0], axis=0)
            x_norm_static = np.concatenate([x_norm, norm_static_fea], axis=1)
            self.x_dict[basin] = x_norm_static
            self.y_dict[basin] = y_norm

    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]


class CamelsDatasetLimited(CamelsDataset):
    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]
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
            # print("假假")
        elif use_static:
            ds = CamelsDatasetWithStatic
            # print("真真")
        else:
            ds = CamelsDataset
            # print('真假')
        return ds

# data = DatasetFactory.get_dataset_type(True, False)
# ds_train = data.get_instance(15, 7, "train", specific_cfg=PretrainConfig.pre_train_config)
