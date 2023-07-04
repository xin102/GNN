import torch
from pathlib import Path,PosixPath


class ProjectConfig:
    project_root = Path(__file__).absolute().parent.parent
    single_gpu = 0
    device = torch.device(f"cuda:{single_gpu}" if torch.cuda.is_available() else 'cpu')

    torch.cuda.set_device(device)
    num_workers = 1  # number of threads to load data
    run_root = Path("./runs")  # save running information
    final_data_root = PosixPath("./final_data")