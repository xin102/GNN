import geopandas
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import dgl
import torch
import networkx
import numpy as np

plt.style.use("ggplot")
from distance import get_distance_hav


# matplotlib.use('agg')


def data_graph(threshold):
    # matplotlib.use('TkAgg')
    file_path = 'basin_set_full_res/HCDN_nhru_final_671.shp'
    # ignore_geometry = True:dtype->DataFrame
    shp_df = geopandas.read_file(file_path,ignore_geometry = True)
    # 'lon_cen':longitude
    # 'lat_cen':latitude
    # position = shp_df[['hru_id','lon_cen','lat_cen']]
    basins_file = 'file_path'
    basin_list = pd.read_csv(basins_file, header=None, dtype=str)[0].values.tolist()
    position = shp_df[shp_df['hru_id'].isin(basin_list)]
    position = position[['lon_cen','lat_cen']]
    node_num = len(basin_list)

    g = dgl.graph(([],[]),idtype=torch.int64)
    g.add_nodes(num=node_num)
    # networkx.draw_networkx(g.to_networkx(),with_labels=True)

    position = np.array(position)
    position = torch.tensor(position)
    # g.ndata['node'] = position

    # ans = 0
    # edge_weights = []
    for i in range(node_num):
        j = i + 1
        # calc distance of basins
        while j < node_num:
            lng1 = position[i][0]
            lat1 = position[i][1]
            lng2 = position[j][0]
            lat2 = position[j][1]
            dist = get_distance_hav(lng1,lat1,lng2,lat2)
            if dist <= threshold:
                g.add_edges(i,j)
            j += 1

    # add reverse edge
    g = dgl.add_reverse_edges(g)
    return g
