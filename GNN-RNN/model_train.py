import torch
from build_graph import data_graph


def train_epoch(model, data_loader, optimizer, scheduler, loss_func, device,i):
    model.train()
    cnt = 0
    loss_mean = 0
    threshold = None
    g = data_graph(threshold)
    # g = g.to(device)
    for x_seq, y_seq_past, y_seq_future in data_loader:
        optimizer.zero_grad()
        x_seq, y_seq_past, y_seq_future = x_seq.to(device), y_seq_past.to(device), y_seq_future.to(device)

        output = model(x_seq,y_seq_past,g)
        # y_data = torch.permute(y_data,(2,0,1))
        y_data = y_seq_future.squeeze(dim=3)
        y_data = y_data.permute((1,0,2))
        loss = loss_func(output,y_data)
        if cnt % 5 == 0:
            print(f"    epoch:{i} loop:{cnt} ----loss:{loss}----")

        loss.backward()
        optimizer.step()
        cnt += 1
        loss_mean = loss_mean + (loss.item() - loss_mean) / cnt # Welford's method
    scheduler.step()
    # return loss_mean,loss_num
    return loss_mean
