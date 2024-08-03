import numpy as np
import torch


def load_data(fname):
    data = np.load(fname)
    data_list = []
    for file in data.files:
        data_list.append(data[file])

    np_data = np.asarray(data_list)
    np_data = np.transpose(np_data, (1, 2, 3, 0))

    return np_data


data_sr = np.load("dataset/era5_tisr.npz")
tisr = data_sr["tisr"][:-1]
tisr = (tisr - np.mean(tisr))/np.std(tisr)


data_oro = np.load("dataset/regridded_era_orography.npy")
oro_mean = np.mean(data_oro)
oro_std = np.std(data_oro)
data_oro= np.tile(data_oro, (16537, 1, 1))
oro = (data_oro - oro_mean) / oro_std


data_tp = load_data("dataset/era5_tp_6hr.npz")
tp = np.log(data_tp[:16538,:,:,0]/1e-2 + 1)
first_tp = torch.tensor(tp[0].reshape(1,48,96,1)).permute(0,3,1,2).to(device)
tp_delta = tp[1:] - tp[:-1]
tp = tp[1:]
tp_mean = np.mean(tp)
tp_std = np.std(tp)
tp_min = np.min(tp)
tp_max = np.max(tp)
tp = (tp - tp_mean)/tp_std

tp_delta_mean = np.mean(tp_delta)
tp_delta_std = np.std(tp_delta)
tp_delta_min = np.min(tp_delta)
tp_delta_max = np.max(tp_delta)
tp_delta = (tp_delta)/tp_delta_std


data_p = load_data("dataset/era5_numpy_file_train_test_logp.npz")
sp = np.exp(data_p[:16538,:,:,0]) * 1000.0 * 100.0
sp_delta = sp[1:] - sp[:-1]
sp = sp[:-1]
sp_mean = np.mean(sp)
sp_std = np.std(sp)
sp_min = np.min(sp)
sp_max = np.max(sp)
sp = (sp - sp_mean)/sp_std

sp_delta_mean = np.mean(sp_delta)
sp_delta_std = np.std(sp_delta)
sp_delta_min = np.min(sp_delta)
sp_delta_max = np.max(sp_delta)
sp_delta = (sp_delta)/sp_delta_std


data = load_data('dataset/era5_numpy_file.npz')
temp = data[:,:,:,0]
temp_delta = temp[1:] - temp[:-1]
temp = temp[:-1]
temp_mean = np.mean(temp)
temp_std = np.std(temp)
temp_min = np.min(temp)
temp_max = np.max(temp)
temp = (temp - temp_mean)/temp_std

temp_delta_mean = np.mean(temp_delta)
temp_delta_std = np.std(temp_delta)
temp_delta_min = np.min(temp_delta)
temp_delta_max = np.max(temp_delta)
temp_delta = (temp_delta)/temp_delta_std


humid = data[:,:,:,1]
humid_delta = humid[1:] - humid[:-1]
humid = humid[:-1]
humid_mean = np.mean(humid)
humid_std = np.std(humid)
humid_min = np.min(humid)
humid_max = np.max(humid)
humid = (humid - humid_mean)/humid_std

humid_delta_mean = np.mean(humid_delta)
humid_delta_std = np.std(humid_delta)
humid_delta_min = np.min(humid_delta)
humid_delta_max = np.max(humid_delta)
humid_delta = (humid_delta)/humid_delta_std


u_wind = data[:,:,:,2]
u_wind_delta = u_wind[1:] - u_wind[:-1]
u_wind = u_wind[:-1]
u_wind_mean = np.mean(u_wind)
u_wind_std = np.std(u_wind)
u_wind_min = np.min(u_wind)
u_wind_max = np.max(u_wind)
u_wind = (u_wind - u_wind_mean)/u_wind_std

u_wind_delta_mean = np.mean(u_wind_delta)
u_wind_delta_std = np.std(u_wind_delta)
u_wind_delta_min = np.min(u_wind_delta)
u_wind_delta_max = np.max(u_wind_delta)
u_wind_delta = (u_wind_delta)/u_wind_delta_std


v_wind = data[:,:,:,3]
v_wind_delta = v_wind[1:] - v_wind[:-1]
v_wind = v_wind[:-1]
v_wind_mean = np.mean(v_wind)
v_wind_std = np.std(v_wind)
v_wind_min = np.min(v_wind)
v_wind_max = np.max(v_wind)
v_wind = (v_wind - v_wind_mean)/v_wind_std

v_wind_delta_mean = np.mean(v_wind_delta)
v_wind_delta_std = np.std(v_wind_delta)
v_wind_delta_min = np.min(v_wind_delta)
v_wind_delta_max = np.max(v_wind_delta)
v_wind_delta = (v_wind_delta)/v_wind_delta_std


input_dataset = np.stack((temp, humid, u_wind, v_wind, sp, tisr, oro), axis=-1)
target_dataset = np.stack((temp_delta, humid_delta, u_wind_delta, v_wind_delta, sp_delta, tp), axis=-1)

input_means = torch.tensor(np.stack((temp_mean, humid_mean, u_wind_mean, v_wind_mean, sp_mean), axis=-1).reshape(1,1,1,5)).permute(0,3,1,2).to(device)
input_stds = torch.tensor(np.stack((temp_std, humid_std, u_wind_std, v_wind_std, sp_std), axis=-1).reshape(1,1,1,5)).permute(0,3,1,2).to(device)
input_mins = torch.tensor(np.stack((temp_min, humid_min, u_wind_min, v_wind_min, sp_min), axis=-1).reshape(1,1,1,5)).permute(0,3,1,2).to(device)
input_maxs = torch.tensor(np.stack((temp_max, humid_max, u_wind_max, v_wind_max, sp_max), axis=-1).reshape(1,1,1,5)).permute(0,3,1,2).to(device)

target_means = torch.tensor(np.stack((temp_delta_mean, humid_delta_mean, u_wind_delta_mean, v_wind_delta_mean, sp_delta_mean, tp_mean), axis=-1).reshape(1,1,1,6)).permute(0,3,1,2).to(device)
target_stds = torch.tensor(np.stack((temp_delta_std, humid_delta_std, u_wind_delta_std, v_wind_delta_std, sp_delta_std, tp_std), axis=-1).reshape(1,1,1,6)).permute(0,3,1,2).to(device)
target_mins = torch.tensor(np.stack((temp_delta_min, humid_delta_min, u_wind_delta_min, v_wind_delta_min, sp_delta_min, tp_min), axis=-1).reshape(1,1,1,6)).permute(0,3,1,2).to(device)
target_maxs = torch.tensor(np.stack((temp_delta_max, humid_delta_max, u_wind_delta_max, v_wind_delta_max, sp_delta_max, tp_max), axis=-1).reshape(1,1,1,6)).permute(0,3,1,2).to(device)

input_dataset = torch.tensor(input_dataset)
target_dataset = torch.tensor(target_dataset)
input_dataset = input_dataset.permute(0,3,1,2)
target_dataset = target_dataset.permute(0,3,1,2)

data_dict = {
    'input_dataset': input_dataset,
    'target_dataset': target_dataset,
    'input_means': input_means,
    'input_stds': input_stds,
    'input_mins': input_mins,
    'input_maxs': input_maxs,
    'target_means': target_means,
    'target_stds': target_stds,
    'target_mins': target_mins,
    'target_maxs': target_maxs
}

torch.save(data_dict, 'LUCIE_training_data.pth')
