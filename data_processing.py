import numpy as np
import os

data_path = '...'

time = np.load(data_path)

freq = np.fft.fft(time)[:,:,:time.shape[-1] // 2]

a = freq.real
b = freq.imag

magnitude = np.abs(freq)

phase = np.zeros_like(a)
phase[a > 0] = np.arctan(b / a)[a > 0]
phase[a < 0] = (np.arctan(b / a) + np.sign(b) * np.pi)[a < 0]
phase[a == 0] = (np.sign(b) * np.pi / 2)[a == 0]

data = np.concatenate([time, magnitude, phase], -1)

np.save(data, os.path.join(data_path, 'time_freq_data.npy'))
