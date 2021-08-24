import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pyESN_fft2 import ESN
from sklearn import preprocessing
from matplotlib.offsetbox import AnchoredText
from scipy.fftpack import fft2, ifft2
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

data = np.load('./data.npy')
data_future = np.load('./data.npy')

data = np.concatenate((data, data_future), axis=1)

np.random.seed(97)
randomtime = np.random.randint(1000, size=10)


for j in np.arange(0.2, 0.95, 0.9):
    for k in np.arange(0.2, 0.95, 0.9):

        reservoir_dimension = 7000
        spectral_radius = j
        random_state = 53
        leaky_rate = 0.95
        noise = 0.0
        noise_data = 0.0
        timescale = 1
        sparsity = k
        lamb = 0.0001
        start_train = 100000
        T_train = 150000

        psi_a = data[0, :, :]
        deltaT_a = data[1, :, :]

        psi_a = psi_a[:, :, :]
        deltaT_a = deltaT_a[:, :, :]

        psi_a_F = fft2(psi_a)
        psi_a_F = psi_a_F.reshape((psi_a.shape[0], -1))

        deltaT_a_F = fft2(deltaT_a)
        deltaT_a_F = deltaT_a_F.reshape((deltaT_a.shape[0], -1))

        psi_a_F_real = psi_a_F.real
        psi_a_F_imag = psi_a_F.imag

        deltaT_a_F_real = deltaT_a_F.real
        deltaT_a_F_imag = deltaT_a_F.imag

        data_atm = np.concatenate((psi_a_F_real, psi_a_F_imag, deltaT_a_F_real, deltaT_a_F_imag), axis=1)
        data_ori = data_atm

        scaler_atm = preprocessing.MinMaxScaler().fit(data_atm)
        data_atm_scaled = scaler_atm.transform(data_atm)
        data_scaled = data_atm_scaled



        data_dim = data_scaled.shape[1]

        inputs_dim = data_scaled.shape[1]
        esn = ESN(n_inputs = inputs_dim,
              n_outputs = data_dim,
              n_reservoir = reservoir_dimension,
              spectral_radius = spectral_radius,
              random_state = random_state, leak = leaky_rate, noise=noise, sparsity=sparsity, lamb = lamb )



        dt = 1 # dt represents how many step difference between inputs and predictions.
        u = data_scaled[:(T_train-start_train-1), :]

        pred_training_scaled, loss, W_out, s_last, y_last = esn.train(u, data_scaled[1:(T_train-start_train), :])

        prediction_length = 200
        prediction_scaled = esn.predict(s_last, y_last, prediction_length)

        prediction = scaler_atm.inverse_transform(prediction_scaled)   # prediction inverse scaling

        pred_training = scaler_atm.inverse_transform(pred_training_scaled)

        # inverse Fourier prediction
        num = psi_a.shape[1]*psi_a.shape[1]
        prediction_psi_a_F_real = prediction[:, :num]
        prediction_psi_a_F_imag = prediction[:, num:2*num]

        prediction_psi_a_F = prediction_psi_a_F_real+prediction_psi_a_F_imag*1j
        prediction_psi_a_F = prediction_psi_a_F.reshape((prediction.shape[0], psi_a.shape[1], psi_a.shape[1]))

        prediction_psi_a = ifft2(prediction_psi_a_F)
        prediction_psi_a = prediction_psi_a.real

        prediction_deltaT_a_F_real = prediction[:, 2*num:3*num]
        prediction_deltaT_a_F_imag = prediction[:, 3*num:]

        prediction_deltaT_a_F = prediction_deltaT_a_F_real+prediction_deltaT_a_F_imag*1j
        prediction_deltaT_a_F = prediction_deltaT_a_F.reshape((prediction.shape[0], psi_a.shape[1], psi_a.shape[1]))

        prediction_deltaT_a = ifft2(prediction_deltaT_a_F)
        prediction_deltaT_a = prediction_deltaT_a.real

