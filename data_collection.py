import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
from scipy.signal import butter, lfilter, lfilter_zi
import csv
from drawnow import *

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype='bandstop')


def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n
    
def compute_band_powers(eegdata, fs):
    """Extract the features (band powers) from the EEG.
    Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata
    Returns:
        (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    """
    # 1. Compute the PSD
    winSampleLength, nbCh = eegdata.shape

    # Apply Hamming window
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T * w).T

    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

    # SPECTRAL FEATURES
    # Average of band powers
    # Delta <4
    ind_delta, = np.where(f < 4)
    meanDelta = np.mean(PSD[ind_delta, :], axis=0)
    # Theta 4-8
    ind_theta, = np.where((f >= 4) & (f <= 8))
    meanTheta = np.mean(PSD[ind_theta, :], axis=0)
    # Alpha 8-12
    ind_alpha, = np.where((f >= 8) & (f <= 12))
    meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
    # Beta 12-30
    ind_beta, = np.where((f >= 12) & (f < 30))
    meanBeta = np.mean(PSD[ind_beta, :], axis=0)

    feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha,
                                     meanBeta), axis=0)

    feature_vector = np.log10(feature_vector)

    return feature_vector

def update_buffer(data_buffer, new_data, notch=False, filter_state=None):
    """
    Concatenates "new_data" into "data_buffer", and returns an array with
    the same size as "data_buffer"
    """
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),
                                   (data_buffer.shape[1], 1)).T
        new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,
                                         zi=filter_state)

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer, filter_state


def get_last_data(data_buffer, newest_samples):
    """
    Obtains from "buffer_array" the "newest samples" (N rows from the
    bottom of the buffer)
    """
    new_buffer = data_buffer[(data_buffer.shape[0] - newest_samples):, :]

    return new_buffer


def update_plot():
    
    axis[0].plot(d)
    axis[0].set_title("Delta")

    axis[1].plot(t)
    axis[1].set_title("Theta")

    axis[2].plot(a)
    axis[2].set_title("Alpha")

    axis[3].plot(b)
    axis[3].set_title("Beta")


# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

# Search for active LSL streams
print('Looking for an EEG stream...')
streams = resolve_byprop('type', 'EEG', timeout=2)
if len(streams) == 0:
    raise RuntimeError('Can\'t find EEG stream.')

# Set active EEG stream to inlet and apply time correction
print("Start acquiring data")
inlet = StreamInlet(streams[0], max_chunklen=12)
eeg_time_correction = inlet.time_correction()

# Get the stream info and description
info = inlet.info()
description = info.desc()
fs = int(info.nominal_srate())

""" 2. INITIALIZE BUFFERS """
# Initialize raw EEG data buffer
eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
filter_state = None  # for use with the notch filter
# Compute the number of epochs in "buffer_length"
n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                          SHIFT_LENGTH + 1))
# Initialize the band power buffer (for plotting)
# bands will be ordered: [delta, theta, alpha, beta]
band_buffer = np.zeros((n_win_test, 4))
""" 3. GET DATA """
# The try/except structure allows to quit the while loop by aborting the
# script with <Ctrl-C>
print('Press Ctrl-C in the console to break the while loop.')

figure, axis = plt.subplots(4)
plt.ion()
axis[0].set_xlim(0, 100)
axis[1].set_xlim(0, 100)
axis[2].set_xlim(0, 100)
axis[3].set_xlim(0, 100)

axis[0].set_ylim(-10, 10)
axis[1].set_ylim(-10, 10)
axis[2].set_ylim(-10, 10)
axis[3].set_ylim(-10, 10)
plt.show()
d = []
t = []
a = []
b = []

header = ['Delta', 'Theta', 'Alpha', 'Beta']
count = 0

with open('eeg_data.csv', mode='w') as eeg_file:
    eeg_write = csv.writer(eeg_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    eeg_write.writerow(header)
    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * fs))
            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
            # Update EEG buffer with the new data
            eeg_buffer, filter_state = update_buffer(
                eeg_buffer, ch_data, notch=True,
                    filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = get_last_data(eeg_buffer,
                                             EPOCH_LENGTH * fs)
            # Compute band powers
            band_powers = compute_band_powers(data_epoch, fs)
            band_buffer, _ = update_buffer(band_buffer,
                                                 np.asarray([band_powers]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers = np.mean(band_buffer, axis=0)
            print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
                  ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

            d.append(band_powers[Band.Delta])
            t.append(band_powers[Band.Theta])
            a.append(band_powers[Band.Alpha])
            b.append(band_powers[Band.Beta])

            update_plot()
            
            plt.pause(.0001)

            if count > 100:
                d.pop(0)
                t.pop(0)
                a.pop(0)
                b.pop(0)
            count += 1

            data = [band_powers[Band.Delta], band_powers[Band.Theta], band_powers[Band.Alpha], band_powers[Band.Beta]]
            eeg_write.writerow(data)

    except KeyboardInterrupt:
        print('Closing!')
