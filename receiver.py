import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter, find_peaks
from scipy.fft import fft, fftfreq

import matplotlib.pyplot as plt

from scipy.signal import find_peaks

# Define the DTMF frequencies
DTMF_FREQS = {
    '1': (697, 1209),
    '2': (697, 1336),
    '3': (697, 1477),
    '4': (770, 1209),
    '5': (770, 1336),
    '6': (770, 1477),
    '7': (852, 1209),
    '8': (852, 1336),
    '9': (852, 1477),
    '*': (941, 1209),
    '0': (941, 1336),
    '#': (941, 1477),
}

def filtro(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def find_fft_peaks(signal, threshold=None):
    """Find the peaks in the Fourier transform of a signal."""
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(signal.size, 1/sample_rate)
    if threshold is not None:
        threshold = threshold * np.max(np.abs(fft))
        peaks, _ = find_peaks(np.abs(fft)[:int(fft.size/2)], height=threshold, distance=200)
    else:
        peaks, _ = find_peaks(np.abs(fft)[:int(fft.size/2)], distance=200)
    return freqs[peaks], np.abs(fft[peaks])

def identify_dtmf_digit(frequency_pair):
    for tone in DTMF_FREQS.items():
      number, pair = tone
      if frequency_pair == pair:
          return number

def find_dtmf_range(frequencies):
    """Find the closest DTMF frequencies to the given frequencies."""
    dtmf_frequencies = []
    for freq in frequencies:
        closest = None
        # Find the closest DTMF frequency to the current frequency
        for pair in DTMF_FREQS.values():
            for dtmf_freq in pair:
                if closest is None or abs(freq - dtmf_freq) < abs(freq - closest):
                    closest = dtmf_freq
        if closest is not None:
            dtmf_frequencies.append(closest)
    if not dtmf_frequencies:
        return None, None
    return min(dtmf_frequencies), max(dtmf_frequencies)

# Record audio from the microphone
duration = 1  # seconds
sample_rate = 44100  # samples per second
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=2)
sd.wait()

sd.play(recording, sample_rate)
sd.wait()

# Apply a bandpass filter to isolate the DTMF tones
b, a = filtro(600, 1600, sample_rate)
filtered_recording = lfilter(b, a, recording[:, 0])

n = sample_rate * duration
yf = fft(filtered_recording)
xf = fftfreq(n, 1 / sample_rate)

# peaks = peakutils.indexes(yf, thres=0.2, min_dist=200)
# peaks_x = peakutils.interpolate(xf, yf, ind=peaks)

peaks, amplitudes = find_fft_peaks(filtered_recording, threshold=0.1)
frequency_pair = find_dtmf_range(peaks)
number = identify_dtmf_digit(frequency_pair)


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("DTMF: " + number)

t = np.linspace(0, duration, int(duration * sample_rate))
ax1.plot(t, filtered_recording)

ax2.plot(xf, yf)
plt.plot(peaks, amplitudes, 'ro')
plt.show()

# Print the decoded digits
# print("Decoded digits:", "".join(digits))