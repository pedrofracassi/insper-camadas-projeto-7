import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

DTMF = {
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

number = input("Digite um número: ")

duration = 4
sample_rate = 44100
t = np.linspace(0, duration, int(duration * sample_rate), False)
tone = np.zeros(len(t))

freq1, freq2 = DTMF[number]
freq1_sin = np.sin(2 * np.pi * freq1 * t)
freq2_sin = np.sin(2 * np.pi * freq2 * t)
tone += freq1_sin + freq2_sin

transformada = np.fft.fft(tone)

sd.play(tone, sample_rate)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("DTMF: " + number)


ax1.plot(t[0:200], freq1_sin[0:200], label=f"freq1 ({freq1}Hz)")
ax1.plot(t[0:200], freq2_sin[0:200], label=f"freq2 ({freq2}Hz)")
ax1.plot(t[0:200], tone[0:200], label="tom dtmf")
ax1.legend()

freqs = np.fft.fftfreq(tone.size, 1/sample_rate)
fft = np.abs(np.fft.fft(tone))
ax2.plot(freqs[:int(freqs.size/2)], fft[:int(freqs.size/2)])

plt.show()

sd.wait() # wait só depois do plt pra tocar o som enquanto mostra o gráfico