# Gráfico da Transformada Rápida de Fourier (FFT) do sinal filtrado 
# Verificar se as frequências acima de 4000 Hz foram eliminadas
# Modular o sinal em amplitude com uma portadora de 14000 Hz e normalizá-lo

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
import sounddevice as sd

# Importar o arquivo .wav
filename = 'caminho/do/arquivo.wav'
data, samplerate = sf.read(filename)

# Extrair o vetor de amplitudes
yAudio = data[:, 0]  # Se for um arquivo estéreo, selecione o canal desejado

# Determinar o tempo desejado entre 2 e 5 segundos
tempo_desejado = 3  # Altere para o tempo desejado em segundos
num_amostras = tempo_desejado * samplerate

# Selecionar apenas as primeiras amostras dentro do tempo desejado
yAudio = yAudio[:num_amostras]

# Normalizar o vetor de amplitudes
yAudioNormalizado = yAudio / max(abs(yAudio))

# Configurar o filtro passa baixa
nyq_rate = samplerate / 2
width = 5.0 / nyq_rate
ripple_db = 60.0  # dB
N, beta = signal.kaiserord(ripple_db, width)
cutoff_hz = 4000.0
taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

# Aplicar o filtro passa baixa
yFiltrado = signal.lfilter(taps, 1.0, yAudio)

# Calcular a Transformada Rápida de Fourier (FFT) do sinal filtrado
fft = np.fft.fft(yFiltrado)
freq = np.fft.fftfreq(len(yFiltrado), 1/samplerate)
amplitude = np.abs(fft)

# Plotar o espectro de frequência
plt.plot(freq, amplitude)
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro de Frequência')
plt.show()

# Modulação em Amplitude (AM)
portadora_freq = 14000  # Frequência da portadora em Hz
portadora = np.sin(2 * np.pi * portadora_freq * np.arange(len(yFiltrado)) / samplerate)
sinal_modulado = yFiltrado * portadora

# Normalizar o sinal modulado
max_value = np.max(np.abs(sinal_modulado))
sinal_modulado_normalizado = sinal_modulado / max_value

# Reproduzir o áudio filtrado
sd.play(yFiltrado, samplerate)
sd.wait()