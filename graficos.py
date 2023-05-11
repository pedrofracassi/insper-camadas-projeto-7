import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Lado emissor

# Gráfico 1: Sinal de áudio original normalizado - domínio do tempo

# Importar o arquivo .wav original
filename = 'caminho/do/arquivo_original.wav'
data, samplerate = sf.read(filename)

# Extrair o vetor de amplitudes
yAudio = data[:, 0]  # Se for um arquivo estéreo, selecione o canal desejado

# Normalizar o sinal original
yAudioNormalizado = yAudio / max(abs(yAudio))

# Plotar o sinal de áudio original no domínio do tempo
tempo = np.arange(len(yAudioNormalizado)) / samplerate

plt.plot(tempo, yAudioNormalizado)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal de Áudio Original Normalizado - Domínio do Tempo')
plt.show()


# Gráfico 2: Sinal de áudio filtrado - domínio do tempo

# Configurar o filtro passa baixa
nyq_rate = samplerate / 2
width = 5.0 / nyq_rate
ripple_db = 60.0  # dB
N, beta = signal.kaiserord(ripple_db, width)
cutoff_hz = 4000.0
taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

# Aplicar o filtro passa baixa no sinal original normalizado
yFiltrado = signal.lfilter(taps, 1.0, yAudioNormalizado)

# Plotar o sinal de áudio filtrado no domínio do tempo
plt.plot(tempo, yFiltrado)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal de Áudio Filtrado - Domínio do Tempo')
plt.show()


# Gráfico 3: Sinal de áudio filtrado - domínio da frequência (Fourier)

# Calcular a Transformada Rápida de Fourier (FFT) do sinal filtrado
fft_filtrado = np.fft.fft(yFiltrado)
freq_filtrado = np.fft.fftfreq(len(yFiltrado), 1/samplerate)
amplitude_filtrado = np.abs(fft_filtrado)

# Plotar o espectro de frequência do sinal filtrado
plt.plot(freq_filtrado, amplitude_filtrado)
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro de Frequência - Sinal Filtrado')
plt.show()


# Gráfico 4: Sinal de áudio modulado - domínio do tempo

# Modulação em Amplitude (AM)
portadora_freq = 14000  # Frequência da portadora em Hz
portadora = np.sin(2 * np.pi * portadora_freq * np.arange(len(yFiltrado)) / samplerate)
sinal_modulado = yFiltrado * portadora

# Plotar o sinal modulado no domínio do tempo
plt.plot(tempo, sinal_modulado)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal de Áudio Modulado - Domínio do Tempo')
plt.show()


# Gráfico 5: Sinal de áudio modulado - domínio da frequência

# Calcular a Transformada Rápida de Fourier (FFT) do sinal modulado
fft_modulado = np.fft.fft(sinal_modulado)
freq_modulado = np.fft.fftfreq(len(sinal_modulado), 1/samplerate)
amplitude_modulado = np.abs(fft_modulado)

# Plotar o espectro de frequência do sinal modulado
plt.plot(freq_modulado, amplitude_modulado)
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro de Frequência - Sinal Modulado')
plt.show()


# Lado receptor

# Gráfico 6: Sinal de áudio gravado modulado – domínio do tempo

# Importar o arquivo de áudio gravado modulado
filename_gravado = 'caminho/do/sinal_modulado.wav'
data_gravado, samplerate_gravado = sf.read(filename_gravado)

# Extrair o vetor de amplitudes do sinal gravado
sinal_gravado = data_gravado[:, 0]  # Se for um arquivo estéreo, selecione o canal desejado

# Plotar o sinal de áudio gravado modulado no domínio do tempo
tempo_gravado = np.arange(len(sinal_gravado)) / samplerate_gravado

plt.plot(tempo_gravado, sinal_gravado)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal de Áudio Gravado Modulado - Domínio do Tempo')
plt.show()


# Gráfico 7: Sinal de áudio gravado modulado - domínio da frequência

# Calcular a Transformada Rápida de Fourier (FFT) do sinal gravado modulado
fft_gravado = np.fft.fft(sinal_gravado)
freq_gravado = np.fft.fftfreq(len(sinal_gravado), 1/samplerate_gravado)
amplitude_gravado = np.abs(fft_gravado)

# Plotar o espectro de frequência do sinal gravado modulado
plt.plot(freq_gravado, amplitude_gravado)
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro de Frequência - Sinal Gravado Modulado')
plt.show()


# Gráfico 8: Sinal de áudio demodulado - domínio do tempo

# Demodular o sinal gravado utilizando a mesma frequência da portadora utilizada na modulação (14000 Hz)
sinal_demodulado = sinal_gravado * portadora

# Plotar o sinal de áudio demodulado no domínio do tempo
plt.plot(tempo_gravado, sinal_demodulado)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal de Áudio Demodulado - Domínio do Tempo')
plt.show()


# Gráfico 9: Sinal de áudio demodulado - domínio da frequência

# Calcular a Transformada Rápida de Fourier (FFT) do sinal demodulado
fft_demodulado = np.fft.fft(sinal_demodulado)
freq_demodulado = np.fft.fftfreq(len(sinal_demodulado), 1/samplerate_gravado)
amplitude_demodulado = np.abs(fft_demodulado)

# Plotar o espectro de frequência do sinal demodulado
plt.plot(freq_demodulado, amplitude_demodulado)
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro de Frequência - Sinal Demodulado')
plt.show()


# Gráfico 10: Sinal de áudio demodulado filtrado - domínio do tempo

# Filtre as frequências superiores a 4000 Hz aplicando um filtro passa baixa
nyq_rate_gravado = samplerate_gravado / 2
width_gravado = 5.0 / nyq_rate_gravado
ripple_db_gravado = 60.0  # dB
N_gravado, beta_gravado = signal.kaiserord(ripple_db_gravado, width_gravado)
cutoff_hz_gravado = 4000.0
taps_gravado = signal.firwin(N_gravado, cutoff_hz_gravado / nyq_rate_gravado, window=('kaiser', beta_gravado))

# Aplicar o filtro passa baixa no sinal demodulado
sinal_demodulado_filtrado = signal.lfilter(taps_gravado, 1.0, sinal_demodulado)

# Calcular a Transformada Rápida de Fourier (FFT) do sinal demodulado filtrado
fft_demodulado_filtrado = np.fft.fft(sinal_demodulado_filtrado)
freq_demodulado_filtrado = np.fft.fftfreq(len(sinal_demodulado_filtrado), 1/samplerate_gravado)
amplitude_demodulado_filtrado = np.abs(fft_demodulado_filtrado)

# Plotar o espectro de frequência do sinal demodulado filtrado
plt.plot(freq_demodulado_filtrado, amplitude_demodulado_filtrado)
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro de Frequência - Sinal Demodulado e Filtrado')
plt.show()