import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Load the .wav file
wav_file = 'input2.wav'
data_wav, samplerate_wav = sf.read(wav_file)

# Measure encoding speed
start_time = time.time()
flac_file = 'output2.flac'
sf.write(flac_file, data_wav, samplerate_wav, format='FLAC')
encoding_time = time.time() - start_time

# Measure decoding speed
start_time = time.time()
data_flac, samplerate_flac = sf.read(flac_file)
decoding_time = time.time() - start_time

# Get file sizes in bytes
wav_size = os.path.getsize(wav_file)
flac_size = os.path.getsize(flac_file)

# Convert file sizes to MB
wav_size_mb = wav_size / (1024 * 1024)
flac_size_mb = flac_size / (1024 * 1024)

# Calculate storage savings and compression ratio
storage_savings_mb = wav_size_mb - flac_size_mb
compression_ratio = wav_size_mb / flac_size_mb

# Calculate MSE, SNR, PSNR
mse = np.mean((data_wav - data_flac) ** 2)
max_value = np.max(data_wav)
snr = 10 * np.log10(np.mean(data_wav ** 2) / mse)
psnr = 10 * np.log10(max_value ** 2 / mse)

# Calculate BER (Bit Error Rate)
wav_bits = np.unpackbits(data_wav.astype(np.uint8), axis=0)
flac_bits = np.unpackbits(data_flac.astype(np.uint8), axis=0)
ber = np.sum(wav_bits != flac_bits) / len(wav_bits)

# Print the results
print(f"Original WAV File Size: {wav_size_mb:.2f} MB")
print(f"Converted FLAC File Size: {flac_size_mb:.2f} MB")
print(f"Storage Saved: {storage_savings_mb:.2f} MB")
print(f"Compression Ratio (CR): {compression_ratio:.2f}")
print(f"Mean Squared Error (MSE): {mse:.5f}")
print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
print(f"Bit Error Rate (BER): {ber:.5f}")
print(f"Encoding Speed: {encoding_time:.2f} seconds")
print(f"Decoding Speed: {decoding_time:.2f} seconds")

# Function to plot waveform
def plot_waveform(data, samplerate, title, subplot_position):
    times = np.arange(len(data)) / float(samplerate)
    
    plt.subplot(2, 1, subplot_position)
    plt.plot(times, data)
    plt.title(title)
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')

# Plot waveforms for both .wav and .flac files
plt.figure(figsize=(15, 10))

plot_waveform(data_wav, samplerate_wav, 'Waveform of WAV File', 1)
plot_waveform(data_flac, samplerate_flac, 'Waveform of FLAC File', 2)

plt.tight_layout()
plt.show()
