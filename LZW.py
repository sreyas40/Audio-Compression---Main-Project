import wave
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

# LZW Compression
def lzw_compress(uncompressed):
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(dict_size)}
    w = b""
    compressed = []
    for c in uncompressed:
        wc = w + bytes([c])
        if wc in dictionary:
            w = wc
        else:
            compressed.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = bytes([c])
    if w:
        compressed.append(dictionary[w])
    return compressed

# LZW Decompression
def lzw_decompress(compressed):
    dict_size = 256
    dictionary = {i: bytes([i]) for i in range(dict_size)}
    w = bytes([compressed.pop(0)])
    decompressed = bytearray(w)
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0:1]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        decompressed.extend(entry)
        dictionary[dict_size] = w + entry[0:1]
        dict_size += 1
        w = entry
    return decompressed

# Function to measure MSE, CR, Encoding and Decoding Speeds
def measure_metrics(original_data, decompressed_data, original_size, compressed_size, encoding_time, decoding_time):
    # Calculate MSE
    mse = np.mean((np.frombuffer(original_data, dtype=np.int16) - np.frombuffer(decompressed_data, dtype=np.int16))**2)
    
    # Compression Ratio
    compression_ratio = original_size / compressed_size
    
    return mse, compression_ratio, encoding_time, decoding_time

# Read the WAV file
input_file = 'input2.wav'
with wave.open(input_file, 'rb') as wav_file:
    n_channels = wav_file.getnchannels()
    sampwidth = wav_file.getsampwidth()
    framerate = wav_file.getframerate()
    n_frames = wav_file.getnframes()
    audio_data = wav_file.readframes(n_frames)

# Print original file size
original_size_mb = os.path.getsize(input_file) / (1024 * 1024)
print(f"Original file size: {original_size_mb:.2f} MB")

# Compress the audio data using LZW
start_time = time.time()
compressed_data = lzw_compress(audio_data)
encoding_time = time.time() - start_time

# Save compressed data to a binary file
compressed_file = 'compressed.bin'
with open(compressed_file, 'wb') as f:
    for code in compressed_data:
        max_bits = (code.bit_length() + 7) // 8  # Number of bytes required to store the code
        f.write(code.to_bytes(max_bits, byteorder=sys.byteorder))  # Adjust bytes to fit the code

# Print compressed file size
compressed_size_mb = os.path.getsize(compressed_file) / (1024 * 1024)
print(f"Compressed file size: {compressed_size_mb:.2f} MB")

# Decompress the audio data
start_time = time.time()
decompressed_data = lzw_decompress(compressed_data)
decoding_time = time.time() - start_time

# Save the decompressed data back to a WAV file
output_file = 'output2.wav'
with wave.open(output_file, 'wb') as wav_file:
    wav_file.setnchannels(n_channels)
    wav_file.setsampwidth(sampwidth)
    wav_file.setframerate(framerate)
    wav_file.writeframes(decompressed_data)

# Print decompressed file size
decompressed_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"Decompressed file size: {decompressed_size_mb:.2f} MB")

# Measure MSE, Compression Ratio, Encoding Speed, and Decoding Speed
original_size = len(audio_data) * 8  # in bits
compressed_size = len(compressed_data) * 12  # assuming 12-bit codes
mse, cr, enc_speed, dec_speed = measure_metrics(audio_data, decompressed_data, original_size, compressed_size, encoding_time, decoding_time)

# Print MSE, Compression Ratio, Encoding Speed, and Decoding Speed
print(f"MSE: {mse:.4f}")
print(f"Compression Ratio: {cr:.2f}")
print(f"Encoding Speed: {enc_speed:.4f} seconds")
print(f"Decoding Speed: {dec_speed:.4f} seconds")

# Plot the original and decompressed waveforms
original_waveform = np.frombuffer(audio_data, dtype=np.int16)
decompressed_waveform = np.frombuffer(decompressed_data, dtype=np.int16)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(original_waveform, label='Original')
plt.title('Original Audio Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(1, 2, 2)
plt.plot(decompressed_waveform, label='Decompressed', color='orange')
plt.title('Decompressed Audio Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
