import os
import librosa
import soundfile as sf

def convert_wav_to_flac(wav_path, flac_path):
    """Convert a single WAV file to FLAC format."""
    audio, sr = librosa.load(wav_path, sr=None)
    sf.write(flac_path, audio, sr, format='FLAC')

def batch_convert_wav_to_flac(input_folder, output_folder):
    """Convert all WAV files in the input_folder to FLAC format in the output_folder."""
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all .wav files in the input folder
    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    
    total_files = len(wav_files)
    if total_files == 0:
        print("No WAV files found in the input folder.")
        return

    for i, wav_file in enumerate(wav_files, 1):
        wav_path = os.path.join(input_folder, wav_file)
        flac_file = os.path.splitext(wav_file)[0] + '.flac'
        flac_path = os.path.join(output_folder, flac_file)
        
        convert_wav_to_flac(wav_path, flac_path)
        
        # Show progress in the terminal
        print(f"[{i}/{total_files}] Converted: {wav_file} -> {flac_file}")

    print(f"Conversion completed! All FLAC files are saved in '{output_folder}'.")

# Example usage
input_folder = 'wavs'  # Replace with the path to your input folder containing .wav files
output_folder = 'flacs'  # Replace with the path where you want to save the .flac files

batch_convert_wav_to_flac(input_folder, output_folder)
