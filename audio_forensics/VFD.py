#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def analyze(file_path, prefix, results_dir):
    y, sr = librosa.load(file_path, sr=None)  # keep original sr
    duration = librosa.get_duration(y=y, sr=sr)

    # Waveform
    plt.figure(figsize=(12,3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform - {prefix} ({Path(file_path).name})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    out_wave = results_dir / f'{prefix}_waveform.png'
    plt.tight_layout()
    plt.savefig(out_wave)
    plt.close()

    # STFT Spectrogram
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(12,4))
    librosa.display.specshow(D_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram (STFT) - {prefix}')
    out_spec = results_dir / f'{prefix}_spectrogram.png'
    plt.tight_layout()
    plt.savefig(out_spec)
    plt.close()

    # Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram - {prefix}')
    out_mel = results_dir / f'{prefix}_mel_spectrogram.png'
    plt.tight_layout()
    plt.savefig(out_mel)
    plt.close()

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(12,4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC (13) - {prefix}')
    out_mfcc = results_dir / f'{prefix}_mfcc.png'
    plt.tight_layout()
    plt.savefig(out_mfcc)
    plt.close()

    # Features (numeric)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    feats = {
        'file': Path(file_path).name,
        'prefix': prefix,
        'sr': sr,
        'duration_s': duration,
        'centroid_mean': float(np.mean(spectral_centroid)),
        'centroid_std': float(np.std(spectral_centroid)),
        'bandwidth_mean': float(np.mean(spectral_bandwidth)),
        'bandwidth_std': float(np.std(spectral_bandwidth)),
        'zcr_mean': float(np.mean(zcr)),
        'zcr_std': float(np.std(zcr)),
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
    }

    return feats

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 analyze_two.py real.wav fake.wav")
        sys.exit(1)

    real_path = Path(sys.argv[1])
    fake_path = Path(sys.argv[2])

    if not real_path.exists() or not fake_path.exists():
        print("One or both audio files do not exist.")
        sys.exit(1)

    results_dir = Path('results')
    ensure_dir(results_dir)

    print(f"Analyzing {real_path.name} (real) ...")
    feats_real = analyze(str(real_path), 'real', results_dir)
    print(f"Analyzing {fake_path.name} (fake) ...")
    feats_fake = analyze(str(fake_path), 'fake', results_dir)

    # Save features CSV
    df = pd.DataFrame([feats_real, feats_fake])
    csv_path = results_dir / 'features.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved images and features into: {results_dir.resolve()}")

if __name__ == '__main__':
    main()
