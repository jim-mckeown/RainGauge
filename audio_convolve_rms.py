import numpy as np
import scipy.signal as signal
import soundfile as sf
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

def load_kernel_from_csv(csv_file_path):
    """Load convolution kernel from CSV file"""
    try:
        if csv_file_path.endswith('.csv'):
            try:
                df = pd.read_csv(csv_file_path, header=None)
            except:
                try:
                    df = pd.read_csv(csv_file_path, header=None, delimiter=';')
                except:
                    df = pd.read_csv(csv_file_path, header=None, delimiter='\t')
            
            kernel_values = df.values.flatten()
        else:
            raise ValueError("File must be a CSV file")
        
        kernel = np.array(kernel_values, dtype=np.float64)
        
        if np.any(kernel < -1.0) or np.any(kernel > 1.0):
            print("Warning: Kernel values outside [-1, +1] range detected. Clipping values.")
            kernel = np.clip(kernel, -1.0, 1.0)
        
        if np.sum(np.abs(kernel)) > 0:
            kernel = kernel / np.sum(np.abs(kernel))
        
        print(f"Loaded kernel from {csv_file_path}")
        print(f"Kernel size: {len(kernel)} samples")
        print(f"Kernel range: [{np.min(kernel):.3f}, {np.max(kernel):.3f}]")
        
        return kernel
        
    except Exception as e:
        raise ValueError(f"Error loading kernel from CSV: {e}")

def create_kernel(size=64, kernel_type='gaussian', custom_csv=None):
    """Create a convolution kernel"""
    if custom_csv:
        return load_kernel_from_csv(custom_csv)
    
    if kernel_type == 'gaussian':
        kernel = signal.windows.gaussian(size, std=size/4)
    elif kernel_type == 'rectangular':
        kernel = np.ones(size)
    elif kernel_type == 'bandpass':
        kernel = signal.firwin(size, [1000, 8000], fs=32000, pass_zero=False)
    else:
        kernel = np.sin(2 * np.pi * np.arange(size) / size)
    
    kernel = kernel / np.sum(np.abs(kernel))
    return kernel

def calculate_rms(signal_data, window_size=64):
    """
    Calculate RMS values with a sliding window
    
    Args:
        signal_data: Input signal array
        window_size: Size of RMS calculation window
    """
    rms_values = np.zeros_like(signal_data)
    half_window = window_size // 2
    
    for i in range(len(signal_data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(signal_data), i + half_window + 1)
        
        window = signal_data[start_idx:end_idx]
        rms_values[i] = np.sqrt(np.mean(window**2))
    
    return rms_values

def convolve_audio_with_rms(input_file, output_file, kernel_size=64, frame_size=1024, 
                           kernel_type='gaussian', kernel_csv=None, rms_window=64):
    """
    Convolve audio and create RMS channel with proper per-sample calculation
    """
    
    # Read input audio file
    print(f"Reading audio file: {input_file}")
    audio, sample_rate = sf.read(input_file)
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        print("Converted stereo to mono")
    
    print(f"Audio shape: {audio.shape}, Sample rate: {sample_rate} Hz")
    
    # Create kernel
    kernel = create_kernel(kernel_size, kernel_type, kernel_csv)
    print(f"Using kernel of size {len(kernel)}")
    
    # Convolve entire audio signal (no frame-based processing for convolution)
    print("Convolving entire audio signal...")
    convolved_audio = signal.convolve(audio, kernel, mode='same', method='auto')
    
    # Calculate RMS with sliding window on the CONVOLVED signal
    print("Calculating RMS values with sliding window...")
    rms_values = calculate_rms(convolved_audio, rms_window)
    
    # Normalize the convolved audio to prevent clipping
    convolved_audio = normalize_audio(convolved_audio)
    
    # Create stereo output: left channel = convolved audio, right channel = RMS values
    rms_audio = scale_rms_to_audio(rms_values, convolved_audio)
    stereo_output = np.column_stack((convolved_audio, rms_audio))
    
    # Write output file
    sf.write(output_file, stereo_output, sample_rate)
    print(f"Output saved to: {output_file}")
    
    # Generate plots for visualization
    generate_plots(audio, convolved_audio, rms_values, kernel, sample_rate)
    
    return convolved_audio, rms_values, kernel

def normalize_audio(audio, target_max=0.9):
    """Normalize audio to prevent clipping"""
    current_max = np.max(np.abs(audio))
    if current_max > 0:
        return audio * (target_max / current_max)
    return audio

def scale_rms_to_audio(rms_values, reference_audio):
    """Scale RMS values to audio range for listening"""
    ref_max = np.max(np.abs(reference_audio))
    rms_max = np.max(rms_values) if np.max(rms_values) > 0 else 1
    scaled_rms = rms_values * (ref_max / rms_max) * 0.3
    return scaled_rms

def generate_plots(original_audio, convolved_audio, rms_values, kernel, sample_rate):
    """Generate visualization plots"""
    # Create shorter time segments for clearer plots
    plot_samples = min(10000, len(original_audio))
    time = np.arange(plot_samples) / sample_rate
    kernel_time = np.arange(len(kernel)) / sample_rate * 1000
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Kernel
    plt.subplot(4, 1, 1)
    plt.plot(kernel_time, kernel, 'b-', linewidth=2)
    plt.title('Convolution Kernel')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot 2: Original vs Convolved audio (first 10k samples)
    plt.subplot(4, 1, 2)
    plt.plot(time, original_audio[:plot_samples], 'b-', alpha=0.7, label='Original')
    plt.plot(time, convolved_audio[:plot_samples], 'r-', alpha=0.7, label='Convolved')
    plt.title('Original vs Convolved Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: RMS values (first 10k samples)
    plt.subplot(4, 1, 3)
    plt.plot(time, rms_values[:plot_samples], 'g-', linewidth=1)
    plt.title('RMS Values (Sliding Window)')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Value')
    plt.grid(True)
    
    # Plot 4: Zoomed-in RMS to show smoothness
    zoom_samples = min(2000, plot_samples)
    zoom_time = np.arange(zoom_samples) / sample_rate
    plt.subplot(4, 1, 4)
    plt.plot(zoom_time, rms_values[:zoom_samples], 'g-', linewidth=1.5)
    plt.title('RMS Values (Zoomed - Should be Smooth)')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('audio_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Analysis plots saved to 'audio_analysis.png'")

def main():
    parser = argparse.ArgumentParser(description='Convolve audio and create smooth RMS channel')
    parser.add_argument('input_file', help='Input WAV file path')
    parser.add_argument('output_file', help='Output WAV file path')
    parser.add_argument('--kernel_size', type=int, default=64, help='Size of convolution kernel')
    parser.add_argument('--kernel_type', default='gaussian', 
                       choices=['gaussian', 'rectangular', 'bandpass', 'custom'],
                       help='Type of convolution kernel')
    parser.add_argument('--kernel_csv', help='Path to custom CSV kernel file')
    parser.add_argument('--rms_window', type=int, default=64, 
                       help='Window size for RMS calculation (samples)')
    
    args = parser.parse_args()
    
    try:
        convolved_audio, rms_values, kernel = convolve_audio_with_rms(
            args.input_file, 
            args.output_file, 
            args.kernel_size, 
            1024,  # frame_size parameter kept for compatibility but not used
            args.kernel_type,
            args.kernel_csv,
            args.rms_window
        )
        
        print("\nProcessing completed successfully!")
        print(f"Convolved audio range: [{np.min(convolved_audio):.3f}, {np.max(convolved_audio):.3f}]")
        print(f"RMS values range: [{np.min(rms_values):.3f}, {np.max(rms_values):.3f}]")
        print("RMS should now be smooth without stepped appearance")
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        raise

if __name__ == "__main__":
    main()