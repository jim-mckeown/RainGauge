import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd

def create_target_kernel(sample_rate=32000, main_freq=1100, kernel_size=1024, 
                        attack_time=0.003, decay_time=0.01):
    """
    Create kernel with smooth attack-decay transition
    """
    t = np.arange(kernel_size) / sample_rate
    
    # Calculate attack and decay components
    attack_component = t / attack_time  # Linear rise from 0 to 1 during attack
    decay_component = np.exp(-np.maximum(t - attack_time, 0) / decay_time)  # Exponential decay
    
    # Create smooth envelope using minimum function
    envelope = np.minimum(attack_component, decay_component)
    
    # Ensure envelope stays between 0 and 1
    envelope = np.clip(envelope, 0, 1)
    
    # Create carrier signal (sine wave at main frequency)
    carrier = np.sin(2 * np.pi * main_freq * t)
    
    # Apply envelope to carrier
    kernel = envelope * carrier
    
    # Normalize to maximum absolute value of 1.0
    max_val = np.max(np.abs(kernel))
    if max_val > 0:
        kernel = kernel / max_val
    
    return kernel, envelope, t, attack_time, decay_time  # Return attack_time and decay_time

def save_kernel_to_csv(kernel, filename):
    """Save kernel to CSV file"""
    df = pd.DataFrame({'kernel_value': kernel})
    df.to_csv(filename, index=False, header=False)
    print(f"Kernel saved to {filename}")

def plot_kernel_analysis(kernel, envelope, t, sample_rate, main_freq, attack_time, decay_time):
    """Plot detailed analysis of the kernel"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Envelope and kernel
    plt.subplot(3, 2, 1)
    plt.plot(t * 1000, envelope, 'b-', linewidth=2, label='Envelope')
    plt.plot(t * 1000, kernel, 'r-', linewidth=1, label='Kernel', alpha=0.7)
    plt.title('Kernel and Envelope')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Zoom on attack phase
    plt.subplot(3, 2, 2)
    attack_samples = int(attack_time * sample_rate) + 50
    plt.plot(t[:attack_samples] * 1000, envelope[:attack_samples], 'b-', label='Envelope')
    plt.plot(t[:attack_samples] * 1000, kernel[:attack_samples], 'r-', label='Kernel', alpha=0.7)
    plt.title(f'Attack Phase (0-{attack_time*1000:.1f}ms + buffer)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Frequency spectrum
    plt.subplot(3, 2, 3)
    frequencies = np.fft.fftfreq(len(kernel), 1/sample_rate)
    fft_values = np.abs(np.fft.fft(kernel))
    positive_freq_mask = frequencies > 0
    plt.semilogy(frequencies[positive_freq_mask], fft_values[positive_freq_mask])
    plt.axvline(main_freq, color='r', linestyle='--', alpha=0.7, label=f'{main_freq}Hz')
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (log)')
    plt.grid(True)
    plt.legend()
    
    # Plot 4: Phase information
    plt.subplot(3, 2, 4)
    phase = np.angle(np.fft.fft(kernel))
    plt.plot(frequencies[positive_freq_mask], phase[positive_freq_mask])
    plt.title('Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    
    # Plot 5: Kernel values (first 100 samples)
    plt.subplot(3, 2, 5)
    plt.plot(kernel[:100])
    plt.title('First 100 Samples of Kernel')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot 6: Statistical information
    plt.subplot(3, 2, 6)
    plt.text(0.1, 0.8, f'Kernel size: {len(kernel)} samples', transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'Duration: {t[-1]*1000:.2f} ms', transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'Attack time: {attack_time*1000:.1f} ms', transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'Decay time constant: {decay_time*1000:.1f} ms', transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'Main frequency: {main_freq} Hz', transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f'Sample rate: {sample_rate} Hz', transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('Kernel Parameters')
    
    plt.tight_layout()
    plt.savefig('kernel_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Parameters
    sample_rate = 32000
    main_freq = 1100
    kernel_size = 1024
    attack_time = 0.003  # 3ms attack
    decay_time = 0.01   # 10ms decay time constant
    
    print("Creating target kernel...")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Main frequency: {main_freq} Hz")
    print(f"Kernel size: {kernel_size} samples")
    print(f"Total duration: {kernel_size/sample_rate*1000:.2f} ms")
    print(f"Attack time: {attack_time*1000:.1f} ms")
    print(f"Decay time constant: {decay_time*1000:.1f} ms")
    
    # Create kernel
    kernel, envelope, t, attack_time, decay_time = create_target_kernel(
        sample_rate=sample_rate,
        main_freq=main_freq,
        kernel_size=kernel_size,
        attack_time=attack_time,
        decay_time=decay_time
    )
    
    # Verify kernel properties
    print(f"\nKernel properties:")
    print(f"Length: {len(kernel)} samples")
    print(f"Min value: {np.min(kernel):.6f}")
    print(f"Max value: {np.max(kernel):.6f}")
    print(f"Mean value: {np.mean(kernel):.6f}")
    print(f"Starts at: {kernel[0]:.6f} (should be near 0)")
    print(f"Peak around sample: {np.argmax(np.abs(kernel))}")
    
    # Save to CSV
    save_kernel_to_csv(kernel, 'target_kernel_1100hz.csv')
    
    # Generate analysis plots
    print("\nGenerating analysis plots...")
    plot_kernel_analysis(kernel, envelope, t, sample_rate, main_freq, attack_time, decay_time)
    
    # Test convolution with a sample signal
    print("\nTesting with sample signal...")
    test_signal = np.sin(2 * np.pi * main_freq * t)  # Pure tone at target frequency
    convolved = signal.convolve(test_signal, kernel, mode='same')
    
    plt.figure(figsize=(12, 4))
    plt.plot(t * 1000, test_signal, 'b-', alpha=0.7, label='Test Signal')
    plt.plot(t * 1000, convolved, 'r-', label='Convolved Result')
    plt.title('Convolution Test with Pure Tone')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.savefig('convolution_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Done! Kernel created and saved.")

if __name__ == "__main__":
    main()