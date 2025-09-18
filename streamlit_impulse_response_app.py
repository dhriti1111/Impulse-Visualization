import streamlit as st
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Signal Generation Functions ---
def generate_impulse(length=50, position=0):
    impulse = np.zeros(length)
    impulse[position] = 1
    return impulse

def generate_sine(length=50, freq=1, amp=1):
    t = np.arange(length)
    sine = amp * np.sin(2 * np.pi * freq * t / length)
    return sine

def generate_square(length=50, freq=1, amp=1):
    t = np.arange(length)
    square = amp * signal.square(2 * np.pi * freq * t / length)
    return square

def generate_noise(length=50, amp=1):
    noise = amp * np.random.randn(length)
    return noise

# --- Convolution ---
def compute_impulse_response(impulse, signal):
    response = np.convolve(signal, impulse, mode='full')
    return response

# --- Analysis Functions ---
def scaling(signal, factor):
    return signal * factor

def sifting(signal, shift):
    # This function is not directly used for plotting the shifted response anymore,
    # as the shifting is handled by the x_offset in plot_signal for visual representation.
    # However, if you needed a new array with zero-padding for other operations,
    # this function would still be relevant.
    if shift == 0:
        return signal.copy()
    elif shift > 0:
        # Right shift: pad left with zeros
        return np.concatenate((np.zeros(shift), signal[:-shift]))
    else:
        # Left shift: pad right with zeros
        return np.concatenate((signal[-shift:], np.zeros(-shift)))

def addition(signal1, signal2):
    return signal1 + signal2

def multiplication(signal1, signal2):
    return signal1 * signal2

# --- Time Scaling ---
def time_scaling(signal, scale):
    # scale > 1: compress (faster), scale < 1: stretch (slower)
    from scipy.signal import resample
    new_length = max(1, int(len(signal) / scale)) # Ensure length is at least 1
    scaled_signal = resample(signal, new_length)
    return scaled_signal

# --- Plotting Helper ---
def plot_signal(signal_data, title, label, x_offset=0):
    fig, ax = plt.subplots(facecolor='none')
    
    N = len(signal_data)
    # The x values should start from x_offset and go up to x_offset + N - 1
    x = np.arange(N) + x_offset 
    
    ax.plot(x, signal_data, label=label, linewidth=2.5)
    
    # Adjust x-limits to show the entire relevant range including negative if x_offset is negative
    min_x_val = np.min(x)
    max_x_val = np.max(x)
    
    # Ensure a little padding on both sides
    # Calculate padding based on the extent of the signal
    x_extent = max_x_val - min_x_val
    x_range_padding = x_extent * 0.1 if x_extent > 0 else 1 # 10% padding, minimum 1 if extent is 0
    
    # Set x-limits to encompass the signal plus padding
    ax.set_xlim(min_x_val - x_range_padding, max_x_val + x_range_padding)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Ensure integer ticks for x-axis
    
    ax.set_title(title, color='white')
    ax.set_xlabel('Sample', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.legend()
    ax.grid(True, color='#FFFFFF', alpha=0.04, linewidth=0.8)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Adjust y-axis limits dynamically
    ymin = np.min(signal_data)
    ymax = np.max(signal_data)

    if label.lower() == "impulse":
        # Specific range for impulse to clearly show its peak at 1
        ax.set_ylim(-1.1, 1.1) 
    elif np.isclose(ymin, ymax): # Handle flat signals
        if ymin == 0:
            ax.set_ylim(-0.1, 0.1)
        else:
            ax.set_ylim(ymin * 0.9, ymax * 1.1)
    else:
        y_range_padding = (ymax - ymin) * 0.1 # 10% padding
        ax.set_ylim(ymin - y_range_padding, ymax + y_range_padding)

    st.pyplot(fig, transparent=True)
    return fig

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Impulsify", layout="wide")
    st.markdown(
        """
        <style>
        html, body, .stApp {
            background: #232526 !important;
            color: #eaffff !important;
        }
        .glow-title {
            font-size: 3em;
            font-weight: bold;
            color: #00eaff;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="glow-title">Impulsify</div>', unsafe_allow_html=True)
    st.write("Analyze the impulse response and properties of signals using convolution.")

    # Sidebar Controls
    st.sidebar.header("Signal Controls")
    signal_type = st.sidebar.selectbox("Choose signal type", ["Sine", "Ramp", "Square", "Sawtooth", "Impulse", "Random Noise"])
    length = st.sidebar.slider("Signal Length", min_value=10, max_value=200, value=50)
    freq = st.sidebar.slider("Frequency", min_value=1, max_value=20, value=1)
    amp = st.sidebar.slider("Amplitude", min_value=0.1, max_value=5.0, value=1.0)
    impulse_pos = st.sidebar.slider("Impulse Position", min_value=0, max_value=length-1, value=0)

    # Generate signals
    impulse = generate_impulse(length, impulse_pos)
    if signal_type == "Sine":
        sig = generate_sine(length, freq, amp)
    elif signal_type == "Square":
        sig = generate_square(length, freq, amp)
    elif signal_type == "Ramp":
        sig = np.linspace(0, amp, length)
    elif signal_type == "Sawtooth":
        t = np.arange(length)
        sig = amp * signal.sawtooth(2 * np.pi * freq * t / length)
    elif signal_type == "Impulse":
        sig = generate_impulse(length, impulse_pos)
    else:
        sig = generate_noise(length, amp)

    st.subheader("Original Signals")
    col1, col2 = st.columns(2)
    with col1:
        plot_signal(impulse, "Impulse Signal", "Impulse")
    with col2:
        plot_signal(sig, f"{signal_type} Signal", signal_type)

    # Tabs for Analysis
    tab_conv, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Convolution", "Amplitude Scaling", "Time Scaling", "Sifting", "Addition", "Multiplication"])
    # --- Convolution ---
    with tab_conv:
        st.header("Convolution Property")
        st.markdown(r"""
        **Formula:**
        $$y[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]$$
        
        **Theory:**
        Convolution is a mathematical operation that combines two signals to produce a third signal. In discrete time, it represents how the shape of one signal (the impulse response) modifies another signal. For LTI systems, convolution with the impulse signal gives the system's output for any input.
        """)
        response = compute_impulse_response(impulse, sig)
        st.write("Convolution of impulse and selected signal:")
        plot_signal(response, "Impulse Response (Convolution)", "Response")

    # --- Scaling ---
    with tab1:
        st.header("Amplitude Scaling Property")
        st.markdown(r"""
        **Formula:**
        $$y[n] = a \cdot x[n]$$
        
        **Theory:**
        Amplitude scaling multiplies every sample of the signal by a constant factor $a$. This increases or decreases the signal's amplitude but does not affect its shape or timing.
        """)
        scale_factor = st.slider("Scaling Factor", min_value=0.1, max_value=5.0, value=1.0)
        base_response = compute_impulse_response(impulse, sig)
        scaled_response = scaling(base_response, scale_factor)
        st.write(f"Impulse response scaled by {scale_factor}")
        fig, ax = plt.subplots(facecolor='none')
        x = np.arange(len(base_response))
        ax.plot(x, base_response, label="Original Impulse Response", color='blue', linewidth=2.5)
        ax.plot(x, scaled_response, label=f"Scaled Response (x{scale_factor})", color='red', linewidth=2.5)
        ax.set_title("Amplitude Scaling", color='white')
        ax.set_xlabel('Sample', color='white')
        ax.set_ylabel('Amplitude', color='white')
        ax.legend()
        ax.grid(True, color='#FFFFFF', alpha=0.04, linewidth=0.8)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig, transparent=True)
        # Add watermark before saving
        ax.text(0.5, 0.5, 'Impulsify', fontsize=36, color='gray', alpha=0.3,
            ha='center', va='center', transform=ax.transAxes, zorder=10)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        st.download_button("Save Plot as PNG", data=buf.getvalue(), file_name="scaling_plot.png", mime="image/png")

    # --- Time Scaling ---
    with tab2:
        st.header("Time Scaling Property")
        st.markdown(r"""
        **Formula:**
        $$y[n] = x[an]$$

        **Theory:**
        Time scaling compresses or stretches a signal in time. If $a > 1$, the signal is compressed (faster); if $a < 1$, the signal is stretched (slower). In discrete signals, this is achieved by resampling.
        """)
        time_scale = st.slider("Time Scaling Factor", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
        base_response = compute_impulse_response(impulse, sig)
        ts_response = time_scaling(base_response, time_scale)
        st.write(f"Time scaling factor: {time_scale} ( <1: stretch, >1: compress )")
        fig, ax = plt.subplots(facecolor='none')
        x_in = np.arange(len(base_response))
        x_out = np.arange(len(ts_response))
        ax.plot(x_in, base_response, label="Original Impulse Response", color='blue', linewidth=2.5)
        ax.plot(x_out, ts_response, label=f"Time Scaled Response (x{time_scale})", color='red', linewidth=2.5)
        ax.set_title("Time Scaling", color='white')
        ax.set_xlabel('Sample', color='white')
        ax.set_ylabel('Amplitude', color='white')
        ax.legend()
        ax.grid(True, color='#FFFFFF', alpha=0.04, linewidth=0.8)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ymin = np.min(ts_response)
        ymax = np.max(ts_response)
        ypad = (ymax - ymin) * 0.1 if ymax > ymin else 1
        ax.set_ylim(ymin - ypad, ymax + ypad)
        st.pyplot(fig, transparent=True)
        # Add watermark before saving
        ax.text(0.5, 0.5, 'Impulsify', fontsize=36, color='gray', alpha=0.3,
            ha='center', va='center', transform=ax.transAxes, zorder=10)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        st.download_button("Save Plot as PNG", data=buf.getvalue(), file_name="time_scaling_plot.png", mime="image/png")

    # --- Sifting ---
    with tab3:
        st.header("Sifting Property (Shifting)")
        st.markdown(r"""
        **Formula:**
        $$y[n] = x[n-n_0]$$

        **Theory:**
        Sifting (shifting) moves the signal left or right by $n_0$ samples. Positive $n_0$ shifts right (delays), negative $n_0$ shifts left (advances). The signal's shape and amplitude remain unchanged.
        """)
        shift_amt = st.slider("Shift Amount (samples)", min_value=-length//2, max_value=length//2, value=0)
        base_response = compute_impulse_response(impulse, sig)
        sifted_response = sifting(base_response, shift_amt)
        st.write(f"Impulse response shifted by {shift_amt} samples")
        fig, ax = plt.subplots(facecolor='none')
        x = np.arange(len(base_response))
        x_shifted = x + shift_amt
        ax.plot(x, base_response, label="Original Impulse Response", color='blue', linewidth=2.5)
        ax.plot(x_shifted, sifted_response, label=f"Shifted Response ({shift_amt})", color='red', linewidth=2.5)
        ax.set_title("Sifting (Shifting)", color='white')
        ax.set_xlabel('Sample', color='white')
        ax.set_ylabel('Amplitude', color='white')
        ax.legend()
        ax.grid(True, color='#FFFFFF', alpha=0.04, linewidth=0.8)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        # Ensure negative x-axis is visible
        min_x = min(np.min(x), np.min(x_shifted))
        max_x = max(np.max(x), np.max(x_shifted))
        xpad = (max_x - min_x) * 0.1 if max_x > min_x else 1
        ax.set_xlim(min_x - xpad, max_x + xpad)
        st.pyplot(fig, transparent=True)
        # Add watermark before saving
        ax.text(0.5, 0.5, 'Impulsify', fontsize=36, color='gray', alpha=0.3,
            ha='center', va='center', transform=ax.transAxes, zorder=10)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        st.download_button("Save Plot as PNG", data=buf.getvalue(), file_name="sifting_plot.png", mime="image/png")

    # --- Addition ---
    with tab4:
        st.header("Addition Property")
        st.markdown(r"""
        **Formula:**
        $$y[n] = x_1[n] + x_2[n]$$
        
        **Theory:**
        Addition combines two signals pointwise. The resulting signal at each sample is the sum of the corresponding samples from both signals. Useful for superposition and mixing.
        """)
        add_type = st.selectbox("Second signal type for addition", ["Sine", "Square", "Ramp", "Sawtooth", "Random Noise"])
        add_freq = st.slider("Second Signal Frequency", min_value=1, max_value=20, value=3)
        add_amp = st.slider("Second Signal Amplitude", min_value=0.1, max_value=5.0, value=1.0)
        if add_type == "Sine":
            sig2 = generate_sine(length, add_freq, add_amp)
        elif add_type == "Square":
            sig2 = generate_square(length, add_freq, add_amp)
        elif add_type == "Ramp":
            sig2 = np.linspace(0, add_amp, length)
        elif add_type == "Sawtooth":
            t = np.arange(length)
            sig2 = add_amp * signal.sawtooth(2 * np.pi * add_freq * t / length)
        else:
            sig2 = generate_noise(length, add_amp)
        
        base_response = compute_impulse_response(impulse, sig)
        sig2_padded = np.pad(sig2, (0, max(0, len(base_response)-len(sig2))), 'constant')[:len(base_response)]
        added_sig = addition(base_response, sig2_padded)
        st.write(f"Added impulse response and second signal ({add_type})")
        fig, ax = plt.subplots(facecolor='none')
        x = np.arange(len(base_response))
        ax.plot(x, base_response, label="Original Impulse Response", color='blue', linewidth=2.5)
        ax.plot(x, added_sig, label=f"Added Signal ({add_type})", color='red', linewidth=2.5)
        ax.set_title("Addition", color='white')
        ax.set_xlabel('Sample', color='white')
        ax.set_ylabel('Amplitude', color='white')
        ax.legend()
        ax.grid(True, color='#FFFFFF', alpha=0.04, linewidth=0.8)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig, transparent=True)
        # Add watermark before saving
        ax.text(0.5, 0.5, 'Impulsify', fontsize=36, color='gray', alpha=0.3,
            ha='center', va='center', transform=ax.transAxes, zorder=10)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        st.download_button("Save Plot as PNG", data=buf.getvalue(), file_name="addition_plot.png", mime="image/png")

    # --- Multiplication ---
    with tab5:
        st.header("Multiplication Property")
        st.markdown(r"""
        **Formula:**
        $$y[n] = x_1[n] \cdot x_2[n]$$
        
        **Theory:**
        Multiplication combines two signals pointwise by multiplying their samples. This is used for modulation and windowing operations in signal processing.
        """)
        mult_type = st.selectbox("Second signal type for multiplication", ["Sine", "Square", "Ramp", "Sawtooth", "Random Noise"], key="mult")
        mult_freq = st.slider("Second Signal Frequency for Mult", min_value=1, max_value=20, value=2, key="mult_freq")
        mult_amp = st.slider("Second Signal Amplitude for Mult", min_value=0.1, max_value=5.0, value=1.0, key="mult_amp")
        if mult_type == "Sine":
            sig2 = generate_sine(length, mult_freq, mult_amp)
        elif mult_type == "Square":
            sig2 = generate_square(length, mult_freq, mult_amp)
        elif mult_type == "Ramp":
            sig2 = np.linspace(0, mult_amp, length)
        elif mult_type == "Sawtooth":
            t = np.arange(length)
            sig2 = mult_amp * signal.sawtooth(2 * np.pi * mult_freq * t / length)
        else:
            sig2 = generate_noise(length, mult_amp)
        base_response = compute_impulse_response(impulse, sig)
        sig2_padded = np.pad(sig2, (0, max(0, len(base_response)-len(sig2))), 'constant')[:len(base_response)]
        multiplied_sig = multiplication(base_response, sig2_padded)
        st.write(f"Multiplied impulse response and second signal ({mult_type})")
        fig, ax = plt.subplots(facecolor='none')
        x = np.arange(len(base_response))
        ax.plot(x, base_response, label="Original Impulse Response", color='blue', linewidth=2.5)
        ax.plot(x, multiplied_sig, label=f"Multiplied Signal ({mult_type})", color='red', linewidth=2.5)
        ax.set_title("Multiplication", color='white')
        ax.set_xlabel('Sample', color='white')
        ax.set_ylabel('Amplitude', color='white')
        ax.legend()
        ax.grid(True, color='#FFFFFF', alpha=0.04, linewidth=0.8)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        st.pyplot(fig, transparent=True)
        # Add watermark before saving
        ax.text(0.5, 0.5, 'Impulsify', fontsize=36, color='gray', alpha=0.3,
            ha='center', va='center', transform=ax.transAxes, zorder=10)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        st.download_button("Save Plot as PNG", data=buf.getvalue(), file_name="multiplication_plot.png", mime="image/png")

if __name__ == "__main__":
    main()