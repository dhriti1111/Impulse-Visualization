import streamlit as st
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
    return np.roll(signal, shift)

def addition(signal1, signal2):
    return signal1 + signal2

def multiplication(signal1, signal2):
    return signal1 * signal2

# --- Plotting Helper ---
def plot_signal(signal, title, label):
    fig, ax = plt.subplots()
    ax.plot(signal, label=label)
    ax.set_title(title)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    return fig

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Impulse Response Analyzer", layout="wide")
    st.title("Impulse Response Analyzer")
    st.write("Analyze the impulse response and properties of signals using convolution.")

    # Sidebar Controls
    st.sidebar.header("Signal Controls")
    signal_type = st.sidebar.selectbox("Choose signal type", ["Sine", "Square", "Random Noise"])
    length = st.sidebar.slider("Signal Length", min_value=10, max_value=200, value=50)
    freq = st.sidebar.slider("Frequency", min_value=1, max_value=20, value=5)
    amp = st.sidebar.slider("Amplitude", min_value=0.1, max_value=5.0, value=1.0)
    impulse_pos = st.sidebar.slider("Impulse Position", min_value=0, max_value=length-1, value=0)

    # Generate signals
    impulse = generate_impulse(length, impulse_pos)
    if signal_type == "Sine":
        sig = generate_sine(length, freq, amp)
    elif signal_type == "Square":
        sig = generate_square(length, freq, amp)
    else:
        sig = generate_noise(length, amp)

    # Impulse Response
    response = compute_impulse_response(impulse, sig)

    st.subheader("Original Signals and Impulse Response")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_signal(impulse, "Impulse Signal", "Impulse")
    with col2:
        plot_signal(sig, f"{signal_type} Signal", signal_type)
    with col3:
        plot_signal(response, "Impulse Response (Convolution)", "Response")

    # Tabs for Analysis
    tab1, tab2, tab3, tab4 = st.tabs(["Scaling", "Sifting", "Addition", "Multiplication"])

    # --- Scaling ---
    with tab1:
        st.header("Scaling Property")
        scale_factor = st.slider("Scaling Factor", min_value=0.1, max_value=5.0, value=1.0)
        scaled_sig = scaling(sig, scale_factor)
        scaled_response = compute_impulse_response(impulse, scaled_sig)
        st.write(f"Signal scaled by {scale_factor}")
        plot_signal(scaled_sig, "Scaled Signal", f"Scaled {signal_type}")
        plot_signal(scaled_response, "Impulse Response of Scaled Signal", "Scaled Response")

    # --- Sifting ---
    with tab2:
        st.header("Sifting Property (Shifting)")
        shift_amt = st.slider("Shift Amount (samples)", min_value=-length//2, max_value=length//2, value=0)
        sifted_sig = sifting(sig, shift_amt)
        sifted_response = compute_impulse_response(impulse, sifted_sig)
        st.write(f"Signal shifted by {shift_amt} samples")
        plot_signal(sifted_sig, "Shifted Signal", f"Shifted {signal_type}")
        plot_signal(sifted_response, "Impulse Response of Shifted Signal", "Shifted Response")

    # --- Addition ---
    with tab3:
        st.header("Addition Property")
        add_type = st.selectbox("Second signal type for addition", ["Sine", "Square", "Random Noise"])
        add_freq = st.slider("Second Signal Frequency", min_value=1, max_value=20, value=3)
        add_amp = st.slider("Second Signal Amplitude", min_value=0.1, max_value=5.0, value=1.0)
        if add_type == "Sine":
            sig2 = generate_sine(length, add_freq, add_amp)
        elif add_type == "Square":
            sig2 = generate_square(length, add_freq, add_amp)
        else:
            sig2 = generate_noise(length, add_amp)
        added_sig = addition(sig, sig2)
        added_response = compute_impulse_response(impulse, added_sig)
        st.write(f"Added {signal_type} and {add_type} signals")
        plot_signal(sig2, f"Second Signal ({add_type})", add_type)
        plot_signal(added_sig, "Added Signal", "Sum")
        plot_signal(added_response, "Impulse Response of Added Signal", "Sum Response")

    # --- Multiplication ---
    with tab4:
        st.header("Multiplication Property")
        mult_type = st.selectbox("Second signal type for multiplication", ["Sine", "Square", "Random Noise"], key="mult")
        mult_freq = st.slider("Second Signal Frequency", min_value=1, max_value=20, value=2, key="mult_freq")
        mult_amp = st.slider("Second Signal Amplitude", min_value=0.1, max_value=5.0, value=1.0, key="mult_amp")
        if mult_type == "Sine":
            sig2 = generate_sine(length, mult_freq, mult_amp)
        elif mult_type == "Square":
            sig2 = generate_square(length, mult_freq, mult_amp)
        else:
            sig2 = generate_noise(length, mult_amp)
        multiplied_sig = multiplication(sig, sig2)
        multiplied_response = compute_impulse_response(impulse, multiplied_sig)
        st.write(f"Multiplied {signal_type} and {mult_type} signals")
        plot_signal(sig2, f"Second Signal ({mult_type})", mult_type)
        plot_signal(multiplied_sig, "Multiplied Signal", "Product")
        plot_signal(multiplied_response, "Impulse Response of Multiplied Signal", "Product Response")

if __name__ == "__main__":
    main()

