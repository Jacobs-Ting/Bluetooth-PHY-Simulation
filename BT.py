import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# ==========================================
# Instrument-Grade Visual Style (Dark Theme)
# ==========================================
plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "#121212",      
    "figure.facecolor": "#0E1117",    
    "grid.color": "#333333",          
    "grid.linestyle": "--",
    "text.color": "#E0E0E0",
    "axes.labelcolor": "#E0E0E0",
    "xtick.color": "#A0A0A0",
    "ytick.color": "#A0A0A0",
    "font.size": 9,
})

# Color Palette
COLOR_SIG_GOOD = "#00E5FF"   # Neon Cyan (Normal Signal)
COLOR_SIG_BAD = "#FF1744"    # Neon Red (Collision/Error)
COLOR_ENV = "#FFEA00"        # Neon Yellow (Time Domain Envelope)
COLOR_SPEC = "#00E676"       # Lime Green (Spectrum Trace)
COLOR_WIFI = "#D50000"       # Dark Red (Wi-Fi Interference Zone)
COLOR_SLOT = "#B388FF"       # Light Purple (Time Slot Boundary)

# ==========================================
# Digital Signal Processing (DSP) Module
# ==========================================
def add_awgn_noise(signal, snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

def generate_bits(pattern, num_bits):
    if pattern == "11110000": return np.tile([1, 1, 1, 1, 0, 0, 0, 0], (num_bits // 8) + 1)[:num_bits]
    elif pattern == "10101010": return np.tile([1, 0, 1, 0, 1, 0, 1, 0], (num_bits // 8) + 1)[:num_bits]
    else: return np.random.randint(0, 2, num_bits)

def generate_gfsk(bits, sps=8, bt=0.5, h=0.32):
    symbols = 2 * bits - 1  
    t = np.arange(-2*sps, 2*sps+1) / sps
    gauss_filter = (np.sqrt(np.pi)/np.sqrt(np.log(2))) * bt * np.exp(-(t * np.pi * bt / np.sqrt(np.log(2)))**2)
    gauss_filter /= np.sum(gauss_filter)
    upsampled = np.zeros(len(bits) * sps)
    upsampled[::sps] = symbols
    freq_dev = np.convolve(upsampled, gauss_filter, mode='same')
    phase = np.cumsum(freq_dev) * h * np.pi / sps
    return np.exp(1j * phase), np.exp(1j * phase)[::sps]

def generate_psk(bits, psk_type='2DH1', sps=8):
    if psk_type == '2DH1':
        if len(bits) % 2 != 0: bits = np.append(bits, 0)
        sym_vals = bits[0::2] * 2 + bits[1::2] 
        shifts = (sym_vals * 2 + 1) * np.pi / 4
    else:
        if len(bits) % 3 != 0: bits = np.append(bits, [0]*(3 - len(bits)%3))
        sym_vals = bits[0::3] * 4 + bits[1::3] * 2 + bits[2::3]
        shifts = sym_vals * np.pi / 4
    phase = np.cumsum(shifts)
    symbols = np.exp(1j * phase)
    upsampled = np.zeros(len(sym_vals) * sps, dtype=complex)
    upsampled[::sps] = symbols
    t = np.linspace(-1, 1, 3*sps)
    lpf = np.sinc(t) * np.hanning(3*sps)
    lpf = (lpf / np.sum(lpf)) * sps  
    return np.convolve(upsampled, lpf, mode='same'), symbols

def generate_edr_packet(payload_bits, sps=8, psk_type='2DH1'):
    gfsk_iq, gfsk_samples = generate_gfsk(generate_bits("PRBS9", 126), sps=sps)
    guard_symbols = 5
    guard_iq, guard_samples = np.zeros(guard_symbols * sps, dtype=complex), np.zeros(guard_symbols, dtype=complex)
    sync_bits = np.ones(11 * (2 if psk_type == '2DH1' else 3), dtype=int) 
    psk_iq, psk_samples = generate_psk(np.concatenate((sync_bits, payload_bits)), psk_type=psk_type, sps=sps)
    return np.concatenate((gfsk_iq, guard_iq, psk_iq)), np.concatenate((gfsk_samples, guard_samples, psk_samples))

PACKET_SPECS = {
    'DH1': {'mod': 'GFSK', 'slots': 1, 'max_sym': 216}, 'DH3': {'mod': 'GFSK', 'slots': 3, 'max_sym': 1464}, 'DH5': {'mod': 'GFSK', 'slots': 5, 'max_sym': 2712},
    '2-DH1': {'mod': '2DH', 'slots': 1, 'max_sym': 216}, '2-DH3': {'mod': '2DH', 'slots': 3, 'max_sym': 1468}, '2-DH5': {'mod': '2DH', 'slots': 5, 'max_sym': 2716},
    '3-DH1': {'mod': '3DH', 'slots': 1, 'max_sym': 221}, '3-DH3': {'mod': '3DH', 'slots': 3, 'max_sym': 1472}, '3-DH5': {'mod': '3DH', 'slots': 5, 'max_sym': 2722},
}

# ==========================================
# State Machine for FHSS Engine
# ==========================================
if 'bt_clock' not in st.session_state: st.session_state.bt_clock = 0
if 'hop_history' not in st.session_state: st.session_state.hop_history = []
if 'pseudo_random_seq' not in st.session_state:
    np.random.seed(1234)
    st.session_state.pseudo_random_seq = np.random.permutation(79)

# ==========================================
# Streamlit UI Setup
# ==========================================
st.set_page_config(page_title="Bluetooth RF PHY Studio", layout="wide", initial_sidebar_state="expanded")
st.title("📶 Bluetooth RF PHY Studio")
st.markdown("### Advanced Fixed-Frequency & FHSS Coexistence Simulator")

app_mode = st.sidebar.radio("🧪 Instrument Operating Mode", ["1️⃣ Fixed-Frequency (Baseband Spectrum)", "2️⃣ FHSS & Coexistence Simulation"])
st.sidebar.divider()

st.sidebar.header("⚙️ RF Parameters")
packet_type = st.sidebar.selectbox("Packet Type", list(PACKET_SPECS.keys()))
pkt_info = PACKET_SPECS[packet_type]
data_pattern = st.sidebar.selectbox("Data Pattern", ["PRBS9 (Random)", "11110000", "10101010"])
base_snr = st.sidebar.slider("Ambient SNR (dB)", 10, 50, 35)
payload_symbols = st.sidebar.number_input(f"Payload Symbols (Max: {pkt_info['max_sym']})", min_value=10, max_value=pkt_info['max_sym'], value=pkt_info['max_sym'], step=50)

is_collision = False
effective_snr = base_snr

if "FHSS" in app_mode:
    st.sidebar.divider()
    st.sidebar.header("📡 FHSS Engine Control")
    if st.sidebar.button("🚀 Trigger TX (Next Hop)", type="primary", use_container_width=True):
        st.session_state.bt_clock += pkt_info['slots'] + 1
        st.session_state.hop_history.append(st.session_state.pseudo_random_seq[(st.session_state.bt_clock // 2) % 79])
        if len(st.session_state.hop_history) > 12: st.session_state.hop_history.pop(0)

    current_ch = st.session_state.hop_history[-1] if st.session_state.hop_history else 0
    current_freq = 2402 + current_ch

    st.sidebar.header("🔥 External Interference")
    enable_wifi = st.sidebar.toggle("Enable Wi-Fi (802.11g/n)")
    wifi_ch_name = st.sidebar.selectbox("Wi-Fi Channel", ["CH 1 (2412 MHz)", "CH 6 (2437 MHz)", "CH 11 (2462 MHz)"])
    wifi_center = int(wifi_ch_name[6:10])

    if enable_wifi and abs(current_freq - wifi_center) <= 10:
        is_collision = True
        effective_snr = 2 

# ==========================================
# Waveform Generation & AWGN
# ==========================================
SPS = 8
if pkt_info['mod'] == 'GFSK':
    iq_ideal, sym_ideal = generate_gfsk(generate_bits(data_pattern, payload_symbols), sps=SPS)
elif pkt_info['mod'] == '2DH':
    iq_ideal, sym_ideal = generate_edr_packet(generate_bits(data_pattern, payload_symbols * 2), sps=SPS, psk_type='2DH1')
else:
    iq_ideal, sym_ideal = generate_edr_packet(generate_bits(data_pattern, payload_symbols * 3), sps=SPS, psk_type='3DH5')

iq_signal = add_awgn_noise(iq_ideal, effective_snr)
sampled_points = add_awgn_noise(sym_ideal, effective_snr)

# ==========================================
# Dashboard Metrics
# ==========================================
if "FHSS" in app_mode:
    if is_collision: st.error(f"💥 **COLLISION DETECTED!** Carrier at {current_freq} MHz (CH {current_ch}) encountered severe Wi-Fi interference.")
    else: st.success(f"✅ **TX SUCCESS!** Carrier hopped to {current_freq} MHz (CH {current_ch}) cleanly.")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Modulation", pkt_info['mod'])
col_m2.metric("Effective SNR", f"{effective_snr} dB", delta="-28 dB (Collision)" if is_collision else "Normal", delta_color="inverse" if is_collision else "normal")

if pkt_info['mod'] != 'GFSK':
    err_vec = sampled_points[142:] - sym_ideal[142:]
    devm_rms, devm_peak = np.sqrt(np.mean(np.abs(err_vec)**2)) * 100, np.max(np.abs(err_vec)) * 100
    lim_rms, lim_peak = (20.0, 30.0) if pkt_info['mod'] == '2DH' else (13.0, 20.0)
    col_m3.metric("RMS DEVM", f"{devm_rms:.2f} %", delta=f"Limit: {lim_rms}%", delta_color="normal" if devm_rms <= lim_rms else "inverse")
    col_m4.metric("Peak DEVM", f"{devm_peak:.2f} %", delta=f"Limit: {lim_peak}%", delta_color="normal" if devm_peak <= lim_peak else "inverse")
else:
    col_m3.metric("RMS DEVM", "N/A (GFSK)")
    col_m4.metric("Peak DEVM", "N/A (GFSK)")

st.divider()

# ==========================================
# Matplotlib Charts Area
# ==========================================
col_chart1, col_chart2 = st.columns([1, 1.8])

with col_chart1:
    fig_const, ax_const = plt.subplots(figsize=(5, 5))
    dot_color = COLOR_SIG_BAD if is_collision else COLOR_SIG_GOOD
    
    if pkt_info['mod'] != 'GFSK':
        ax_const.set_title(f"PSK Constellation ({data_pattern})", color='white', pad=10)
        if pkt_info['mod'] == '2DH':
            ax_const.add_patch(plt.Polygon([[1,0], [0,1], [-1,0], [0,-1]], fill=False, edgecolor='#666666', linestyle='--'))
            r2 = np.sqrt(2)/2
            ax_const.add_patch(plt.Polygon([[r2,r2], [-r2,r2], [-r2,-r2], [r2,-r2]], fill=False, edgecolor='#880000', alpha=0.5))
        ax_const.scatter(np.real(sampled_points[142:]), np.imag(sampled_points[142:]), s=10, c=dot_color, alpha=0.8)
    else:
        ax_const.set_title(f"GFSK Constellation ({data_pattern})", color='white', pad=10)
        ax_const.scatter(np.real(sampled_points), np.imag(sampled_points), s=10, c=dot_color, alpha=0.8)
        
    ax_const.axhline(0, color='#555555', linewidth=1); ax_const.axvline(0, color='#555555', linewidth=1)
    ax_const.set_xlim(-1.5, 1.5); ax_const.set_ylim(-1.5, 1.5)
    ax_const.grid(True)
    st.pyplot(fig_const)

with col_chart2:
    fig_time, ax_time = plt.subplots(figsize=(10, 4.2))
    ax_time.set_title("Time Domain Envelope (Baseband)", color='white', pad=10)
    time_us = np.arange(len(iq_signal)) / SPS 
    env_color = COLOR_SIG_BAD if is_collision else COLOR_ENV
    
    ax_time.plot(time_us, np.abs(iq_signal), color=env_color, linewidth=1.5, alpha=0.9)
    
    slots = pkt_info['slots']
    for i in range(slots + 1):
        x_pos = i * 625
        ax_time.axvline(x=x_pos, color=COLOR_SLOT, linestyle=':', linewidth=1.5, alpha=0.8)
        if i < slots:
            ax_time.text(x_pos + 10, 1.85, f"Slot {i}", color=COLOR_SLOT, fontweight='bold')
    
    ax_time.set_xlim(-50, (slots * 625) + 150)
    ax_time.set_ylim(0, 2.0)
    ax_time.set_xlabel("Time (µs)")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True)
    st.pyplot(fig_time)

st.divider()

if "Fixed-Frequency" in app_mode:
    fig_spec, ax_spec = plt.subplots(figsize=(15, 3.5))
    ax_spec.set_title("Power Spectral Density (PSD)", color='white', pad=10)
    f, Pxx = welch(iq_signal, fs=SPS*1e6, nperseg=512, return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx_db = 10 * np.log10(np.fft.fftshift(Pxx) / np.max(Pxx))
    
    ax_spec.plot(f / 1e6, Pxx_db, color=COLOR_SPEC, linewidth=1.5)
    ax_spec.fill_between(f / 1e6, Pxx_db, -80, color=COLOR_SPEC, alpha=0.15)
    
    ax_spec.set_xlim(-3, 3)
    ax_spec.set_ylim(-60, 5)
    ax_spec.set_xlabel("Frequency Offset (MHz)")
    ax_spec.set_ylabel("Relative Power (dB)")
    ax_spec.grid(True)
    st.pyplot(fig_spec)

else:
    fig_hop, ax_hop = plt.subplots(figsize=(15, 3.5))
    ax_hop.set_title("2.4 GHz ISM Band Spectrum Analyzer", color='white', pad=10)
    freqs = 2402 + np.arange(79)
    
    ax_hop.vlines(x=freqs, ymin=0, ymax=0.05, color='#444444', linewidth=1.5)
    
    if enable_wifi:
        ax_hop.axvspan(wifi_center - 10, wifi_center + 10, color=COLOR_WIFI, alpha=0.25)
        ax_hop.text(wifi_center, 1.15, f"802.11 Interference ({wifi_ch_name})", color='#FF5252', fontsize=11, fontweight='bold', ha='center')

    if len(st.session_state.hop_history) > 1:
        for i, ch in enumerate(st.session_state.hop_history[:-1]):
            alpha = 0.15 + (0.5 * (i / len(st.session_state.hop_history)))
            ax_hop.vlines(x=2402+ch, ymin=0, ymax=0.6, color='#2979FF', alpha=alpha, linewidth=3)

    if st.session_state.hop_history:
        h_color = COLOR_SIG_BAD if is_collision else COLOR_SIG_GOOD
        ax_hop.vlines(x=current_freq, ymin=0, ymax=1.0, color=h_color, linewidth=5)
        ax_hop.scatter(current_freq, 1.0, color=h_color, s=80, edgecolors='white', zorder=5)
        ax_hop.text(current_freq, 1.08, f"{current_freq} MHz", color=h_color, ha='center', fontweight='bold')

    ax_hop.set_xlim(2400, 2482); ax_hop.set_ylim(0, 1.35)
    ax_hop.set_xlabel("Frequency (MHz)")
    ax_hop.set_yticks([]) 
    ax_hop.set_xticks(np.arange(2402, 2481, 4)) 
    ax_hop.grid(axis='x')
    st.pyplot(fig_hop)