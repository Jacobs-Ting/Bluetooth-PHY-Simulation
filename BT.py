import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# ==========================================
# 數位訊號處理 (DSP) 模組
# ==========================================
def add_awgn_noise(signal, snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

def generate_bits(pattern, num_bits):
    if pattern == "11110000":
        base = [1, 1, 1, 1, 0, 0, 0, 0]
        return np.tile(base, (num_bits // 8) + 1)[:num_bits]
    elif pattern == "10101010":
        base = [1, 0, 1, 0, 1, 0, 1, 0]
        return np.tile(base, (num_bits // 8) + 1)[:num_bits]
    else:
        return np.random.randint(0, 2, num_bits)

def generate_gfsk(bits, sps=8, bt=0.5, h=0.32):
    symbols = 2 * bits - 1  
    num_symbols = len(bits)
    t = np.arange(-2*sps, 2*sps+1) / sps
    gauss_filter = (np.sqrt(np.pi)/np.sqrt(np.log(2))) * bt * np.exp(-(t * np.pi * bt / np.sqrt(np.log(2)))**2)
    gauss_filter = gauss_filter / np.sum(gauss_filter)
    upsampled = np.zeros(num_symbols * sps)
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
    header_bits = generate_bits("PRBS9", 126) 
    gfsk_iq, gfsk_samples = generate_gfsk(header_bits, sps=sps)
    guard_symbols = 5
    guard_iq = np.zeros(guard_symbols * sps, dtype=complex)
    guard_samples = np.zeros(guard_symbols, dtype=complex)
    sync_bits_len = 11 * (2 if psk_type == '2DH1' else 3)
    sync_bits = np.ones(sync_bits_len, dtype=int) 
    full_psk_bits = np.concatenate((sync_bits, payload_bits))
    psk_iq, psk_samples = generate_psk(full_psk_bits, psk_type=psk_type, sps=sps)
    return np.concatenate((gfsk_iq, guard_iq, psk_iq)), np.concatenate((gfsk_samples, guard_samples, psk_samples))

PACKET_SPECS = {
    'DH1':   {'mod': 'GFSK', 'slots': 1, 'max_sym': 216},
    'DH3':   {'mod': 'GFSK', 'slots': 3, 'max_sym': 1464},
    'DH5':   {'mod': 'GFSK', 'slots': 5, 'max_sym': 2712},
    '2-DH1': {'mod': '2DH',  'slots': 1, 'max_sym': 216},
    '2-DH3': {'mod': '2DH',  'slots': 3, 'max_sym': 1468},
    '2-DH5': {'mod': '2DH',  'slots': 5, 'max_sym': 2716},
    '3-DH1': {'mod': '3DH',  'slots': 1, 'max_sym': 221},
    '3-DH3': {'mod': '3DH',  'slots': 3, 'max_sym': 1472},
    '3-DH5': {'mod': '3DH',  'slots': 5, 'max_sym': 2722},
}

# ==========================================
# 狀態機管理 (Session State)
# ==========================================
if 'bt_clock' not in st.session_state:
    st.session_state.bt_clock = 0
if 'hop_history' not in st.session_state:
    st.session_state.hop_history = []
if 'pseudo_random_seq' not in st.session_state:
    np.random.seed(1234)
    st.session_state.pseudo_random_seq = np.random.permutation(79)

# ==========================================
# Streamlit UI 介面
# ==========================================
st.set_page_config(page_title="藍牙波形模擬器", layout="wide")
st.title("📶 藍牙 RF 綜合模擬器 (定頻 / 跳頻)")

# ★ 新增：運作模式切換
app_mode = st.sidebar.radio("🧪 選擇測試模式", ["1️⃣ 定頻測試 (Fixed Frequency)", "2️⃣ 跳頻模擬 (FHSS & Coexistence)"])
st.sidebar.markdown("---")

st.sidebar.header("共通參數設定")
packet_type = st.sidebar.selectbox("選擇藍牙封包類型", list(PACKET_SPECS.keys()))
pkt_info = PACKET_SPECS[packet_type]
data_pattern = st.sidebar.selectbox("Payload Data Pattern", ["PRBS9 (Random)", "11110000", "10101010"])
base_snr = st.sidebar.slider("環境基準訊噪比 SNR (dB)", 10, 50, 30)
payload_symbols = st.sidebar.number_input(f"Payload Symbol (Max: {pkt_info['max_sym']})", min_value=10, max_value=pkt_info['max_sym'], value=pkt_info['max_sym'], step=50)

# ==========================================
# 模式專屬邏輯 (跳頻與干擾)
# ==========================================
is_collision = False
effective_snr = base_snr

if "跳頻模擬" in app_mode:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 FHSS 跳頻引擎")
    if st.sidebar.button("🚀 傳送下一個封包 (Next Hop)", type="primary"):
        clock_advance = pkt_info['slots'] + 1
        st.session_state.bt_clock += clock_advance
        current_ch = st.session_state.pseudo_random_seq[(st.session_state.bt_clock // 2) % 79]
        st.session_state.hop_history.append(current_ch)
        if len(st.session_state.hop_history) > 12:
            st.session_state.hop_history.pop(0)

    current_ch = st.session_state.hop_history[-1] if st.session_state.hop_history else 0
    current_freq = 2402 + current_ch

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔥 Wi-Fi 干擾源設定")
    enable_wifi = st.sidebar.checkbox("啟用 Wi-Fi 干擾 (802.11g/n)", value=False)
    wifi_channels = {"CH 1 (2412 MHz)": 2412, "CH 6 (2437 MHz)": 2437, "CH 11 (2462 MHz)": 2462}
    wifi_ch_name = st.sidebar.selectbox("選擇 Wi-Fi 佔用通道", list(wifi_channels.keys()))
    wifi_center = wifi_channels[wifi_ch_name]

    if enable_wifi and abs(current_freq - wifi_center) <= 10:
        is_collision = True
        effective_snr = 2 # 碰撞時 SNR 爛掉

# ==========================================
# 波形生成
# ==========================================
SPS = 8
if pkt_info['mod'] == 'GFSK':
    bits = generate_bits(data_pattern, payload_symbols * 1)
    iq_signal_ideal, sampled_points_ideal = generate_gfsk(bits, sps=SPS)
elif pkt_info['mod'] == '2DH':
    bits = generate_bits(data_pattern, payload_symbols * 2)
    iq_signal_ideal, sampled_points_ideal = generate_edr_packet(bits, sps=SPS, psk_type='2DH1')
else:
    bits = generate_bits(data_pattern, payload_symbols * 3)
    iq_signal_ideal, sampled_points_ideal = generate_edr_packet(bits, sps=SPS, psk_type='3DH5')

iq_signal = add_awgn_noise(iq_signal_ideal, effective_snr)
sampled_points = add_awgn_noise(sampled_points_ideal, effective_snr)

# ==========================================
# UI 提示與 DEVM 計算
# ==========================================
if "跳頻模擬" in app_mode:
    if is_collision:
        st.error(f"💥 **發生頻譜碰撞！** 藍牙載波跳至 {current_freq} MHz (CH {current_ch})，落入 {wifi_ch_name} (20MHz 頻寬) 干擾範圍！")
    else:
        st.success(f"✅ **傳輸成功！** 藍牙載波位於 {current_freq} MHz (CH {current_ch})，避開了干擾。")

st.subheader("🎯 訊號品質指標 (Signal Quality Metrics)")
col_m1, col_m2, col_m3 = st.columns(3)

if pkt_info['mod'] != 'GFSK':
    payload_start_idx = 142
    ideal_payload_symbols = sampled_points_ideal[payload_start_idx:]
    noisy_payload_symbols = sampled_points[payload_start_idx:]
    error_vector = noisy_payload_symbols - ideal_payload_symbols
    devm_rms = np.sqrt(np.mean(np.abs(error_vector)**2)) * 100
    devm_peak = np.max(np.abs(error_vector)) * 100
    limit_rms = 20.0 if pkt_info['mod'] == '2DH' else 13.0
    limit_peak = 30.0 if pkt_info['mod'] == '2DH' else 20.0
    color_rms = "normal" if devm_rms <= limit_rms else "inverse"
    color_peak = "normal" if devm_peak <= limit_peak else "inverse"
    col_m1.metric(label="RMS DEVM (%)", value=f"{devm_rms:.2f} %", delta=f"Limit: {limit_rms}%", delta_color=color_rms)
    col_m2.metric(label="Peak DEVM (%)", value=f"{devm_peak:.2f} %", delta=f"Limit: {limit_peak}%", delta_color=color_peak)
else:
    col_m1.metric("RMS DEVM (%)", "N/A", delta="GFSK 不適用", delta_color="off")
    col_m2.metric("Peak DEVM (%)", "N/A", delta="GFSK 不適用", delta_color="off")

col_m3.info(f"當前有效 SNR：**{effective_snr} dB**")
st.divider()

# ==========================================
# 繪圖區塊 1: 星座圖與時域包絡線
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    fig_const, ax_const = plt.subplots(figsize=(5, 5))
    if pkt_info['mod'] != 'GFSK':
        plot_samples = sampled_points[142:] 
        ax_const.set_title(f"PSK Constellation")
        if pkt_info['mod'] == '2DH':
            ax_const.add_patch(plt.Polygon([[1,0], [0,1], [-1,0], [0,-1]], fill=False, edgecolor='gray', linestyle='--', alpha=0.5))
            r2 = np.sqrt(2)/2
            ax_const.add_patch(plt.Polygon([[r2,r2], [-r2,r2], [-r2,-r2], [r2,-r2]], fill=False, edgecolor='red', alpha=0.5))
    else:
        plot_samples = sampled_points
        ax_const.set_title(f"GFSK Constellation")
        
    dot_color = 'red' if is_collision else 'blue'
    ax_const.scatter(np.real(plot_samples), np.imag(plot_samples), s=15, c=dot_color, alpha=0.6)
    ax_const.axhline(0, color='black', linewidth=0.5); ax_const.axvline(0, color='black', linewidth=0.5)
    ax_const.set_xlim(-1.5, 1.5); ax_const.set_ylim(-1.5, 1.5)
    st.pyplot(fig_const)

with col2:
    fig_time, ax_time = plt.subplots(figsize=(10, 5))
    time_us = np.arange(len(iq_signal)) / SPS 
    envelope = np.abs(iq_signal)
    line_color = 'darkred' if is_collision else 'black'
    ax_time.plot(time_us, envelope, color=line_color, linewidth=1.2)
    
    allocated_slots = pkt_info['slots']
    for i in range(allocated_slots + 1):
        slot_start = i * 625
        ax_time.axvline(x=slot_start, color='purple', linestyle=':', linewidth=2)
        if i < allocated_slots: ax_time.text(slot_start + 10, 1.85, f"Slot {i}", color='purple', fontweight='bold')
    
    ax_time.set_xlim(-50, (allocated_slots * 625) + 200); ax_time.set_ylim(0, 2.0)
    st.pyplot(fig_time)

# ==========================================
# 繪圖區塊 2: 依據模式顯示不同圖表
# ==========================================
st.divider()

if "定頻測試" in app_mode:
    # 模式 1: 顯示基頻頻譜
    st.subheader("📊 基頻頻譜 PSD (Baseband Spectrum)")
    fig_spec, ax_spec = plt.subplots(figsize=(15, 3))
    f, Pxx = welch(iq_signal, fs=SPS*1e6, nperseg=512, return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    Pxx_db = 10 * np.log10(Pxx / np.max(Pxx))
    ax_spec.plot(f / 1e6, Pxx_db, color='green')
    ax_spec.set_xlabel("Frequency (MHz)")
    ax_spec.set_ylabel("Power (dB)")
    ax_spec.set_ylim(-60, 5)
    ax_spec.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_spec)

else:
    # 模式 2: 顯示跳頻監視器
    st.subheader("📻 2.4GHz 頻譜共存監視器 (Bluetooth vs Wi-Fi)")
    channels = np.arange(79)
    frequencies = 2402 + channels
    fig_hop, ax_hop = plt.subplots(figsize=(15, 3))
    
    ax_hop.vlines(x=frequencies, ymin=0, ymax=0.1, color='gray', alpha=0.2, linewidth=2)
    
    if enable_wifi:
        ax_hop.axvspan(wifi_center - 10, wifi_center + 10, color='red', alpha=0.2, label=f'Wi-Fi 20MHz ({wifi_ch_name})')
        ax_hop.text(wifi_center, 1.1, f"Wi-Fi {wifi_ch_name[:4]}", color='red', fontsize=12, fontweight='bold', ha='center')

    if len(st.session_state.hop_history) > 1:
        for i, ch in enumerate(st.session_state.hop_history[:-1]):
            alpha_val = 0.1 + (0.5 * (i / len(st.session_state.hop_history)))
            ax_hop.vlines(x=2402 + ch, ymin=0, ymax=0.5, color='dodgerblue', alpha=alpha_val, linewidth=4)
            ax_hop.scatter(2402 + ch, 0.5, color='dodgerblue', alpha=alpha_val, s=30)

    if st.session_state.hop_history:
        hop_color = 'red' if is_collision else 'blue'
        hop_label = f'Collision at {current_freq} MHz!' if is_collision else f'Current Hop: {current_freq} MHz'
        ax_hop.vlines(x=current_freq, ymin=0, ymax=1.0, color=hop_color, linewidth=6, label=hop_label)
        ax_hop.scatter(current_freq, 1.0, color=hop_color, s=100)

    ax_hop.set_xlim(2400, 2482); ax_hop.set_ylim(0, 1.3)
    ax_hop.set_xlabel("Frequency (MHz)")
    ax_hop.set_yticks([]) 
    ax_hop.set_xticks(np.arange(2402, 2481, 4)) 
    ax_hop.legend(loc='upper right')
    ax_hop.grid(axis='x', linestyle='--', alpha=0.5)
    st.pyplot(fig_hop)