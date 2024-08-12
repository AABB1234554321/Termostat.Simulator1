import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import ParameterGrid

# --- Uygulama Başlığı ve Açıklaması ---

st.title("Makine Öğrenmesi Tabanlı İç Ortam Sıcaklık Kontrol Simülatörü")
st.markdown("""
Bu interaktif simülasyon, oda sıcaklığını istenen seviyede tutmak için farklı kontrol algoritmalarını (Açma-Kapama, PID, Q-Öğrenme) karşılaştırmanıza olanak tanır. 
Dış hava sıcaklığı verilerini CSV dosyası yükleyerek veya interpolasyon yöntemi ile sağlayabilirsiniz.
""")

# --- Dış Hava Sıcaklığı Veri Girişi ---

st.sidebar.header("Dış Hava Sıcaklığı Veri Girişi")
data_source = st.sidebar.radio("Veri Kaynağı Seçin:", ("CSV Dosyası", "İnterpolasyon"))

if data_source == "CSV Dosyası":
    uploaded_file = st.sidebar.file_uploader("CSV dosyası yükleyin", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            outdoor_temp_values = df['Outdoor Temp (C)'].values
        except KeyError:
            st.error("CSV dosyası 'Outdoor Temp (C)' sütununu içermiyor. Lütfen doğru dosyayı yükleyin.")
            st.stop()
elif data_source == "İnterpolasyon":
    st.sidebar.markdown("**Her saat için bir sıcaklık değeri girin (Toplam 24 değer):**")
    hourly_temps = []
    for hour in range(24):
        temp = st.sidebar.number_input(f"{hour}:00", min_value=-20, max_value=50, value=15)
        hourly_temps.append(temp)
    outdoor_temp_values = None  # İnterpolasyon için bu değişkeni daha sonra kullanacağız

# --- Simülasyon Parametreleri ---

st.sidebar.header("Simülasyon Parametreleri")
initial_room_temperature = st.sidebar.number_input("Başlangıç Oda Sıcaklığı (°C)", min_value=10, max_value=30, value=20)
thermostat_setting = st.sidebar.number_input("Termostat Ayarı (°C)", min_value=15, max_value=25, value=22)
heater_power = st.sidebar.slider("Isıtıcı Gücü (°C/dakika)", min_value=0.1, max_value=0.5, value=0.3)
base_heat_loss = st.sidebar.slider("Temel Isı Kaybı (°C/dakika)", min_value=0.05, max_value=0.2, value=0.1)
simulation_minutes = st.sidebar.number_input("Simülasyon Süresi (Dakika)", min_value=10, max_value=5000, value=60)
thermostat_sensitivity = st.sidebar.slider("Termostat Hassasiyeti (°C)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# --- Algoritma Seçimi ---

algorithms = ["Açma-Kapama", "PID", "Q-Öğrenme"]
selected_algorithms = st.sidebar.multiselect("Simülasyon Türü(lerini) Seçin:", algorithms)

# --- Q-Öğrenme ve PID Parametreleri (Kullanıcı girişi kaldırıldı) ---

# Q-learning için hiperparametre aralıkları
q_learning_param_grid = {
    'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9],
    'discount_factor': [0.9, 0.95, 0.99],
    'exploration_decay': [0.0001, 0.001, 0.01]
}

# PID için hiperparametre aralıkları
pid_param_grid = {
    'Kp': [0.1, 0.5, 1.0, 2.0, 5.0],
    'Ki': [0.01, 0.05, 0.1, 0.2, 0.5],
    'Kd': [0.001, 0.01, 0.1, 0.2, 0.5]
}

# --- Global Variables ---

num_states = 101  # Durum sayısı arttırıldı
num_actions = 2
q_table = np.zeros((num_states, num_actions))

# Default value for episodes
episodes = 5000 

# --- Yardımcı Fonksiyonlar ---

def get_state(temperature):
    """Sıcaklığı durumlara ayırır."""
    return int(min(100, max(0, (temperature - 10) / 0.2)))  # Durum sayısı arttığı için sıcaklık aralığı da ayarlandı

def get_action(state, q_table, exploration_rate):
    """Epsilon-greedy politikasına göre bir eylem seçer."""
    if np.random.uniform(0, 1) < exploration_rate:
        return np.random.choice(num_actions)  # Exploration
    else:
        return np.argmax(q_table[state, :])  # Exploitation

def get_reward(state, action, thermostat_setting):
    """Durum ve eyleme göre ödülü hesaplar. Daha hassas bir ödül fonksiyonu."""
    state_temp = 10 + state * 0.2 

    if state_temp == thermostat_setting:
        return 20  # Tam olarak istenen sıcaklıkta
    elif abs(state_temp - thermostat_setting) <= 0.5:
        return 10  # Kabul edilebilir aralıkta
    elif action == 1 and state_temp > thermostat_setting:  # Çok sıcak ve ısıtıcı açık
        return -10
    elif action == 0 and state_temp < thermostat_setting:  # Çok soğuk ve ısıtıcı kapalı
        return -5
    else:
        return -1  # Diğer durumlarda hafif ceza

# --- Dış Ortam Sıcaklığı Hesaplama ---

def get_outdoor_temp(minute, outdoor_temp_values):
    if outdoor_temp_values is not None:  # CSV dosyası kullanılıyorsa
        index = int(minute // 5) 
        return outdoor_temp_values[min(index, len(outdoor_temp_values) - 1)]
    else:  # İnterpolasyon kullanılıyorsa
        hour = int(minute // 60)
        minute_in_hour = minute % 60
        x = np.linspace(0, 23, 24)
        y = hourly_temps
        f = interp1d(x, y, kind='cubic')
        return f(hour + minute_in_hour / 60)

# --- Simülasyon Mantığı (Açma-Kapama) ---

def run_on_off_simulation(initial_room_temperature, thermostat_sensitivity, outdoor_temp_values, 
                          simulation_minutes, heater_power, base_heat_loss, thermostat_setting):
    """Açma-kapama kontrol algoritması ile oda sıcaklığı simülasyonunu çalıştırır."""
    time = []
    room_temperatures = []
    room_temperature = initial_room_temperature
    heater_status = False
    last_switch_time = 0  # Son açma/kapama zamanını takip etmek için

    for minute in np.arange(0, simulation_minutes, 0.1):
        time.append(minute)

        outside_temperature = get_outdoor_temp(minute, outdoor_temp_values)

        if room_temperature < thermostat_setting - thermostat_sensitivity and minute - last_switch_time >= 1:
            heater_status = True
            last_switch_time = minute
        elif room_temperature > thermostat_setting + thermostat_sensitivity and minute - last_switch_time >= 1:
            heater_status = False
            last_switch_time = minute

        heat_loss = base_heat_loss * (room_temperature - outside_temperature) / 10

        if heater_status:
            room_temperature += heater_power * 0.1
        else:
            room_temperature -= heat_loss * 0.1

        room_temperatures.append(room_temperature)

    on_off_comfort_area = calculate_area_between_temp(time, room_temperatures, thermostat_setting)
    return time, room_temperatures, on_off_comfort_area

# --- Simülasyon Mantığı (Q-Öğrenme) ---

def run_q_learning_simulation(initial_room_temperature, thermostat_sensitivity, outdoor_temp_values, 
                              simulation_minutes, heater_power, base_heat_loss, thermostat_setting, episodes,
                              learning_rate, discount_factor, exploration_decay):
    """Q-öğrenme kontrol algoritması ile oda sıcaklığı simülasyonunu çalıştırır."""

    global q_table  # Global q_table'ı kullanıyoruz

    exploration_rate = 1.0  # Başlangıç keşif oranı yüksek

    for episode in range(episodes):
        room_temperature = initial_room_temperature
        state = get_state(room_temperature)
        last_switch_time = 0

        for minute in np.arange(0, simulation_minutes, 0.1):
            outside_temperature = get_outdoor_temp(minute, outdoor_temp_values)

            action = get_action(state, q_table, exploration_rate)

            if action == 1 and minute - last_switch_time >= 1:  # Isıtıcıyı aç (süre kısıtına uyarak)
                room_temperature += heater_power * 0.1
                last_switch_time = minute
            else:  # Isıtıcıyı kapat
                heat_loss = base_heat_loss * (room_temperature - outside_temperature) / 10
                room_temperature -= heat_loss * 0.1

            next_state = get_state(room_temperature)
            reward = get_reward(next_state, action, thermostat_setting)

            # Q-tablosunu güncelle
            q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])

            state = next_state

        # Keşif oranını azalt
        exploration_rate = max(0.01, exploration_rate - exploration_decay)  # Minimum keşif oranı 0.01

    # Öğrenilen Q-tablosu ile son simülasyonu çalıştır
    time = []
    room_temperatures = []
    room_temperature = initial_room_temperature
    state = get_state(room_temperature)
    last_switch_time = 0

    for minute in np.arange(0, simulation_minutes, 0.1):
        outside_temperature = get_outdoor_temp(minute, outdoor_temp_values)

        action = np.argmax(q_table[state, :])  # En iyi aksiyonu seç

        if action == 1 and minute - last_switch_time >= 1:
            room_temperature += heater_power * 0.1
            last_switch_time = minute
        else:
            heat_loss = base_heat_loss * (room_temperature - outside_temperature) / 10
            room_temperature -= heat_loss * 0.1

        state = get_state(room_temperature)

        time.append(minute)
        room_temperatures.append(room_temperature)

    q_learning_comfort_area = calculate_area_between_temp(time, room_temperatures, thermostat_setting)
    return time, room_temperatures, q_learning_comfort_area

# --- Simülasyon Mantığı (PID) ---

def run_pid_simulation(initial_room_temperature, thermostat_sensitivity, outdoor_temp_values, 
                       simulation_minutes, heater_power, base_heat_loss, thermostat_setting, Kp, Ki, Kd):
    """PID kontrol algoritması ile oda sıcaklığı simülasyonunu çalıştırır."""
    time = []
    room_temperatures = []
    heater_output = []

    integral_error = 0
    previous_error = 0
    room_temperature = initial_room_temperature
    last_switch_time = 0
    heater_status = False  # Isıtıcı durumunu takip etmek için

    for minute in np.arange(0, simulation_minutes, 0.1):
        time.append(minute)

        outside_temperature = get_outdoor_temp(minute, outdoor_temp_values)

        error = thermostat_setting - room_temperature

        proportional_term = Kp * error

        integral_error += error * 0.1

        integral_term = Ki * integral_error

        derivative_term = Kd * (error - previous_error) / 0.1

        previous_error = error

        pid_output = proportional_term + integral_term + derivative_term

        # PID çıkışını sınırla 
        pid_output = max(0, min(pid_output, 1))

        # Isıtıcı durumunu ve açma/kapama süresini kontrol et
        if pid_output > 0 and (not heater_status or minute - last_switch_time >= 1):
            heater_status = True
            last_switch_time = minute
        elif pid_output == 0:
            heater_status = False

        heater_output.append(pid_output if heater_status else 0)  # Isıtıcı kapalıysa çıkışı 0 yap

        heat_loss = base_heat_loss * (room_temperature - outside_temperature) / 10

        room_temperature += (heater_power * heater_output[-1] - heat_loss) * 0.1

        room_temperatures.append(room_temperature)

    pid_comfort_area = calculate_area_between_temp(time, room_temperatures, thermostat_setting)
    return time, room_temperatures, pid_comfort_area

# --- Alan Hesaplama Fonksiyonları ---

def calculate_area_between_temp(time, room_temperatures, set_temp):
    """Mevcut sıcaklık ve ayarlanan sıcaklık arasındaki alanı hesaplar."""
    area = 0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        avg_temp = (room_temperatures[i] + room_temperatures[i - 1]) / 2
        area += abs(avg_temp - set_temp) * dt
    return area

def calculate_area_metrics(time, room_temperatures, set_temp):
    """Aşım ve alt geçiş alanlarını hesaplar."""
    overshoot = 0
    undershoot = 0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        avg_temp = (room_temperatures[i] + room_temperatures[i - 1]) / 2

        if avg_temp > set_temp:
            overshoot += (avg_temp - set_temp) * dt
        elif avg_temp < set_temp:
            undershoot += (set_temp - avg_temp) * dt
    return overshoot, undershoot

# --- Ana Çalıştırma Fonksiyonu ---

def run_simulations():
    """Seçilen simülasyonları çalıştırır ve sonuçları görselleştirir."""

    results = {}
    overshoots = {}
    undershoots = {}

    if "Açma-Kapama" in selected_algorithms:
        time_on_off, room_temperatures_on_off, on_off_comfort_area = \
            run_on_off_simulation(initial_room_temperature, thermostat_sensitivity, outdoor_temp_values,
                                  simulation_minutes, heater_power, base_heat_loss, thermostat_setting)

        overshoot_on_off, undershoot_on_off = calculate_area_metrics(time_on_off, room_temperatures_on_off, thermostat_setting)

        results["Açma-Kapama"] = {'time': time_on_off, 'room_temperatures': room_temperatures_on_off}
        overshoots["Açma-Kapama"] = overshoot_on_off
        undershoots["Açma-Kapama"] = undershoot_on_off

    if "Q-Öğrenme" in selected_algorithms:
        best_params = None
        best_comfort_area = float('-inf')  # En iyi konfor alanı başlangıçta -sonsuz

        # Izgara araması ile en iyi parametreleri bul
        for params in ParameterGrid(q_learning_param_grid):
            time_q, room_temperatures_q, q_learning_comfort_area = \
                run_q_learning_simulation(initial_room_temperature, thermostat_sensitivity, outdoor_temp_values,
                                          simulation_minutes, heater_power, base_heat_loss, thermostat_setting, episodes,
                                          params['learning_rate'], params['discount_factor'], params['exploration_decay'])

            if q_learning_comfort_area > best_comfort_area:
                best_comfort_area = q_learning_comfort_area
                best_params = params

        # En iyi parametrelerle son simülasyonu çalıştır
        time_q, room_temperatures_q, _ = \
            run_q_learning_simulation(initial_room_temperature, thermostat_sensitivity, outdoor_temp_values,
                                      simulation_minutes, heater_power, base_heat_loss, thermostat_setting, episodes,
                                      best_params['learning_rate'], best_params['discount_factor'], best_params['exploration_decay'])

        overshoot_q, undershoot_q = calculate_area_metrics(time_q, room_temperatures_q, thermostat_setting)

        results["Q-Öğrenme"] = {'time': time_q, 'room_temperatures': room_temperatures_q}
        overshoots["Q-Öğrenme"] = overshoot_q
        undershoots["Q-Öğrenme"] = undershoot_q

        st.write(f"**Q-Öğrenme için En İyi Parametreler:** {best_params}")

    if "PID" in selected_algorithms:
        best_params = None
        best_comfort_area = float('-inf')

        # Izgara araması ile en iyi parametreleri bul
        for params in ParameterGrid(pid_param_grid):
            time_pid, room_temperatures_pid, pid_comfort_area = \
                run_pid_simulation(initial_room_temperature, thermostat_sensitivity, outdoor_temp_values,
                                   simulation_minutes, heater_power, base_heat_loss, thermostat_setting,
                                   params['Kp'], params['Ki'], params['Kd'])

            if pid_comfort_area > best_comfort_area:
                best_comfort_area = pid_comfort_area
                best_params = params

        # En iyi parametrelerle son simülasyonu çalıştır
        time_pid, room_temperatures_pid, _ = \
            run_pid_simulation(initial_room_temperature, thermostat_sensitivity, outdoor_temp_values,
                               simulation_minutes, heater_power, base_heat_loss, thermostat_setting,
                               best_params['Kp'], best_params['Ki'], best_params['Kd'])

        overshoot_pid, undershoot_pid = calculate_area_metrics(time_pid, room_temperatures_pid, thermostat_setting)

        results["PID"] = {'time': time_pid, 'room_temperatures': room_temperatures_pid}
        overshoots["PID"] = overshoot_pid
        undershoots["PID"] = undershoot_pid

        st.write(f"**PID için En İyi Parametreler:** {best_params}")

    # --- Grafikleri Oluştur ve Görselleştir ---

    st.subheader("Oda Sıcaklığı Kontrol Simülasyonu")  # Daha açıklayıcı bir başlık
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, data in results.items():
        ax.plot(data['time'], data['room_temperatures'], label=f"Oda Sıcaklığı ({algo})")

    ax.axhline(y=thermostat_setting, color='r', linestyle='--', label="Termostat Ayarı")
    ax.set_xlabel("Zaman (Dakika)")
    ax.set_ylabel("Sıcaklık (°C)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # --- Konfor ve Enerji Metrikleri ---

    st.subheader("Konfor ve Enerji Tüketimi Metrikleri")

    # Aşım ve Alt Geçiş Değerleri Tablosu
    metrics_data = []
    for algo in labels:
        metrics_data.append({
            'Algoritma': algo,
            'Aşım (°C*dakika)': overshoots[algo],
            'Alt Geçiş (°C*dakika)': undershoots[algo]
        })
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df.style.format("{:.2f}"))  # Tabloyu daha iyi biçimlendirme ile göster

    # Aşım ve Alt Geçiş Çubuk Grafiği
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(labels))

    ax2.bar(x - width/2, overshoot_values, width, label='Aşım', color='skyblue')
    ax2.bar(x + width/2, undershoot_values, width, label='Alt Geçiş', color='lightcoral')

    ax2.set_ylabel('Alan (°C*dakika)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()

    st.pyplot(fig2)

    # --- Toplam Aşım ve Alt Geçiş Karşılaştırması ---

    st.subheader("Toplam Aşım ve Alt Geçiş Karşılaştırması")

    # Toplam Alan Değerleri Tablosu
    total_metrics_data = []
    for algo, total_value in total_overshoot_undershoot.items():
        total_metrics_data.append({
            'Algoritma': algo,
            'Toplam Alan (°C*dakika)': total_value
        })
    total_metrics_df = pd.DataFrame(total_metrics_data)
    st.table(total_metrics_df.style.format("{:.2f}"))

    # Toplam Aşım ve Alt Geçiş Çubuk Grafiği
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    ax3.bar(total_overshoot_undershoot.keys(), total_overshoot_undershoot.values(), color=['blue', 'green', 'orange'])
    ax3.set_ylabel('Toplam Alan (°C*dakika)')

    st.pyplot(fig3)

    # --- Dış Ortam Sıcaklığı Grafiği ---

    st.subheader("Dış Ortam Sıcaklığı Grafiği")
    outdoor_time = np.arange(0, simulation_minutes, 5)
    outdoor_temps = [get_outdoor_temp(minute, outdoor_temp_values) for minute in outdoor_time]
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(outdoor_time, outdoor_temps, label="Dış Ortam Sıcaklığı", color='purple')
    ax4.set_xlabel("Zaman (dakika)")
    ax4.set_ylabel("Dış Ortam Sıcaklığı (°C)")
    ax4.legend()
    st.pyplot(fig4)

# --- Main Execution ---

if __name__ == "__main__":
    if st.sidebar.button("Simülasyonları Çalıştır", key='run_button') and (outdoor_temp_values is not None or data_source == "İnterpolasyon"):
        run_simulations()
    elif st.sidebar.button("Simülasyonları Çalıştır", key='error_button'):
        st.error("Lütfen önce dış hava sıcaklığı verilerini sağlayın.")
