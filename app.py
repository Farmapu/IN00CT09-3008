from pathlib import Path
import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt

LINEAR_PATH = Path("data/Linear_Acceleration.csv")
CORDINATES_PATH = Path("data/Location.csv")

if LINEAR_PATH:
    df = pd.read_csv(LINEAR_PATH, sep=";")

if CORDINATES_PATH:
    gpsdf = pd.read_csv(CORDINATES_PATH, sep=";")

lat = gpsdf["Latitude"].values
lon = gpsdf["Longitude"].values

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

distances = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])

total_distance = np.sum(distances)

df = df.apply(pd.to_numeric, errors="coerce").dropna()
df = df.sort_values("Time (s)")
ax = df["Linear Acceleration x (m/s^2)"].values
time = df["Time (s)"].values

ax = ax - np.mean(ax)
window = 20
ax_smooth = pd.Series(ax).rolling(window, center=True).mean().bfill().ffill().values
vx = np.zeros_like(ax_smooth)

for i in range(1, len(time)):
    dt = time[i] - time[i-1]
    vx[i] = vx[i-1] + 0.5 * (ax_smooth[i] + ax_smooth[i-1]) * dt

avg_speed = np.mean(np.abs(vx))

min_step_time = 0.5

min_step_samples = int(min_step_time / np.median(np.diff(time)))

peaks = []
for i in range(1, len(ax_smooth)-1):
    if ax_smooth[i] > ax_smooth[i-1] and ax_smooth[i] > ax_smooth[i+1]:
        if len(peaks) == 0 or (i - peaks[-1]) >= min_step_samples:
            peaks.append(i)

num_steps = len(peaks)

time = df["Time (s)"]
acc = df["Linear Acceleration x (m/s^2)"]

acc_ema = acc.ewm(alpha=0.5, adjust=False).mean()
fig, ax = plt.subplots(figsize=(10, 4))

#Tulosten printtaus
st.title("Päivän liikunta")
st.text(f"Askelia: {num_steps:.1f}")
st.text(f"Keskinopeus: {avg_speed:.1f}m/s")
st.text(f"Kokonaismatka: {total_distance:.1f}m")
if num_steps > 0:
    avg_step_length = total_distance / num_steps
    st.text(f"Askeleen pituus: {avg_step_length:.5f} m")

st.title("Nopeus")
ax.plot(time, acc_ema, color="red", linewidth=2)
ax.set_xlabel("Aika (s)")
ax.set_ylabel("Suodatettu ay (m/s²)")
ax.grid(True)

st.pyplot(fig)
dt = np.median(np.diff(time))
fs = 1.0 / dt

acc = acc - np.mean(acc)
window = np.hanning(len(acc))
acc_win = acc * window
N = len(acc_win)

fft_vals = np.fft.rfft(acc_win)
freqs = np.fft.rfftfreq(N, d=dt)
power = (np.abs(fft_vals) ** 2) / N

st.title("Tehospektri")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(freqs, power, color="steelblue", linewidth=1)
ax.set_xlabel("Taajuus [Hz]")
ax.set_ylabel("Teho")
ax.grid(True)
ax.set_xlim(0, 14)
ax.set_ylim(bottom=0)
st.pyplot(fig)

st.title("Karttakuva")
gpsdf = gpsdf.dropna(subset=["Latitude", "Longitude"])
gpsdf["Latitude"] = gpsdf["Latitude"].astype(float)
gpsdf["Longitude"] = gpsdf["Longitude"].astype(float)
path = gpsdf[["Longitude", "Latitude"]].values.tolist()
path_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": path}],
    get_path="path",
    get_color=[255, 0, 0],
    width_scale=5,
    width_min_pixels=1,
)
point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=gpsdf,
    get_position="[Longitude, Latitude]",
    get_radius=5,
    get_color=[0, 0, 255],
    pickable=True,
)
view_state = pdk.ViewState(
    latitude=gpsdf["Latitude"].mean(),
    longitude=gpsdf["Longitude"].mean(),
    zoom=14,
)
deck = pdk.Deck(
    layers=[path_layer, point_layer],
    initial_view_state=view_state,
    tooltip={
        "text": "Time: {Time(s)}s\nSpeed: {Velocity(m/s)} m/s"
    }
)
st.pydeck_chart(deck)