import sys, time
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import imageio_ffmpeg

from matplotlib.animation import FuncAnimation, FFMpegWriter

from tqdm import tqdm

mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()



### HW PHYSICS ###

def apply_response(signal, response_spec):
    if response_spec is None:
        return signal

    kernel = assign_kernel(response_spec)

    if kernel is None:
        return signal

    pad_left = len(kernel) // 2
    pad_right = len(kernel) - 1 - pad_left

    padded_signal = np.pad(signal, (pad_left, pad_right), mode="edge")
    return np.convolve(padded_signal, kernel, mode="valid")

def assign_kernel(spec):
    response_type = spec.get("type", "none")
    n = spec.get("n", 101)

    if response_type == "gaussian":
        sigma = spec.get("sigma", 15)
        x = np.arange(n) - (n - 1) / 2
        k = np.exp(-0.5 * (x / sigma) ** 2)

    elif response_type == "exp":
        tau = spec.get("tau", 15)
        x = np.arange(n)
        k = np.exp(-x / tau)

    elif response_type == "none":
        return None

    else:
        raise ValueError(f"Unknown response type: {response_type}")

    return k / np.sum(k)

def gaussian_kernel(n=101, sigma=15):
    x = np.arange(n) - (n - 1) / 2
    k = np.exp(-0.5 * (x / sigma)**2)
    return (k / np.sum(k)).tolist()

def exponential_kernel(n=101, tau=15):
    x = np.arange(n)
    k = np.exp(-x / tau)
    return (k / np.sum(k)).tolist()


def apply_polynomial_nonlinearity(x, nonlinear_spec):
    if nonlinear_spec is None:
        return x

    quadratic = nonlinear_spec.get("quadratic", 0.0)
    cubic = nonlinear_spec.get("cubic", 0.0)
    compression = nonlinear_spec.get("compression", 0.0)
    clip = nonlinear_spec.get("clip", None)

    y = x.astype(float, copy=True)
    y = y + quadratic * x**2 + cubic * x**3

    if compression != 0.0:
        y = y / (1.0 + compression * y**2)

    if clip is not None:
        y = np.clip(y, -clip, clip)

    return y

def apply_optical_nonlinearity(acoustic_field, optical_spec):
    if optical_spec is None:
        optical_field = acoustic_field.copy()
        optical_intensity = optical_field**2
        return optical_field, optical_intensity

    efficiency = optical_spec.get("efficiency", 1.0)
    saturation = optical_spec.get("saturation", 0.0)
    contrast_floor = optical_spec.get("contrast_floor", 0.0)
    optical_cubic = optical_spec.get("cubic", 0.0)

    acoustic_power = acoustic_field**2

    if saturation > 0.0:
        optical_intensity = efficiency * acoustic_power / (1.0 + saturation * acoustic_power)
    else:
        optical_intensity = efficiency * acoustic_power

    if optical_cubic != 0.0:
        optical_intensity = optical_intensity + optical_cubic * acoustic_power**2

    optical_intensity = np.maximum(optical_intensity, contrast_floor)
    optical_field = np.sqrt(np.maximum(optical_intensity, 0.0)) * np.sign(acoustic_field)

    return optical_field, optical_intensity








### WAVEFORM ABSTRACTION ###

def poly4(coefs, t):
	return coefs[0] + coefs[1] * t + coefs[2] * t**2 + coefs[3] * t**3 + coefs[4] * t**4

def NCO(frequency, phase, t):
	return np.sin(frequency * t + phase)

def tone(amplitude, NCO):
	return amplitude*NCO

def multitone(tones):
	return np.sum(tones)

def waveform(specs):
    tones = specs["tones"]

    duration = specs["duration"]
    t, dt = make_timebase(specs)

    waveform = np.zeros_like(t)
    tone_data = {}

    for tone in tones:
        name = tone["name"]

        tone_t0 = tone.get("t0", 0)
        tone_duration = tone.get("tone_duration", duration)
        tone_t1 = tone_t0 + tone_duration

        active = (t >= tone_t0) & (t <= tone_t1)
        local_t = t - tone_t0

        amplitude = np.zeros_like(t)
        frequency = np.zeros_like(t)
        phase_offset = np.zeros_like(t)

        amplitude[active] = poly4(tone["amplitude"], local_t[active])
        frequency[active] = poly4(tone["frequency"], local_t[active])
        phase_offset[active] = poly4(tone["phase"], local_t[active])

        accumulated_phase = 2 * np.pi * np.cumsum(frequency * 1000) * dt #in ms and MHz
        signal = amplitude * np.cos(accumulated_phase + phase_offset)

        waveform += signal

        tone_data[name] = {
            "space_coord": tone["space_coord"],
            "t0": tone_t0,
            "duration": tone_duration,
            "t1": tone_t1,
            "active": active,
            "amplitude": amplitude,
            "frequency": frequency,
            "phase_offset": phase_offset,
            "carrier_phase": accumulated_phase,
            "phase": accumulated_phase + phase_offset,
            "signal": signal,
        }

    return t, waveform, tone_data

def waveform_with_distorsion(specs):

    tones = specs["tones"]

    duration = specs["duration"]
    t, dt = make_timebase(specs)

    waveform = np.zeros_like(t)
    tone_data = {}

    for tone in tones:
        name = tone["name"]

        tone_t0 = tone.get("t0", 0.0)
        tone_duration = tone.get("tone_duration", duration)
        tone_t1 = tone_t0 + tone_duration

        active = (t >= tone_t0) & (t <= tone_t1)
        after = t > tone_t1
        local_t = t - tone_t0

        amplitude = np.zeros_like(t)
        frequency = np.full_like(t, tone["frequency"][0])
        phase_offset = np.zeros_like(t)

        amplitude[active] = poly4(tone["amplitude"], local_t[active])
        frequency[active] = poly4(tone["frequency"], local_t[active])
        phase_offset[active] = poly4(tone["phase"], local_t[active])

        final_frequency = poly4(tone["frequency"], tone_duration)
        final_phase_offset = poly4(tone["phase"], tone_duration)

        frequency[after] = final_frequency
        phase_offset[after] = final_phase_offset

        raw_amplitude = amplitude.copy()
        raw_frequency = frequency.copy()
        raw_phase_offset = phase_offset.copy()

        response = tone.get("response", {})

        amplitude = apply_response(amplitude, response.get("amplitude"))
        frequency = apply_response(frequency, response.get("frequency"))
        phase_offset = apply_response(phase_offset, response.get("phase"))

        frequency_plot = frequency.copy()
        phase_offset_plot = phase_offset.copy()
        frequency_plot[amplitude < 1e-3] = np.nan
        phase_offset_plot[amplitude < 1e-3] = np.nan

        accumulated_phase = 2 * np.pi * np.cumsum(frequency * 1000) * dt
        signal = amplitude * np.cos(accumulated_phase + phase_offset)

        waveform += signal

        tone_data[name] = {
            "space_coord": tone["space_coord"],
            "t0": tone_t0,
            "tone_duration": tone_duration,
            "t1": tone_t1,
            "active": active,
            "raw_amplitude": raw_amplitude,
            "raw_frequency": raw_frequency,
            "raw_phase_offset": raw_phase_offset,
            "amplitude": amplitude,
            "frequency": frequency,
            "frequency_plot": frequency_plot,
            "phase_offset": phase_offset,
            "phase_offset_plot": phase_offset_plot,
            "carrier_phase": accumulated_phase,
            "phase": accumulated_phase + phase_offset,
            "signal": signal,
        }

    return t, waveform, tone_data

def waveform_multitone_full_sim(specs):
    tones = specs["tones"]
    duration = specs["duration"]
    response = specs.get("response", {})
    t, dt = make_timebase(specs)

    rf_input = np.zeros_like(t)
    tone_data = {}

    for tone in tones:
        name = tone["name"]
        tone_response = tone.get("response", response)

        tone_t0 = tone.get("t0", 0.0)
        tone_duration = tone.get("tone_duration", duration)
        tone_t1 = tone_t0 + tone_duration

        active = (t >= tone_t0) & (t <= tone_t1)
        after = t > tone_t1
        local_t = t - tone_t0

        amplitude = np.zeros_like(t)
        frequency = np.full_like(t, tone["frequency"][0])
        phase_offset = np.zeros_like(t)

        amplitude[active] = poly4(tone["amplitude"], local_t[active])
        frequency[active] = poly4(tone["frequency"], local_t[active])
        phase_offset[active] = poly4(tone["phase"], local_t[active])

        final_frequency = poly4(tone["frequency"], tone_duration)
        final_phase_offset = poly4(tone["phase"], tone_duration)
        frequency[after] = final_frequency
        phase_offset[after] = final_phase_offset

        raw_amplitude = amplitude.copy()
        raw_frequency = frequency.copy()
        raw_phase_offset = phase_offset.copy()

        if tone_response is not None:
            amplitude = apply_response(amplitude, tone_response.get("amplitude"))
            frequency = apply_response(frequency, tone_response.get("frequency"))
            phase_offset = apply_response(phase_offset, tone_response.get("phase"))

        frequency_plot = frequency.copy()
        phase_offset_plot = phase_offset.copy()
        frequency_plot[amplitude < 1e-3] = np.nan
        phase_offset_plot[amplitude < 1e-3] = np.nan

        accumulated_phase = 2 * np.pi * np.cumsum(frequency * 1000.0) * dt
        signal = amplitude * np.cos(accumulated_phase + phase_offset)
        rf_input += signal

        tone_data[name] = {
            "space_coord": tone["space_coord"],
            "t0": tone_t0,
            "tone_duration": tone_duration,
            "t1": tone_t1,
            "active": active,
            "raw_amplitude": raw_amplitude,
            "raw_frequency": raw_frequency,
            "raw_phase_offset": raw_phase_offset,
            "amplitude": amplitude,
            "frequency": frequency,
            "frequency_plot": frequency_plot,
            "phase_offset": phase_offset,
            "phase_offset_plot": phase_offset_plot,
            "carrier_phase": accumulated_phase,
            "phase": accumulated_phase + phase_offset,
            "signal": signal,
        }

    rf_after_chain = apply_response(rf_input, response.get("rf_chain"))
    rf_nonlinear = apply_polynomial_nonlinearity(rf_after_chain, response.get("nonlinear"))
    acoustic_field = apply_response(rf_nonlinear, response.get("acoustic"))
    optical_field, optical_intensity = apply_optical_nonlinearity(acoustic_field, response.get("optical"))

    chain_data = {
        "rf_input": rf_input,
        "rf_after_chain": rf_after_chain,
        "rf_nonlinear": rf_nonlinear,
        "acoustic_field": acoustic_field,
        "optical_field": optical_field,
        "optical_intensity": optical_intensity,
    }

    return t, optical_field, tone_data, chain_data





### VISUALIZATION ###

def plot_specs(t_x, x_waveform, x_data, t_y, y_waveform, y_data):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 12), sharex="col", constrained_layout=True)

    datasets = [
        ("X", t_x, x_waveform, x_data, 0),
        ("Y", t_y, y_waveform, y_data, 1),
    ]

    for axis_name, t, wave, data_dict, col in datasets:

        peak = np.max(np.abs(wave))
        axes[0, col].plot(t, wave / peak if peak > 0 else wave, color="k", linewidth=2, label="Summed")
        for name, data in data_dict.items():
            axes[0, col].plot(t, data["signal"], alpha=0.25, linewidth=1, label=name)
        axes[0, col].set_title(f"{axis_name}: waveform")
        axes[0, col].set_ylabel("Signal")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True)

        for name, data in data_dict.items():
            axes[1, col].plot(t, data["amplitude"], label=name)
        axes[1, col].set_title(f"{axis_name}: amplitude(t)")
        axes[1, col].set_ylabel("Amplitude")
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True)

        for name, data in data_dict.items():
            axes[2, col].plot(t, data.get("frequency_plot", data["frequency"]), label=name)
        axes[2, col].set_title(f"{axis_name}: frequency(t)")
        axes[2, col].set_ylabel("Frequency")
        axes[2, col].legend(fontsize=8)
        axes[2, col].grid(True)

        for name, data in data_dict.items():
            axes[3, col].plot(t, data.get("phase_offset_plot", data["phase_offset"]), label=name)
        axes[3, col].set_title(f"{axis_name}: phase(t)")
        axes[3, col].set_ylabel("Phase")
        axes[3, col].set_xlabel("t [ms]")
        axes[3, col].legend(fontsize=8)
        axes[3, col].grid(True)

    plt.show()

def visualize_aod_positions(fx_map, fy_map, x_data, y_data, title="2D crossed AOD addressed positions", show_map_labels=True, max_label_points=100):
    fx_map = np.asarray(fx_map)
    fy_map = np.asarray(fy_map)

    if fx_map.shape != fy_map.shape:
        raise ValueError("fx_map and fy_map must have the same shape.")

    def first_active_frequency(data):
        idx = np.flatnonzero(data["active"])
        if len(idx) == 0:
            return None
        freq = data.get("raw_frequency", data["frequency"])
        return freq[idx[0]]


    x_tone_freqs = {name: first_active_frequency(data) for name, data in x_data.items()}
    y_tone_freqs = {name: first_active_frequency(data) for name, data in y_data.items()}

    x_tone_freqs = {name: fx for name, fx in x_tone_freqs.items() if fx is not None}
    y_tone_freqs = {name: fy for name, fy in y_tone_freqs.items() if fy is not None}

    rows, cols = fx_map.shape
    addressed_points = []

    plt.figure(figsize=(9, 8))

    for i in range(rows):
        for j in range(cols):
            plt.scatter(j, i, color="lightgray", s=80, zorder=1)

            if show_map_labels and rows * cols <= max_label_points:
                plt.text(j, i - 0.12, f"({j},{i})\nfx={fx_map[i,j]:.4f}\nfy={fy_map[i,j]:.4f}", ha="center", va="top", fontsize=6, color="gray")

    for x_name, fx in x_tone_freqs.items():
        col = np.argmin(np.abs(fx_map[0, :] - fx))
        plt.axvline(col, alpha=0.25, linewidth=3, label=f"{x_name}: fx={fx:.4f}")

    for y_name, fy in y_tone_freqs.items():
        row = np.argmin(np.abs(fy_map[:, 0] - fy))
        plt.axhline(row, alpha=0.25, linewidth=3, label=f"{y_name}: fy={fy:.4f}")

    for x_name, fx in x_tone_freqs.items():
        j = np.argmin(np.abs(fx_map[0, :] - fx))

        for y_name, fy in y_tone_freqs.items():
            i = np.argmin(np.abs(fy_map[:, 0] - fy))

            plt.scatter(j, i, color="dodgerblue", s=180, zorder=5)
            plt.text(j, i + 0.18, f"({j}, {i})\n{x_name}, {y_name}\nfx={fx:.4f}\nfy={fy:.4f}", ha="center", va="bottom", fontsize=7, zorder=6)

            addressed_points.append({
                "i": int(i),
                "j": int(j),
                "x_position": int(j),
                "y_position": int(i),
                "x_tone": x_name,
                "y_tone": y_name,
                "fx": float(fx),
                "fy": float(fy),
            })

    plt.gca().invert_yaxis()
    plt.xlabel("x position / column index")
    plt.ylabel("y position / row index")
    plt.title(title)
    plt.grid(True)
    # plt.legend(fontsize=7, loc="best")
    plt.show()

    return addressed_points

def visualize_aod_trajectories(fx_map, fy_map, x_data, y_data, title="2D AOD trajectories from frequency modulation", stride=100):
    fx_map = np.asarray(fx_map)
    fy_map = np.asarray(fy_map)

    if fx_map.shape != fy_map.shape:
        raise ValueError("fx_map and fy_map must have the same shape.")

    rows, cols = fx_map.shape
    trajectories = []

    plt.figure(figsize=(9, 8))

    for i in range(rows):
        for j in range(cols):
            plt.scatter(j, i, color="lightgray", s=80, zorder=1)

    for x_name, x_tone in x_data.items():
        fx_t = x_tone["frequency"]

        for y_name, y_tone in y_data.items():
            fy_t = y_tone["frequency"]

            x_pos = np.interp(fx_t, fx_map[0, :], np.arange(cols))
            y_pos = np.interp(fy_t, fy_map[:, 0], np.arange(rows))

            line, = plt.plot(x_pos[::stride], y_pos[::stride], linewidth=2, label=f"{x_name} × {y_name}", zorder=3)
            color = line.get_color()

            plt.scatter(x_pos[0], y_pos[0], marker="o", s=130, facecolors="none", edgecolors=color, linewidth=2, zorder=5)
            plt.scatter(x_pos[-1], y_pos[-1], marker="X", s=150, color=color, zorder=5)

            plt.text(x_pos[0], y_pos[0] + 0.08, f"Start\n{x_name}\n{y_name}\n({x_pos[0]:.2f}, {y_pos[0]:.2f})\nfx={fx_t[0]:.4f}\nfy={fy_t[0]:.4f}", fontsize=7, ha="center", va="bottom", color=color)
            plt.text(x_pos[-1], y_pos[-1] - 0.08, f"End\n({x_pos[-1]:.2f}, {y_pos[-1]:.2f})\nfx={fx_t[-1]:.4f}\nfy={fy_t[-1]:.4f}", fontsize=7, ha="center", va="top", color=color)

            trajectories.append({"x_tone": x_name, "y_tone": y_name, "fx": fx_t, "fy": fy_t, "x": x_pos, "y": y_pos})

    plt.gca().invert_yaxis()
    plt.xlabel("X position / column index")
    plt.ylabel("Y position / row index")
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=7)
    plt.show()

    return trajectories

def animate_aod_trajectories(fx_map, fy_map, x_data, y_data, filename="aod_trajectories.mp4", fps=30, stride=100, verbose=False):
    fx_map = np.asarray(fx_map)
    fy_map = np.asarray(fy_map)

    if fx_map.shape != fy_map.shape:
        raise ValueError("fx_map and fy_map must have the same shape.")

    rows, cols = fx_map.shape
    trajectories = []

    for x_name, x_tone in x_data.items():
        for y_name, y_tone in y_data.items():
            fx_t = x_tone["frequency"]
            fy_t = y_tone["frequency"]

            x_pos = np.interp(fx_t, fx_map[0, :], np.arange(cols))
            y_pos = np.interp(fy_t, fy_map[:, 0], np.arange(rows))

            x_anim = x_pos[::stride]
            y_anim = y_pos[::stride]
            fx_anim = fx_t[::stride]
            fy_anim = fy_t[::stride]

            if x_name == "x_tone_1" and y_name == "y_tone_2" and verbose:
                print("x_anim =", x_anim)
                print("y_anim =", y_anim)
            trajectories.append({"label": f"{x_name} × {y_name}", "x": x_anim, "y": y_anim, "fx": fx_anim, "fy": fy_anim})

    n_frames = len(trajectories[0]["x"])

    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(rows):
        for j in range(cols):
            ax.scatter(j, i, color="lightgray", s=80, zorder=1)

    lines = []
    points = []

    for traj in trajectories:
        line, = ax.plot([], [], linewidth=2, label=traj["label"], zorder=3)
        color = line.get_color()

        point, = ax.plot([], [], marker="o", markersize=7, color=color, linestyle="None", zorder=5)

        ax.plot(traj["x"][0], traj["y"][0], marker="o", markersize=9, markerfacecolor="none", markeredgecolor=color, markeredgewidth=2, linestyle="None", zorder=5)
        ax.plot(traj["x"][-1], traj["y"][-1], marker="X", markersize=9, color=color, linestyle="None", zorder=6)

        lines.append(line)
        points.append(point)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=10)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xlabel("X position / column index")
    ax.set_ylabel("Y position / row index")
    ax.set_title("2D AOD trajectory animation")
    ax.grid(True)
    # ax.legend(fontsize=7, loc="best")

    def update(n):
        for traj, line, point in zip(trajectories, lines, points):
            line.set_data(traj["x"][:n + 1], traj["y"][:n + 1])
            point.set_data([traj["x"][n]], [traj["y"][n]])

        time_text.set_text(f"frame={n + 1}/{n_frames}")
        return lines + points + [time_text]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)

    writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(filename, writer=writer)
    plt.close(fig)

    return filename

def visualize_baseband_spectrogram(t, tone_data, center_freq=200.0, title="Baseband spectrogram", nfft=32768, hop=512, max_detuning=0.05, db_floor=-80):
    dt = t[1] - t[0]
    waveform_bb = np.zeros_like(t, dtype=complex)

    for name, data in tone_data.items():
        f_bb = data["frequency"] - center_freq
        phase_bb = 2 * np.pi * np.cumsum(f_bb * 1000.0) * dt + data["phase_offset"]
        waveform_bb += data["amplitude"] * np.exp(1j * phase_bb)

    nfft = min(nfft, len(t))
    hop = min(hop, nfft)
    window = np.hanning(nfft)

    times = []
    spectra = []

    for start in range(0, len(waveform_bb) - nfft + 1, hop):
        segment = waveform_bb[start:start + nfft] * window
        spectrum = np.fft.fftshift(np.fft.fft(segment))
        spectra.append(np.abs(spectrum))
        times.append(t[start + nfft // 2])

    if not spectra:
        raise ValueError("No spectrogram frames generated. Decrease nfft or increase the simulation length.")

    spectra = np.array(spectra).T
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=dt * 1e-3)) / 1e6

    keep = np.abs(freqs) <= max_detuning
    freqs = freqs[keep]
    spectra = spectra[keep, :]

    spectra_db = 20 * np.log10(spectra / np.max(spectra) + 1e-12)
    spectra_db = np.maximum(spectra_db, db_floor)

    plt.figure(figsize=(12, 6))
    plt.imshow(spectra_db, aspect="auto", origin="lower", extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap="viridis", vmin=db_floor, vmax=0)
    plt.colorbar(label="Magnitude [dB]")

    for name, data in tone_data.items():
        plt.plot(t, data["frequency"] - center_freq, linewidth=1, color="white", alpha=0.8, label=name)

    plt.xlabel("Time [ms]")
    plt.ylabel(f"Detuning from {center_freq} MHz")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

def visualize_programmed_spectrum(t, tone_data, center_freq=200.0, title="Programmed spectrum", sigma=0.01, n_freq=500, max_detuning=0.05):
    freqs = np.linspace(-max_detuning, max_detuning, n_freq)
    spectrum = np.zeros((n_freq, len(t)))

    for name, data in tone_data.items():
        detuning = data.get("frequency_plot", data["frequency"]) - center_freq
        amplitude = data["amplitude"]

        for k in range(len(t)):
            if np.isnan(detuning[k]):
                continue
            spectrum[:, k] += amplitude[k] * np.exp(-0.5 * ((freqs - detuning[k]) / sigma) ** 2)

    peak = np.max(spectrum)
    if peak > 0:
        spectrum_db = 20 * np.log10(spectrum / peak + 1e-12)
    else:
        spectrum_db = np.full_like(spectrum, -80.0)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        spectrum_db,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], freqs[0], freqs[-1]],
        cmap="viridis",
        vmin=-80,
        vmax=0,
    )
    plt.colorbar(label="Magnitude [dB]")

    for name, data in tone_data.items():
        plt.plot(
            t,
            data.get("frequency_plot", data["frequency"]) - center_freq,
            color="white",
            linewidth=1,
            alpha=0.9,
            label=name,
        )

    plt.xlabel("Time [ms]")
    plt.ylabel(f"Detuning from {center_freq} MHz")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

def visualize_full_chain(t, chain_data, title="Full multitone RF/acoustic/optical chain", stride=None):
    if stride is None:
        stride = max(1, len(t) // 20000)

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 12), sharex=True, constrained_layout=True)

    names = [
        ("RF input", "rf_input"),
        ("After RF chain", "rf_after_chain"),
        ("After nonlinear RF stage", "rf_nonlinear"),
        ("Acoustic field", "acoustic_field"),
        ("Optical intensity", "optical_intensity"),
    ]

    for ax, (label, key) in zip(axes, names):
        y = chain_data[key]
        peak = np.max(np.abs(y))
        y_plot = y / peak if peak > 0 else y
        ax.plot(t[::stride], y_plot[::stride], linewidth=1)
        ax.set_ylabel(label)
        ax.grid(True)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Time [ms]")
    plt.show()

def visualize_intermodulation_spectrum(chain_data, sampling_rate, center_freq=200.0, title="Intermodulation spectrum", nfft=None, max_detuning=5.0, db_floor=-100):
    stages = [
        ("RF input", chain_data["rf_input"]),
        ("After RF chain", chain_data["rf_after_chain"]),
        ("After nonlinear RF", chain_data["rf_nonlinear"]),
        ("Acoustic", chain_data["acoustic_field"]),
        ("Optical field", chain_data["optical_field"]),
    ]

    plt.figure(figsize=(14, 6))

    for label, signal in stages:
        freq, mag_db = compute_spectrum_db(signal, sampling_rate, nfft=nfft, db_floor=db_floor)
        detuning = freq - center_freq
        keep = np.abs(detuning) <= max_detuning
        plt.plot(detuning[keep], mag_db[keep], linewidth=1, label=label)

    plt.xlabel(f"Detuning from {center_freq} MHz")
    plt.ylabel("Magnitude [dB]")
    plt.title(title)
    plt.ylim(db_floor, 5)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def visualize_optical_effect(t,x_data,y_data,time_index=None,time_stride=None,wavelength_nm=830.0,acoustic_velocity_mm_per_us=4.2,aperture_x_mm=2.5,aperture_y_mm=0.32,grid_size=512,beam_waist_x_mm=0.35,beam_waist_y_mm=0.10,modulation_depth=0.8,db_floor=-80,title="2D diffracted optical intensity from crossed AODs",):
    wavelength_mm = wavelength_nm * 1e-6

    x = np.linspace(-aperture_x_mm / 2, aperture_x_mm / 2, grid_size)
    y = np.linspace(-aperture_y_mm / 2, aperture_y_mm / 2, grid_size)
    X, Y = np.meshgrid(x, y, indexing="xy")

    optical_input = np.exp(-((X / beam_waist_x_mm) ** 2 + (Y / beam_waist_y_mm) ** 2))

    if time_index is not None:
        time_indices = [time_index]
    else:
        if time_stride is None:
            time_stride = max(1, len(t) // 200)
        time_indices = range(0, len(t), time_stride)

    intensity_accumulated = np.zeros((grid_size, grid_size))

    for k in time_indices:
        t_us = t[k] * 1000.0

        acoustic_x = np.zeros(grid_size)
        acoustic_y = np.zeros(grid_size)

        for name, data in x_data.items():
            amp = data["amplitude"][k]
            freq = data["frequency"][k]
            phase = data["phase_offset"][k]
            k_ac = freq / acoustic_velocity_mm_per_us
            acoustic_x += amp * np.cos(2 * np.pi * (freq * t_us - k_ac * x) + phase)

        for name, data in y_data.items():
            amp = data["amplitude"][k]
            freq = data["frequency"][k]
            phase = data["phase_offset"][k]
            k_ac = freq / acoustic_velocity_mm_per_us
            acoustic_y += amp * np.cos(2 * np.pi * (freq * t_us - k_ac * y) + phase)

        acoustic_phase = modulation_depth * (acoustic_y[:, None] + acoustic_x[None, :])
        optical_field = optical_input * np.exp(1j * acoustic_phase)

        far_field = np.fft.fftshift(np.fft.fft2(optical_field))
        intensity = np.abs(far_field) ** 2
        intensity_accumulated += intensity

    intensity_accumulated /= np.max(intensity_accumulated)

    intensity_db = 10 * np.log10(intensity_accumulated + 1e-15)
    intensity_db = np.maximum(intensity_db, db_floor)

    fx = np.fft.fftshift(np.fft.fftfreq(grid_size, d=x[1] - x[0]))
    fy = np.fft.fftshift(np.fft.fftfreq(grid_size, d=y[1] - y[0]))

    theta_x_mrad = wavelength_mm * fx * 1000.0
    theta_y_mrad = wavelength_mm * fy * 1000.0

    plt.figure(figsize=(8, 8))
    plt.imshow(
        intensity_db,
        origin="lower",
        extent=[theta_x_mrad[0], theta_x_mrad[-1], theta_y_mrad[0], theta_y_mrad[-1]],
        cmap="inferno",
        vmin=db_floor,
        vmax=0,
        aspect="auto",
    )
    plt.colorbar(label="Normalized intensity [dB]")
    plt.xlabel("X deflection angle [mrad]")
    plt.ylabel("Y deflection angle [mrad]")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return intensity_accumulated, intensity_db, theta_x_mrad, theta_y_mrad

def compute_spectrum_db(signal, sampling_rate, nfft=None, db_floor=-120):
    if nfft is None:
        nfft = min(len(signal), 2**16)
    nfft = min(nfft, len(signal))

    window = np.hanning(nfft)
    segment = signal[:nfft] * window
    spec = np.fft.fftshift(np.fft.fft(segment))
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sampling_rate)) / 1e6
    mag = np.abs(spec)
    peak = np.max(mag)

    if peak > 0:
        mag_db = 20 * np.log10(mag / peak + 1e-15)
    else:
        mag_db = np.full_like(mag, db_floor, dtype=float)

    mag_db = np.maximum(mag_db, db_floor)
    return freq, mag_db



### DATA ACQUISTION ###

def print_memory_report(x_specs, y_specs, x_waveform, y_waveform, x_data, y_data, verbose=True):
    def sizeof(obj):
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, dict):
            return sys.getsizeof(obj) + sum(sizeof(k) + sizeof(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            return sys.getsizeof(obj) + sum(sizeof(v) for v in obj)
        return sys.getsizeof(obj)

    def human(nbytes):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if nbytes < 1024 or unit == "TB":
                return f"{nbytes:.2f} {unit}"
            nbytes /= 1024

    specs_size = sizeof(x_specs) + sizeof(y_specs)

    waveform_size = x_waveform.nbytes + y_waveform.nbytes

    amplitude_size = sum(t["amplitude"].nbytes for t in x_data.values()) + sum(t["amplitude"].nbytes for t in y_data.values())
    frequency_size = sum(t["frequency"].nbytes for t in x_data.values()) + sum(t["frequency"].nbytes for t in y_data.values())
    phase_size = sum(t["phase"].nbytes for t in x_data.values()) + sum(t["phase"].nbytes for t in y_data.values())
    signal_size = sum(t["signal"].nbytes for t in x_data.values()) + sum(t["signal"].nbytes for t in y_data.values())

    total_generated = waveform_size + amplitude_size + frequency_size + phase_size + signal_size

    if verbose:
        print("\n================== MEMORY REPORT ==================")
        print(f"{'Specs':<25}{human(specs_size):>15}")
        print("---------------------------------------------------")
        print(f"{'Waveforms':<25}{human(waveform_size):>15}")
        print(f"{'Amplitude arrays':<25}{human(amplitude_size):>15}")
        print(f"{'Frequency arrays':<25}{human(frequency_size):>15}")
        print(f"{'Phase arrays':<25}{human(phase_size):>15}")
        print(f"{'Signal arrays':<25}{human(signal_size):>15}")
        print("---------------------------------------------------")
        print(f"{'Generated data total':<25}{human(total_generated):>15}")
        print("===================================================\n")

    return specs_size/1024, total_generated/1024

def print_full_chain_report(specs, chain_data):
    def rms(x):
        return float(np.sqrt(np.mean(x**2)))

    print("\n" + "=" * 80)
    print(f"{specs['axis']} FULL MULTITONE CHAIN REPORT")
    print("=" * 80)
    for key in ["rf_input", "rf_after_chain", "rf_nonlinear", "acoustic_field", "optical_field", "optical_intensity"]:
        x = chain_data[key]
        print(f"{key:<24} peak={np.max(np.abs(x)):>12.6g} | rms={rms(x):>12.6g}")
    print("=" * 80)

def print_specs(specs):
    tones = specs["tones"]

    print("\n" + "=" * 120)
    print(f"{specs['axis']} AOD SPECIFICATIONS")
    print("=" * 120)
    print(f"{'Duration':<20}: {specs['duration']:.3f} ms")
    print(f"{'Sampling':<20}: {specs['sampling']:,} Sa/s")
    print(f"{'Number of tones':<20}: {len(tones)}")

    if tones:
        freqs = [tone["frequency"][0] for tone in tones]
        print(f"{'Frequency range':<20}: {min(freqs):.3f} MHz -> {max(freqs):.3f} MHz")

    print("-" * 120)

    for tone in tones:
        f0 = tone["frequency"][0]
        f1 = poly4(tone["frequency"], tone["tone_duration"])

        print(f"{tone['name']:<12} Pos={tone['space_coord']:>4} | t0={tone['t0']:>6.3f} ms | dur={tone['tone_duration']:>6.3f} ms")
        print(f"{'':<12} A coefs: {tone['amplitude']}")
        print(f"{'':<12} f coefs: {tone['frequency']} | f: {f0:.6f} -> {f1:.6f} MHz")
        print(f"{'':<12} p coefs: {tone['phase']}")
        print("-" * 120)

    print("=" * 120)




### AUTOMATIZATIONS ###

def generate_large_aod_specs(axis, n_tones=512, duration=4.1, sampling=int(10e6), f_min=200.020, f_max=200.520, pulse_duration=4.0, t0=0.0, sweep_span=1.0, amplitude=0.5, phase=np.pi / 4, response=None, seed=0):
    rng = np.random.default_rng(seed)
    frequencies = np.linspace(f_min, f_max, n_tones)
    sweep_rate = sweep_span / pulse_duration

    specs = {"axis": axis, "sampling": sampling, "duration": duration, "tones": []}

    for i, f0 in enumerate(frequencies):
        specs["tones"].append({
            "name": f"{axis.lower()}_tone_{i}",
            "space_coord": i,
            "t0": t0,
            "tone_duration": pulse_duration,
            "amplitude": [amplitude, 0.0, 0.0, 0.0, 0.0],
            "frequency": [float(f0), sweep_rate, 0.0, 0.0, 0.0],
            "phase": [float(phase), 0.0, 0.0, 0.0, 0.0],
            "response": response,
        })

    return specs

def make_aod_frequency_maps(x_specs, y_specs):
    x_freqs = [tone["frequency"][0] for tone in x_specs["tones"]]
    y_freqs = [tone["frequency"][0] for tone in y_specs["tones"]]

    fx_map = np.tile(x_freqs, (len(y_freqs), 1))
    fy_map = np.tile(np.array(y_freqs).reshape(-1, 1), (1, len(x_freqs)))

    return fx_map, fy_map

def make_aod_frequency_maps_from_data(x_data, y_data):

    n_x = len(x_data)
    n_y = len(y_data)

    x_freqs = np.concatenate([data.get("raw_frequency", data["frequency"])[data["active"]] for data in x_data.values()])
    y_freqs = np.concatenate([data.get("raw_frequency", data["frequency"])[data["active"]] for data in y_data.values()])

    x_axis = np.linspace(np.min(x_freqs), np.max(x_freqs), n_x)
    y_axis = np.linspace(np.min(y_freqs), np.max(y_freqs), n_y)

    fx_map = np.tile(x_axis, (n_y, 1))
    fy_map = np.tile(y_axis.reshape(-1, 1), (1, n_x))

    return fx_map, fy_map

def make_timebase(specs):
    duration_ms = specs["duration"]
    sampling_rate = specs["sampling"]
    n_samples = int(round(duration_ms * 1e-3 * sampling_rate))
    t = np.arange(n_samples) / sampling_rate * 1e3
    dt = 1 / sampling_rate * 1e3
    return t, dt





### PROGRAMS ###
def main(x_specs, y_specs):

    t0_stamp = time.perf_counter_ns()*1e-6
    t_x, x_waveform, x_data = waveform(x_specs)
    t_y, y_waveform, y_data = waveform(y_specs)
    calc_time = time.perf_counter_ns()*1e-6 - t0_stamp

    plot_specs(t_x, x_waveform, x_data, t_y, y_waveform, y_data)

    specs_size, total_data = print_memory_report(x_specs,y_specs,x_waveform,y_waveform,x_data,y_data,verbose=True)

    fx_map, fy_map = make_aod_frequency_maps(x_specs, y_specs)

    addressed_points = visualize_aod_positions(fx_map=fx_map,fy_map=fy_map,x_data=x_data,y_data=y_data,)

    # print(addressed_points)

    trajectories = visualize_aod_trajectories(fx_map=fx_map,fy_map=fy_map,x_data=x_data,y_data=y_data,stride=150,)
    print(trajectories[0]['fy'])
    # animate_aod_trajectories(fx_map, fy_map, x_data, y_data, filename="aod_trajectories.mp4", fps=30, stride=150)

    visualize_baseband_spectrogram(t_x, x_data, center_freq=200.0, title="X AOD baseband spectrogram", nfft=8192, hop=256, max_detuning=1.05)
    visualize_baseband_spectrogram(t_y, y_data, center_freq=200.0, title="Y AOD baseband spectrogram", nfft=8192, hop=256, max_detuning=1.05)

    visualize_programmed_spectrum(t_x, x_data, center_freq=200.0, title="X AOD programmed spectrum", sigma=0.001, max_detuning=1.05)
    visualize_programmed_spectrum(t_y, y_data, center_freq=200.0, title="Y AOD programmed spectrum", sigma=0.001, max_detuning=1.05)
    return specs_size, total_data, calc_time

def main_distorsion(x_specs, y_specs):
    t0_stamp = time.perf_counter_ns() * 1e-6
    t_x, x_waveform, x_data = waveform_with_distorsion(x_specs)
    t_y, y_waveform, y_data = waveform_with_distorsion(y_specs)
    calc_time = time.perf_counter_ns() * 1e-6 - t0_stamp

    plot_specs(t_x, x_waveform, x_data, t_y, y_waveform, y_data)

    specs_size, total_data = print_memory_report(x_specs, y_specs, x_waveform, y_waveform, x_data, y_data, verbose=True)
    fx_map, fy_map = make_aod_frequency_maps_from_data(x_data, y_data)

    addressed_points = visualize_aod_positions(fx_map=fx_map, fy_map=fy_map, x_data=x_data, y_data=y_data)

    trajectories = visualize_aod_trajectories(fx_map=fx_map, fy_map=fy_map, x_data=x_data, y_data=y_data, stride=150_000)

    visualize_baseband_spectrogram(t_x, x_data, center_freq=200.0, title="X AOD distorted baseband spectrogram", nfft=512, hop=32, max_detuning=1.2)
    visualize_baseband_spectrogram(t_x, y_data, center_freq=200.0, title="Y AOD distorted baseband spectrogram", nfft=512, hop=32, max_detuning=1.2)

    visualize_programmed_spectrum(t_x, x_data, center_freq=200.0, title="X AOD distorted programmed spectrum", sigma=0.01, max_detuning=1.2)
    visualize_programmed_spectrum(t_y, y_data, center_freq=200.0, title="Y AOD distorted programmed spectrum", sigma=0.01, max_detuning=1.2)

    return specs_size, total_data, calc_time

def main_full_sim(x_specs, y_specs, plot=True):
    t0_stamp = time.perf_counter_ns() * 1e-6
    t_x, x_optical, x_data, x_chain = waveform_multitone_full_sim(x_specs)
    t_y, y_optical, y_data, y_chain = waveform_multitone_full_sim(y_specs)
    calc_time = time.perf_counter_ns() * 1e-6 - t0_stamp

    specs_size, total_data = print_memory_report(x_specs, y_specs, x_optical, y_optical, x_data, y_data, verbose=True)
    print_full_chain_report(x_specs, x_chain)
    print_full_chain_report(y_specs, y_chain)

    if plot:
        plot_specs(t_x, x_chain["rf_input"], x_data, t_y, y_chain["rf_input"], y_data)

        # visualize_baseband_spectrogram(t_x, x_data, center_freq=200.0, title="X AOD full-sim baseband spectrogram", nfft=512, hop=32, max_detuning=1.2)
        # visualize_baseband_spectrogram(t_y, y_data, center_freq=200.0, title="Y AOD full-sim baseband spectrogram", nfft=512, hop=32, max_detuning=1.2)

        # visualize_programmed_spectrum(t_x, x_data, center_freq=200.0, title="X AOD full-sim programmed spectrum", sigma=0.01, max_detuning=1.2)
        # visualize_programmed_spectrum(t_y, y_data, center_freq=200.0, title="Y AOD full-sim programmed spectrum", sigma=0.01, max_detuning=1.2)

        visualize_full_chain(t_x, x_chain, title="X AOD full RF/acoustic/optical chain", stride=1)
        visualize_full_chain(t_y, y_chain, title="Y AOD full RF/acoustic/optical chain", stride=1)

        visualize_intermodulation_spectrum(x_chain, x_specs["sampling"], center_freq=200.0, title="X AOD intermodulation spectrum", nfft=2**18, max_detuning=5.0)
        visualize_intermodulation_spectrum(y_chain, y_specs["sampling"], center_freq=200.0, title="Y AOD intermodulation spectrum", nfft=2**18, max_detuning=5.0)

        # visualize_optical_effect(t_x, x_data, y_data, time_stride=max(1, len(t_x) // 200), wavelength_nm=830.0, acoustic_velocity_mm_per_us=4.2)

    return specs_size, total_data, calc_time, x_chain, y_chain




### EXECUTION ###

if __name__ == '__main__':

    manual_mode = False
    full_sim_mode = True
    benchmark_mode = False

    simulation_duration = 1.1
    pulse_duration = 1.0
    sampling = int(1000e6)
    sweep_span = 0.10
    scan_variable = np.arange(2, 3, 1)
    memory_size = [[], [], []]

    response = {
        "amplitude": {"type": "exp", "tau": 13, "n": 81},
        "frequency": {"type": "gaussian", "sigma": 11, "n": 71},
        "phase": {"type": "gaussian", "sigma": 11, "n": 71},
        "rf_chain": {"type": "gaussian", "sigma": 4, "n": 31},
        "acoustic": {"type": "gaussian", "sigma": 11, "n": 71},
        "nonlinear": {"compression": 0.02, "quadratic": 0.002, "cubic": 0.01, "clip": None},
        "optical": {"efficiency": 0.70, "saturation": 0.05, "contrast_floor": 0.0, "cubic": 0.0},
    }

    for value in tqdm(scan_variable):

        if manual_mode:
            x_specs = {
                "axis": "X",
                "duration": simulation_duration,
                "sampling": sampling,
                "response": response,
                "tones": [
                    {"name": "x_tone_0", "space_coord": 0, "t0": 0.0, "tone_duration": pulse_duration, "amplitude": [0.5, 0.0, 0.0, 0.0, 0.0], "frequency": [200.027, sweep_span / pulse_duration, 0.0, 0.0, 0.0], "phase": [np.pi / 4, 0.0, 0.0, 0.0, 0.0]},
                    {"name": "x_tone_1", "space_coord": 1, "t0": 0.0, "tone_duration": pulse_duration, "amplitude": [0.5, 0.0, 0.0, 0.0, 0.0], "frequency": [200.131, sweep_span / pulse_duration, 0.0, 0.0, 0.0], "phase": [np.pi / 4, 0.0, 0.0, 0.0, 0.0]},
                ],
            }

            y_specs = {
                "axis": "Y",
                "duration": simulation_duration,
                "sampling": sampling,
                "response": response,
                "tones": [
                    {"name": "y_tone_0", "space_coord": 0, "t0": 0.0, "tone_duration": pulse_duration, "amplitude": [0.4, 0.0, 0.0, 0.0, 0.0], "frequency": [200.025, sweep_span / pulse_duration, 0.0, 0.0, 0.0], "phase": [np.pi / 3, 0.0, 0.0, 0.0, 0.0]},
                    {"name": "y_tone_1", "space_coord": 1, "t0": 0.0, "tone_duration": pulse_duration, "amplitude": [0.4, 0.0, 0.0, 0.0, 0.0], "frequency": [200.131, sweep_span / pulse_duration, 0.0, 0.0, 0.0], "phase": [np.pi / 3, 0.0, 0.0, 0.0, 0.0]},
                ],
            }

        else:
            x_specs = generate_large_aod_specs(axis="X", n_tones=value, duration=simulation_duration, sampling=sampling, f_min=200.027, f_max=200.131, pulse_duration=pulse_duration, t0=0.0, sweep_span=sweep_span, amplitude=0.5, phase=np.pi / 4, response=None, seed=1)
            y_specs = generate_large_aod_specs(axis="Y", n_tones=value, duration=simulation_duration, sampling=sampling, f_min=200.025, f_max=200.131, pulse_duration=pulse_duration, t0=0.0, sweep_span=sweep_span, amplitude=0.4, phase=np.pi / 3, response=None, seed=2)
            x_specs["response"] = response
            y_specs["response"] = response
            print_specs(x_specs)
            print_specs(y_specs)

        if full_sim_mode:
            mem_spec_size, mem_total_size, calc_time, x_chain, y_chain = main_full_sim(x_specs, y_specs, plot=True)
        else:
            mem_spec_size, mem_total_size, calc_time = main_distorsion(x_specs, y_specs)

        memory_size[0].append(mem_spec_size)
        memory_size[1].append(mem_total_size)
        memory_size[2].append(calc_time)

    if benchmark_mode:
        fig, ax1 = plt.subplots()
        ax1.plot(scan_variable, memory_size[0], color='dodgerblue', label='Specs memory')
        ax1.plot(scan_variable, memory_size[1], color='teal', label='Total memory')
        ax1.set_xlabel("# tones")
        ax1.set_ylabel("Memory", color='teal')
        ax1.tick_params(axis='y', labelcolor='teal')

        ax2 = ax1.twinx()
        ax2.plot(scan_variable, memory_size[2], color='red', label='Time calculation')
        ax2.set_ylabel("Time [ms]", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        ax1.set_xscale("log")

        plt.tight_layout()
        plt.show()
