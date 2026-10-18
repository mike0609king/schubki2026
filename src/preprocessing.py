import mne
from mne.io import Raw
from scipy import signal


def bandpass(x, sfreq: float, l_freq: float, h_freq: float, order=4):
    sos = signal.butter(
        order, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos"
    )
    return signal.sosfiltfilt(sos, x, axis=-1)


def notch(x, sfreq: float, freq: float = 50.0, q: float = 30.0):
    b, a = signal.iirnotch(freq, q, fs=sfreq)
    return signal.filtfilt(b, a, x, axis=-1)


def smoothing(data: Raw, smooth_lp: float = 35.0) -> Raw:
    raw_smooth = data.copy().filter(
        l_freq=None, h_freq=smooth_lp, fir_design="firwin", verbose=False
    )
    return raw_smooth


def ica(data: Raw):
    raw_eeg = data.copy().pick("eeg")
    ica = mne.preprocessing.ICA(
        n_components=25, method="fastica", random_state=42, max_iter="auto"
    )
    ica.fit(raw_eeg)
    print(f"ICA fitted with {ica.n_components_} components.")

    # Try automatic detection of EOG-like components
    # Since there are no formal EOG channels, we use a frontal EEG channel ('Fp1') as a proxy
    proxy_eog = "Fp1" if "Fp1" in raw_eeg.ch_names else None
    if proxy_eog:
        print(f"Using {proxy_eog} as a proxy channel for EOG artifact detection...")
        eog_inds, _ = ica.find_bads_eog(raw_eeg, ch_name=proxy_eog, threshold=3.0)
    else:
        eog_inds, _ = [], []
    print(f"EOG-like components (auto): {eog_inds}")

    # Muscle-like components
    muscle_inds, _ = ica.find_bads_muscle(raw_eeg)
    print(f"Muscle-like components (auto): {muscle_inds}")

    # Combine and exclude
    exclude = list(set(eog_inds + list(muscle_inds)))
    print(f"Excluding components: {exclude}")
    ica.exclude = exclude

    raw_clean = raw_eeg.copy()
    ica.apply(raw_clean)
    return raw_clean
