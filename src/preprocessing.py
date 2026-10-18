import mne
from mne.io import Raw


def bandpass_filter(data: Raw, fmin: float = 0.1, fmax: float = 75.0) -> Raw:
    ret = data.copy().filter(l_freq=fmin, h_freq=fmax, fir_design="firwin")
    return ret


def notch_filter(data: Raw, notch_freqs: float = 50.0) -> Raw:
    """
    Notch filter for powerline noise (50 Hz and/or 60 Hz)
    """
    raw_notched = data.copy().notch_filter(
        freqs=notch_freqs, picks="eeg", fir_design="firwin", verbose=False
    )
    return raw_notched


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
        eog_inds, _ = ica.find_bads_eog(
            raw_eeg, ch_name=proxy_eog, threshold=3.0
        )
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


def preprocess(data: Raw) -> Raw:
    data = bandpass_filter(data, fmin=0.1, fmax=75.0)
    data = notch_filter(data, notch_freqs=50.0)
    data = smoothing(data, smooth_lp=35.0)
    data = ica(data)
    return data
