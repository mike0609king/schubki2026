from mne.io import Raw


# TODO check default values
# TODO We might need to compute nequist frequency here or outside for cutoff
def bandpass_filter(data: Raw, fmin: float = 0.1, fmax: float = 50.0):
    ret = data.copy().filter(l_freq=fmin, h_freq=fmax, fir_design="firwin")
    return ret


# TODO check whether notch should be 50 or 60
def notch_filter(data: Raw, notch_freqs: float) -> Raw:
    """
    Notch filter for powerline noise (50 Hz and/or 60 Hz)
    """
    raw_notched = data.copy().notch_filter(
        freqs=notch_freqs, picks="eeg", fir_design="firwin", verbose=False
    )
    return raw_notched


# TODO implement a necessary smoothing filter
def smoothing(data: Raw, smooth_lp: float = 35.0) -> Raw:
    raw_smooth = data.copy().filter(
        l_freq=None, h_freq=smooth_lp, fir_design="firwin", verbose=False
    )
    return raw_smooth


# TODO Perhaps ICA needs some sort of playgroud to review the results. In this
# case we need a jupyter notebook with some useful visualizations pre-implemented.
def ica(data: Raw):
    raw_eeg = raw_smooth.copy().pick("eeg")
    # TODO fastica vs infomax vs picard
    ica = mne.preprocessing.ICA(
        n_components=25, method="fastica", random_state=42, max_iter="auto"
    )
    ica.fit(raw_eeg)
    print(f"ICA fitted with {ica.n_components_} components.")

    # Try automatic detection of EOG-like components
    # Since there are no formal EOG channels, we use a frontal EEG channel ('Fp1') as a proxy
    # TODO during competition we need to make sure this channel exists
    proxy_eog = "Fp1" if "Fp1" in raw_eeg.ch_names else None
    if proxy_eog:
        print(f"Using {proxy_eog} as a proxy channel for EOG artifact detection...")
        eog_inds, eog_scores = ica.find_bads_eog(
            raw_eeg, ch_name=proxy_eog, threshold=3.0
        )
    else:
        eog_inds, eog_scores = [], []
    print(f"EOG-like components (auto): {eog_inds}")

    # Muscle-like components
    muscle_inds, muscle_scores = ica.find_bads_muscle(raw_eeg)
    print(f"Muscle-like components (auto): {muscle_inds}")

    # Combine and exclude
    exclude = list(set(eog_inds + list(muscle_inds)))
    print(f"Excluding components: {exclude}")
    ica.exclude = exclude

    raw_clean = raw_eeg.copy()
    ica.apply(raw_clean)
    return raw_clean
