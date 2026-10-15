# Preprocessing

**Output.** 
- output format `(n_trials, n_channels, n_times)`
- unit micro Volt, use `data.get_data("uV")`

**Checklist.**
- Which channels to remove?
- Check ICA result: Alpha rhythms should be visible (look at O channels first), if not change algorithm

**Notes.**
- We do not use smoothing for encoders. We would use smoothing for classical features.
