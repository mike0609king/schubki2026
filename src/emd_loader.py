from pathlib import Path

import numpy as np

import h5py
import cv2

class EmdDataset:
    def __init__(self, emd_dir: Path):
        self._emd_dir = emd_dir

    def get_num_videos(self):
        return 1102

    def get_video_by_id(self, video_id: int):
        video_dir = self._emd_dir / "stimuli" / "stimuli"
        file = next(Path(video_dir).rglob(f"{video_id:04d}*"), None)

        cap = cv2.VideoCapture(str(file))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret = np.empty((frame_count, height, width, 3), np.dtype('uint8'))

        for i in range(frame_count):
            has_next, ret[i] = cap.read()
            if not has_next:
                break
        cap.release()
        return ret

    def get_data_by_ids(self, sub_id: int, ses_id: int):
        assert 1 <= sub_id <= 6
        assert 1 <= ses_id <= 8

        data_dir = self._emd_dir / "derivatives" / "eeg" / f"sub-{sub_id:02d}"
        file_name = Path(f"sub-{sub_id:02}_ses-{ses_id:02}_preprocessed_eeg.h5")
        eeg = h5py.File(str(data_dir / file_name), "r")["eeg"][:]
        return eeg
