import os
import pandas as pd
import numpy as np

class DataTypeDetector:
    def init(self, data):
        self.data = data

    def detect(self):
        if isinstance(self.data, pd.DataFrame):
            return "dataframe"

        elif isinstance(self.data, list):
            if all(isinstance(d, dict) and "content" in d for d in self.data):
                return "preprocessed"

            else:
                return "list"

        elif isinstance(self.data, np.ndarray):
            return "ndarray"

        elif isinstance(self.data, str):
            if os.path.isdir(self.data):
                return "folder"

            elif os.path.isfile(self.data):
                ext = os.path.splitext(self.data)[1].lower()
                if ext in [".mp4", ".avi", ".mov"]:
                    return "video_file"

                elif ext in [".wav", ".mp3", ".flac"]:
                    return "audio_file"

                elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    return "image_file"

                elif ext in [".csv", ".xlsx"]:
                    return "table_file"

                else:
                    return "file"

            else:
                return "raw_text"

        else:
            return 'unsupported_data_type'