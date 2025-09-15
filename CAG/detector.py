from typing import Any, List, Dict
import pandas as pd
import numpy as np
import os

class DataTypeDetector:
    """
    Detects the specific data type of a given input to route it for
    appropriate processing in a RAG pipeline.
    """
    def detect(self, data: Any) -> str:
        """
        Analyzes the data and returns a string identifier for its type.
        Args:
            data: The input data to analyze (e.g., filepath, DataFrame, text).
        Returns:
            A string representing the detected data type.
        """
        if isinstance(data, pd.DataFrame):
            return "dataframe"

        elif isinstance(data, list):
            if all(isinstance(d, dict) and "content" in d for d in data):
                return "preprocessed"

            else:
                return "list"

        elif isinstance(data, np.ndarray):
            return "ndarray"

        elif isinstance(data, str):
            if os.path.isdir(data):
                return "folder"

            elif os.path.isfile(data):
                ext = os.path.splitext(data)[1].lower()

                if ext in [".mp4", ".avi", ".mov"]:
                    return "video_file"

                elif ext in [".wav", ".mp3", ".flac"]:
                    return "audio_file"

                elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    return "image_file"

                elif ext in [".csv", ".xlsx", ".xls"]:
                    return "table_file"

                elif ext in [".txt", ".md", ".pdf", ".docx"]:
                    return "document_file"

                else:
                    return "file"

            else:
                return "raw_text"

        else:
            return 'unsupported_data_type'