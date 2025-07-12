import os
import pandas as pd
import numpy as np
import tempfile
from detector import DataTypeDetector
def test_all_cases():
    # 📌 1. DataFrame
    df = pd.DataFrame({"a": [1, 2]})
    assert DataTypeDetector(df).detect() == "dataframe"

    # 📌 2. Preprocessed List
    preprocessed = [{"content": "hello"}, {"content": "world"}]
    assert DataTypeDetector(preprocessed).detect() == "preprocessed"

    # 📌 3. List (نه preprocessed)
    normal_list = ["hello", 123]
    assert DataTypeDetector(normal_list).detect() == "list"

    # 📌 4. ndarray
    arr = np.array([1, 2, 3])
    assert DataTypeDetector(arr).detect() == "ndarray"

    # 📌 5. folder path
    with tempfile.TemporaryDirectory() as folder:
        assert DataTypeDetector(folder).detect() == "folder"

    # 📌 6. video file
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        assert DataTypeDetector(f.name).detect() == "video_file"

    # 📌 7. audio file
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        assert DataTypeDetector(f.name).detect() == "audio_file"

    # 📌 8. image file
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        assert DataTypeDetector(f.name).detect() == "image_file"

    # 📌 9. table file
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        assert DataTypeDetector(f.name).detect() == "table_file"

    # 📌 10. unknown file
    with tempfile.NamedTemporaryFile(suffix=".abc") as f:
        assert DataTypeDetector(f.name).detect() == "file"

    # 📌 11. raw text (مثل آدرس فایل که وجود نداره)
    assert DataTypeDetector("this is some raw string").detect() == "raw_text"

    # 📌 12. Unsupported
    assert DataTypeDetector(12345).detect() == "unsupported_data_type"

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_all_cases()
