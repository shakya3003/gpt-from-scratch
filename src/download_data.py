import os
import urllib.request

def download_tinyshakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "tinyshakespeare.txt")

    if not os.path.exists(file_path):
        print(f"Downloading TinyShakespeare dataset to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete!")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_tinyshakespeare()
