import json
import os
import subprocess
from pathlib import Path
from typing import Tuple

def get_file_and_folder() -> Tuple[str, str]:
    folder = input("Please select a folder: ")
    files = os.listdir(folder)
    file_path = os.path.join(folder, files[0])
    return file_path, folder

def check_file_type(file_path: str) -> str:
    file_type = os.path.splitext(file_path)[1]
    return file_type

def play_audio(file_path: str):
    subprocess.run(["afplay", file_path])
    num = input("Label this audio: ")
    return num

def show_image(file_path: str):
    from PIL import Image
    img = Image.open(file_path)
    img.show()
    num = input("Label this image: ")
    return num

def save_data(file_path: str, folder: str, num: str):
    file_name = os.path.basename(file_path)
    data = {file_name: num}
    with open(os.path.join(folder, "index.json"), "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    file_path, folder = get_file_and_folder()
    file_type = check_file_type(file_path)
    if file_type == ".wav":
        num = play_audio(file_path)
    elif file_type in [".jpg", ".png", ".jpeg", ".gif"]:
        num = show_image(file_path)
    save_data(file_path, folder, num)