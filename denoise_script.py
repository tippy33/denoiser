import subprocess
from pathlib import Path

TARGET_DIR = "/Volumes/One Touch/ASD_Dataset/all-audios-copy"
SAVE_DIR = "/Volumes/One Touch/ASD_Dataset/all-audios-denoised"

def run_denoiser():
  target_dir = Path(TARGET_DIR)
  for i, file in enumerate(target_dir.iterdir()):
    file_size = file.stat().st_size
    if file.suffix == ".wav" and file_size <= 30 * 1024:
      # if str(file).endswith("p18-s13.wav") or str(file).endswith("p18-s17.wav") or str(file).endswith("p18-s18.wav") or str(file).endswith("p18-s19.wav") or str(file).endswith("p18-s20.wav"):
      #   pass
      # else:
      print(f"Executing: {file}")
      command = ["python", "-m", "denoiser.enhance", "--dns64", "--noisy_file", str(file), "--out_dir", str(SAVE_DIR)]
      try:
        subprocess.run(command, check=True)
      except subprocess.CalledProcessError as e:
        print(f"Error occurred while processing {file}: {e}")
    # if i == 0:
    #   break
  return 

if __name__ == "__main__":
  run_denoiser()