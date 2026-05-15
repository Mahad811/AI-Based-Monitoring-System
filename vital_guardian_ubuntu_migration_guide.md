# Vital Guardian: Native Ubuntu Migration Guide

This document contains everything you (and the new Antigravity AI instance) need to successfully migrate the Vital Guardian project from Windows/WSL to a native, high-performance Ubuntu environment. 

---

## Part 1: Ubuntu Dual-Boot Guide

Running Ubuntu natively will give TensorFlow and PyTorch direct, unrestricted access to your RTX 4050 GPU, eliminating memory crashes and maximizing FPS.

### Requirements
1. **USB Flash Drive**: At least 8GB (will be erased).
2. **Free Storage**: At least 100GB of free space on your Windows `C:` drive.
3. **OS Version**: **Ubuntu 22.04 LTS** (Highly recommended over 24.04 for the most stable NVIDIA CUDA and TensorFlow compatibility).

### Step-by-Step Installation

#### 1. Download & Prepare the USB
1. **Download the OS**: Download the [Ubuntu 22.04 LTS Desktop ISO](https://releases.ubuntu.com/jammy/). (Wait for the ~3GB file to finish downloading).
2. **Create Bootable USB**: 
   - Download [Rufus](https://rufus.ie/).
   - Plug in your USB drive (Warning: All data on the USB will be erased).
   - In Rufus, select your USB drive under "Device".
   - Click "SELECT" and pick the Ubuntu 22.04 ISO file.
   - Leave the Partition Scheme as **GPT** and Target System as **UEFI (non CSM)**.
   - Click "Start". If prompted, choose "Write in ISO Image mode".

#### 2. Windows Disk Allocation
We need to carve out space for Ubuntu without deleting Windows.
1. Right-click the Windows Start Menu and select **Disk Management**.
2. Locate your `C:` drive (or whichever drive has lots of free space).
3. Right-click it and select **Shrink Volume**.
4. In the "Enter the amount of space to shrink in MB" box, enter your desired allocation:
   - For 100GB: enter `102400`
   - For 150GB: enter `153600` (Recommended if you plan to download large AI datasets later).
5. Click **Shrink**. You will now see a black block labeled "Unallocated". Do NOT format this; leave it exactly as it is.

#### 3. BIOS / UEFI Prep
1. Restart your laptop and immediately start tapping the BIOS key (usually `F2`, `F12`, `F10`, or `DEL` depending on your laptop manufacturer).
2. Navigate to the "Security" or "Boot" tab and find **Secure Boot**. Change it to **Disabled**. *(This is completely necessary because proprietary NVIDIA drivers cannot be loaded into the Linux kernel if Secure Boot is enforcing strict signature checks).*
3. Navigate to the Boot Order menu and move your "USB Flash Drive" to the very top (Option 1).
4. Save Changes and Exit.

#### 4. The Ubuntu Installer Setup
Your laptop will reboot into the Ubuntu USB. Select "Try or Install Ubuntu".
1. **Language & Keyboard**: Select English.
2. **Wireless**: Connect to your Wi-Fi network immediately. This is required so the installer can download the latest graphics drivers during setup.
3. **Updates and Other Software**:
   - **What apps would you like to install?** Select **Normal Installation** (this gives you web browsers, media players, and basic utilities).
   - **Other Options**: 
     - Check ✅ **"Download updates while installing Ubuntu"**.
     - Check ✅ **"Install third-party software for graphics and Wi-Fi hardware and additional media formats"**. *(CRITICAL: This ensures your laptop's Wi-Fi card, Bluetooth, and audio codecs work out of the box).*
4. **Installation Type (Partitioning)**:
   - You will see a screen asking how you want to install. Select **"Install Ubuntu alongside Windows Boot Manager"**.
   - *Why this option?* This is the safest, automated option. It will automatically find the 100GB/150GB "Unallocated" space you created earlier, format it as `EXT4` (the Linux file system), and install Ubuntu there. It automatically creates a swapfile and leaves your Windows `C:` drive completely untouched.
   - *(Note: If you do not see the "alongside Windows" option, choose "Something Else", click on the "Free Space", click the `+` icon, allocate all of it as `Ext4 journaling file system`, and set the Mount Point to `/`).*
5. **Finalizing**:
   - Choose your timezone.
   - Create your username and password (you will type this password frequently when using `sudo` commands).
   - Click Install and wait ~15 minutes.
   
#### 5. Rebooting
Once finished, the installer will ask you to restart. Pull the USB drive out when the screen tells you to. When the laptop turns back on, you will see a black screen with white text called the **GRUB Menu**. Here, you can use your arrow keys to select "Ubuntu" or "Windows Boot Manager". Select Ubuntu!

---

## Part 2: Vital Guardian Setup Guide (For User & Antigravity)

> [!NOTE] 
> **CONTEXT FOR NEW ANTIGRAVITY INSTANCE**: 
> You are picking up work on "Vital Guardian", an edge-AI patient monitoring system for hospitals. The system uses a FastAPI backend with WebSockets to stream inference data to a web dashboard. 
> 
> **CRITICAL MISSION**: The user will be cloning their *original* GitHub repository. That repository is currently configured for a remote Kaggle endpoint or extremely slow OpenVINO CPU processing. **Your objective is to patch the code to achieve real-time, native GPU inference on their local NVIDIA RTX 4050.**
> 
> **Architectural Breakdown & Required Code Patches:**
> 1. **Vision Pipeline (YOLOv11 + MoViNet):**
>    - Uses YOLOv11 for person detection, and two 3D MoViNet CNNs for Fall and Seizure classification.
>    - **PATCH REQUIRED (`config/config.yaml`)**: The raw repo tries to use `yolo11n_openvino_model`. You MUST change it to `model: yolo11n.pt` and set `device: "0"` to force native PyTorch CUDA execution.
>    - **PATCH REQUIRED (`scripts/demo/demo_server.py`)**: To fix severe lagging, find where `process_every` is set for the `person_detector` and change it to `3`. This caches the bounding box and skips 66% of compute, tripling FPS.
> 2. **GPU Memory Clashing (The Crash Fix):**
>    - Both TensorFlow (MoViNet) and PyTorch (YOLO) need to exist on the same GPU. TensorFlow greedily grabs 100% of the VRAM on startup, which causes PyTorch to crash the entire server with a `Floating point exception` when the video loads.
>    - **PATCH REQUIRED (`scripts/demo/demo_server.py`)**: You MUST inject `os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'` immediately below the first `import os` statement.
>    - **PATCH REQUIRED**: In `demo_server.py`, comment out `tf.config.threading.set_inter_op_parallelism_threads` and `intra_op_parallelism` so the CPU isn't artificially bottlenecked.
> 3. **Audio Pipeline Bug (libsndfile crash):**
>    - Uses YAMNet and `faster-whisper` for distress detection.
>    - **PATCH REQUIRED (`scripts/demo/demo_server.py`)**: The raw repo uses `audio_data, _ = librosa.load(wav_path, sr=16000)`. In Linux, `libsndfile` occasionally triggers a fatal Floating Point Exception during resampling under heavy GPU load. You MUST completely rewrite the `librosa.load` block to use `scipy.io.wavfile.read` instead. Convert the output to mono, cast to `np.float32`, and normalize it to `[-1.0, 1.0]` by dividing by `32768.0` if necessary.
> 4. **Warnings Spam:**
>    - **PATCH REQUIRED (`scripts/demo/demo_server.py`)**: Inject `import absl.logging; absl.logging.set_verbosity(absl.logging.ERROR)` at the top to kill the unsaved custom gradient warnings.

### Step 1: Install System & GPU Drivers
Once logged into Ubuntu, open the terminal (`Ctrl+Alt+T`) and run:
```bash
sudo apt update && sudo apt upgrade -y
# Install essential system libraries for OpenCV and Audio (librosa/sounddevice)
sudo apt install -y python3-venv python3-pip python3-dev build-essential ffmpeg libsm6 libxext6 libasound2-dev portaudio19-dev
```
**NVIDIA Drivers**:
1. Open the "Software & Updates" app from the Ubuntu app menu.
2. Go to the "Additional Drivers" tab.
3. Select the latest tested proprietary NVIDIA driver (e.g., `nvidia-driver-535` or `550`) and click "Apply Changes".
4. **Reboot your laptop.**
5. Verify the GPU is detected by running `nvidia-smi` in the terminal.

### Step 2: Clone the Project
```bash
# Clone the repository from GitHub
git clone https://github.com/YOUR_GITHUB_USERNAME/AI-Based-Monitoring-System.git
cd AI-Based-Monitoring-System
```

### Step 3: Create Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install ML Dependencies (GPU Enabled)
Because Ubuntu natively handles GPU drivers, we can use standard pip commands. Make sure you are inside the activated `venv`.

```bash
# 1. Install PyTorch with CUDA 12.1 (for YOLO and Whisper)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install TensorFlow with CUDA support (for MoViNet and YAMNet)
pip install "tensorflow[and-cuda]>=2.14.0"

# 3. Install remaining project requirements
pip install -r requirements.txt
```

### Step 5: Configure the Application
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and configure the following:
   ```env
   INFERENCE_MODE=LOCAL
   GEMINI_API_KEY=AIzaSyAXYc78Z2xx6wrpXEvT8115SjQVUOYVmy8
   DATABASE_URL=sqlite:///./vital_guardian.db
   
   # Because you are now on native Ubuntu, the real microphone WILL work!
   MIC_ENABLED=true
   AUDIO_ANALYTICS_ENABLED=true
   ```
3. Open `config/config.yaml` and ensure YOLO uses the GPU:
   ```yaml
   person_detector:
     model: yolo11n.pt
     device: "0"  # This ensures PyTorch uses the RTX 4050
   ```
4. Open `scripts/demo/demo_server.py` and ensure the GPU memory fix is present near the top imports:
   ```python
   import os
   os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
   # Optionally suppress warnings here
   ```

### Step 6: Run the Server
With the `venv` activated, launch the server:
```bash
python3 scripts/demo/demo_server.py
```
Open a second terminal and monitor the GPU usage to ensure models are loaded properly:
```bash
watch -n 1 nvidia-smi
```

You are now running Vital Guardian on a native, bare-metal Linux environment with maximum possible AI performance!
