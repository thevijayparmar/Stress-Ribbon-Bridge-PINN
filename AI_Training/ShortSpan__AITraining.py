import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import csv
import json
import gc
import psutil
import time
from tqdm import tqdm
import keyboard

# =====================================================================
# HARDWARE & DIRECTORY SETUP (STRICT ENFORCEMENT)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("🔍 Initializing Hardware Check...")

if not torch.cuda.is_available():
    print("\n🚨 CRITICAL ERROR: NVIDIA GPU NOT DETECTED BY PYTORCH! 🚨")
    print("PyTorch is attempting to use the CPU. The process has been forcefully aborted.")
    print("Fix: Run this command to install the CUDA version of PyTorch:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    input("\nPress ENTER to exit...")
    sys.exit(1)

device = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
print(f"✅ GPU Confirmed: {gpu_name}")

torch.cuda.set_per_process_memory_fraction(0.8, device=0)
print("✅ GPU Memory Capped at 80% to maintain system stability.")

try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except ImportError:
    print("⚠️ pynvml failed to load. Live GPU stats will be disabled.")
    sys.exit(1)

# =====================================================================
# INTERACTIVE CONTROLS SETUP (PAUSE / RESUME ONLY)
# =====================================================================
is_paused = False
stop_requested = False 

def toggle_pause(e):
    global is_paused
    is_paused = not is_paused
    state = "PAUSED" if is_paused else "RESUMED"
    print(f"\n⏸️ [SYSTEM {state}] - Press 'P' to toggle.\n")

keyboard.on_press_key('p', toggle_pause)

print("⌨️ Interactive Controls Active: Press 'P' to Pause/Resume.")
time.sleep(2)

# =====================================================================
# PHASE 1 & 2: MEMORY-OPTIMIZED DATA LOADING
# =====================================================================
print("\n⏳ Scanning local folder for multi-part CSV datasets...")

train_files = sorted(glob.glob(os.path.join(BASE_DIR, '#01_Generator_16.41M_V7*.csv')))
val_files = sorted(glob.glob(os.path.join(BASE_DIR, '#02_Generator_Validation_0.46M_V7*.csv')))

if not train_files or not val_files:
    print(f"\n🚨 CRITICAL ERROR: Could not find CSV files in:\n{BASE_DIR}")
    input("\nPress ENTER to exit...")
    sys.exit(1)

print(f"📥 Loading {len(train_files)} Training & {len(val_files)} Validation files...")

def load_and_compress(file_list):
    df_list = []
    for f in tqdm(file_list, desc="Loading CSVs", leave=False):
        df_chunk = pd.read_csv(f)
        if 'Out_Geometry_Possible' in df_chunk.columns:
            df_chunk = df_chunk[df_chunk['Out_Geometry_Possible'] == 1]
        float_cols = df_chunk.select_dtypes(include=['float64']).columns
        df_chunk[float_cols] = df_chunk[float_cols].astype('float32')
        df_list.append(df_chunk)
    combined = pd.concat(df_list, ignore_index=True)
    del df_list
    gc.collect()
    return combined

df_train_valid = load_and_compress(train_files)
df_val_valid = load_and_compress(val_files)

print(f"✅ Data Loaded: {len(df_train_valid):,} Train rows | {len(df_val_valid):,} Val rows.")

inputs = ['In_Span_m', 'In_HeightDiff_m', 'In_UDL_kNm', 'In_Tallow_kN', 'In_RecSag_m']
targets = ['Out_Param_a_m', 'Out_H_kN']

X_train_raw, Y_train_raw = df_train_valid[inputs].values, df_train_valid[targets].values
X_val_raw, Y_val_raw = df_val_valid[inputs].values, df_val_valid[targets].values

del df_train_valid, df_val_valid
gc.collect()

# =====================================================================
# PHASE 3: SCALING & DICTIONARY EXPORT
# =====================================================================
print("\n🧮 Generating Translation Dictionary & Scaling Data...")
scaler_X, scaler_Y = MinMaxScaler(), MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
Y_train_scaled = scaler_Y.fit_transform(Y_train_raw)
X_val_scaled = scaler_X.transform(X_val_raw)
Y_val_scaled = scaler_Y.transform(Y_val_raw)

del X_train_raw, Y_train_raw, X_val_raw, Y_val_raw
gc.collect()

scale_bounds = {
    "Inputs": {col: {"min": float(scaler_X.data_min_[i]), "max": float(scaler_X.data_max_[i])} for i, col in enumerate(inputs)},
    "Outputs": {col: {"min": float(scaler_Y.data_min_[i]), "max": float(scaler_Y.data_max_[i])} for i, col in enumerate(targets)}
}
json_path = os.path.join(BASE_DIR, 'Translation_Dictionary_Short.json')
with open(json_path, 'w') as f:
    json.dump(scale_bounds, f, indent=4)
print(f"💾 SUCCESS: Saved 'Translation_Dictionary_Short.json' to {BASE_DIR}")

batch_size = 2048
dataset_train = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(Y_train_scaled, dtype=torch.float32))
dataset_val = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(Y_val_scaled, dtype=torch.float32))

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

del X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled
gc.collect()

# =====================================================================
# PHASE 4: NEURAL NETWORK ARCHITECTURES
# =====================================================================
class BridgeSurrogate(nn.Module):
    def __init__(self):
        super(BridgeSurrogate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

class BridgePINN(nn.Module):
    def __init__(self):
        super(BridgePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.Mish(),
            nn.Linear(128, 256), nn.Mish(),
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.net(x)

surrogate_model = BridgeSurrogate().to(device)
pinn_model = BridgePINN().to(device)

optimizer_surr = optim.Adam(surrogate_model.parameters(), lr=0.001)
optimizer_pinn = optim.Adam(pinn_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

w_min = torch.tensor(scaler_X.data_min_[2], dtype=torch.float32).to(device)
w_range = torch.tensor(scaler_X.data_range_[2], dtype=torch.float32).to(device)
a_min = torch.tensor(scaler_Y.data_min_[0], dtype=torch.float32).to(device)
a_range = torch.tensor(scaler_Y.data_range_[0], dtype=torch.float32).to(device)

# --- THE FIX IS HERE ---
H_min = torch.tensor(scaler_Y.data_min_[1], dtype=torch.float32).to(device)
H_range = torch.tensor(scaler_Y.data_range_[1], dtype=torch.float32).to(device)
# -----------------------

def physics_loss_fn(Y_pred_scaled, X_scaled):
    w_real = X_scaled[:, 2] * w_range + w_min
    a_real = Y_pred_scaled[:, 0] * a_range + a_min
    
    H_physics_real = w_real * a_real
    H_physics_scaled = (H_physics_real - H_min) / H_range
    H_pred_scaled = Y_pred_scaled[:, 1]
    
    return torch.mean((H_pred_scaled - H_physics_scaled) ** 2)

def get_system_stats():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    vram_used = mem_info.used / (1024**3)
    temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
    return f"CPU:{cpu}% RAM:{ram}% | GPU:{util}% Temp:{temp}°C VRAM:{vram_used:.1f}GB"

# =====================================================================
# PHASE 5: THE TRAINING LOOP (WITH SMART CHECKPOINTING)
# =====================================================================
epochs = 500
history = []
start_time = time.time()

# --- SMART TRACKER SETTINGS ---
threshold_epoch = 400
best_phys_error = float('inf')
best_epoch_saved = 0
# ------------------------------

print("\n🔥 INITIATING DUAL-TRAINING PROTOCOL 🔥\n")

epoch_bar = tqdm(range(epochs), desc="Total Progress", position=0)

for epoch in epoch_bar:
    surrogate_model.train()
    pinn_model.train()
    
    train_loss_surr = 0.0
    train_loss_pinn = 0.0
    
    batch_bar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{epochs}", position=1, leave=False)
    
    for X_batch, Y_batch in batch_bar:
        while is_paused:
            time.sleep(1)

        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        optimizer_surr.zero_grad()
        Y_pred_surr = surrogate_model(X_batch)
        loss_surr = criterion(Y_pred_surr, Y_batch)
        loss_surr.backward()
        optimizer_surr.step()
        train_loss_surr += loss_surr.item()
        
        optimizer_pinn.zero_grad()
        Y_pred_pinn = pinn_model(X_batch)
        data_loss = criterion(Y_pred_pinn, Y_batch)
        phys_loss = physics_loss_fn(Y_pred_pinn, X_batch)
        loss_pinn = data_loss + 0.5 * phys_loss
        loss_pinn.backward()
        optimizer_pinn.step()
        train_loss_pinn += loss_pinn.item()

        batch_bar.set_postfix_str(get_system_stats())

    surrogate_model.eval()
    pinn_model.eval()
    val_loss_surr, val_loss_pinn, physics_error_tracker = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for X_batch, Y_batch in loader_val:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Y_pred_surr = surrogate_model(X_batch)
            Y_pred_pinn = pinn_model(X_batch)
            
            val_loss_surr += criterion(Y_pred_surr, Y_batch).item()
            val_loss_pinn += criterion(Y_pred_pinn, Y_batch).item()
            
            w_real = X_batch[:, 2] * w_range + w_min
            a_real = Y_pred_pinn[:, 0] * a_range + a_min
            H_real = Y_pred_pinn[:, 1] * H_range + H_min
            physics_error_tracker += torch.mean(torch.abs(H_real - (w_real * a_real))).item()

    t_l_s, v_l_s = train_loss_surr / len(loader_train), val_loss_surr / len(loader_val)
    t_l_p, v_l_p = train_loss_pinn / len(loader_train), val_loss_pinn / len(loader_val)
    mean_phys_err = physics_error_tracker / len(loader_val)
    
    # ---------------------------------------------------------
    # NEW SMART TRACKER: Only save best after Threshold Epoch
    # ---------------------------------------------------------
    current_epoch = epoch + 1
    
    if current_epoch >= threshold_epoch:
        if mean_phys_err < best_phys_error:
            best_phys_error = mean_phys_err
            
            if best_epoch_saved > 0:
                old_surr = os.path.join(BASE_DIR, f"BEST{best_epoch_saved}_surrogate.onnx")
                old_pinn = os.path.join(BASE_DIR, f"BEST{best_epoch_saved}_pinn.onnx")
                if os.path.exists(old_surr): os.remove(old_surr)
                if os.path.exists(old_pinn): os.remove(old_pinn)
            
            best_epoch_saved = current_epoch
            
            dummy_input = torch.randn(1, 5).to(device)
            torch.onnx.export(surrogate_model, dummy_input, os.path.join(BASE_DIR, f"BEST{best_epoch_saved}_surrogate.onnx"), export_params=True, opset_version=15, do_constant_folding=True, input_names=['input'], output_names=['output'])
            torch.onnx.export(pinn_model, dummy_input, os.path.join(BASE_DIR, f"BEST{best_epoch_saved}_pinn.onnx"), export_params=True, opset_version=15, do_constant_folding=True, input_names=['input'], output_names=['output'])
            
            epoch_bar.set_postfix_str(f"⭐ BEST Err: ±{best_phys_error:.2f} kN")
        else:
            epoch_bar.set_postfix_str(f"Phys Err: ±{mean_phys_err:.2f} kN")
    else:
        epoch_bar.set_postfix_str(f"Phys Err: ±{mean_phys_err:.2f} kN (Warming up)")
        
    history.append([current_epoch, t_l_s, v_l_s, t_l_p, v_l_p, mean_phys_err])

# =====================================================================
# PHASE 6: EXPORT & CONFIDENCE REPORTING
# =====================================================================
print("\n\n📊 Generating Confidence Matrix for 'a' Parameter...")
surrogate_model.eval()
errors_a, spans = [], []

with torch.no_grad():
    for X_batch, Y_batch in loader_val:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        Y_pred = surrogate_model(X_batch)
        a_real_true = Y_batch[:, 0] * a_range + a_min
        a_real_pred = Y_pred[:, 0] * a_range + a_min
        span_real = X_batch[:, 0] * scaler_X.data_range_[0] + scaler_X.data_min_[0]
        
        errors_a.extend(torch.abs(a_real_true - a_real_pred).cpu().numpy())
        spans.extend(span_real.cpu().numpy())

errors_a, spans = np.array(errors_a), np.array(spans)
mask_short = spans <= 100
mask_med = (spans > 100) & (spans <= 500)
mask_long = spans > 500

mae_short = np.mean(errors_a[mask_short]) if np.any(mask_short) else 0
mae_med = np.mean(errors_a[mask_med]) if np.any(mask_med) else 0
mae_long = np.mean(errors_a[mask_long]) if np.any(mask_long) else 0

txt_path = os.path.join(BASE_DIR, 'Confidence_Metrics_Report.txt')
with open(txt_path, 'w') as f:
    f.write("=== AI CONFIDENCE METRICS (Mean Absolute Error for 'a') ===\n")
    f.write(f"Short Spans (<=100m) : ±{mae_short:.4f} meters\n")
    f.write(f"Medium Spans (100-500m): ±{mae_med:.4f} meters\n")
    f.write(f"Long Spans (>500m)   : ±{mae_long:.4f} meters\n")
print(f"💾 SUCCESS: Saved 'Confidence_Metrics_Report.txt' to {BASE_DIR}")

csv_path = os.path.join(BASE_DIR, 'AI_Learning_History.csv')
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train_Loss_Surr', 'Val_Loss_Surr', 'Train_Loss_PINN', 'Val_Loss_PINN', 'Mean_Physics_Error_kN'])
    writer.writerows(history)
print(f"💾 SUCCESS: Saved 'AI_Learning_History.csv' to {BASE_DIR}")

# 3. NATIVE PYTORCH BACKUPS
torch.save(surrogate_model.state_dict(), os.path.join(BASE_DIR, "500Epochs_surrogate_backup.pth"))
torch.save(pinn_model.state_dict(), os.path.join(BASE_DIR, "500Epochs_pinn_backup.pth"))
print(f"💾 SUCCESS: Saved Native '.pth' backup weights to {BASE_DIR}")

# 4. ONNX EXPORT (The Final Epoch 500 Unconditional Save)
dummy_input = torch.randn(1, 5).to(device)
torch.onnx.export(surrogate_model, dummy_input, os.path.join(BASE_DIR, "500Epochs_surrogate.onnx"), export_params=True, opset_version=15, do_constant_folding=True, input_names=['input'], output_names=['output'])
torch.onnx.export(pinn_model, dummy_input, os.path.join(BASE_DIR, "500Epochs_pinn.onnx"), export_params=True, opset_version=15, do_constant_folding=True, input_names=['input'], output_names=['output'])

print(f"\n💾 SUCCESS: Saved Final '500Epochs' models.")
if best_epoch_saved > 0:
    print(f"⭐ The absolute smartest model was saved as 'BEST{best_epoch_saved}_surrogate/pinn.onnx'")

# =====================================================================
# FINAL TERMINAL HOLD
# =====================================================================
total_time = (time.time() - start_time) / 60
print(f"\n🎉 ALL PROCESSES COMPLETED IN {total_time:.1f} MINUTES! 🎉")
input("\n🟢 Processing Finished. Press ENTER to safely close this window...")