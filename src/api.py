from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks, resample


def _get_checkpoint_path(checkpoint_path: str) -> str:
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    if os.path.exists(f"{checkpoint_path}.pth"):
        return f"{checkpoint_path}.pth"
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

        self.swish = Swish()

    def forward(self, x):
        identity = x
        out = self.swish(self.bn1(self.conv1(x)))
        out = self.swish(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(identity)
        return self.swish(out)


class ECGResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.block1 = ResBlock(1, 32)
        self.block2 = ResBlock(32, 64)
        self.block3 = ResBlock(64, 128)

        self.global_max = nn.AdaptiveMaxPool1d(1)
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        max_p = self.global_max(x).squeeze(-1)
        avg_p = self.global_avg(x).squeeze(-1)
        feat = torch.cat([max_p, avg_p], dim=1)
        return self.softmax(self.fc(feat))


def preprocess_ecg_window(ecg, fs=360, global_size=450, new_fs=120):
    ecg_down = resample(ecg, int(global_size * new_fs / fs))
    return (ecg_down - np.mean(ecg_down)) / (np.std(ecg_down) + 1e-8)


def ecg_to_beats(ecg_raw, fs=360, global_size=450, new_fs=120):
    peaks, _ = find_peaks(ecg_raw, distance=int(0.25 * fs))
    if len(peaks) < 2:
        return np.array([])

    rr = np.diff(peaks)
    hb_size = int(np.mean(rr))
    beats = []

    for p in peaks:
        start = p - hb_size // 2
        end = p + hb_size // 2
        if start < 0 or end > len(ecg_raw):
            continue

        hb = ecg_raw[start:end]
        hb = np.pad(hb, (0, max(0, global_size - len(hb))), "constant")[:global_size]
        beats.append(preprocess_ecg_window(hb, fs, global_size, new_fs))

    return np.array(beats)


class ECGRequest(BaseModel):
    ecg_signal: List[float]
    sampling_rate: Optional[int] = 360


class ECGResponse(BaseModel):
    predictions: List[int]
    majority_vote: int
    mean_probabilities: List[float]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = _get_checkpoint_path('checkpoints/best_model.pth')
MODEL = ECGResNet(num_classes=5).to(DEVICE)
MODEL.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
MODEL.eval()

app = FastAPI(title="ECG Heartbeat Classification API")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "checkpoint": os.path.basename(CHECKPOINT_PATH)}


@app.post("/predict", response_model=ECGResponse)
def predict(request: ECGRequest):
    signal = np.asarray(request.ecg_signal, dtype=np.float32)
    if signal.ndim != 1 or signal.size < 20:
        raise HTTPException(status_code=400, detail="Input ECG signal must be a 1D list with at least 20 samples.")

    beats = ecg_to_beats(signal, fs=request.sampling_rate)
    if beats.size == 0:
        raise HTTPException(status_code=400, detail="Cannot detect any heartbeat segments from the input ECG.")

    x_tensor = torch.tensor(beats, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(x_tensor)

    probs = outputs.cpu().numpy()
    preds = probs.argmax(axis=1).tolist()
    majority_vote = int(np.bincount(preds).argmax())
    mean_prob = probs.mean(axis=0).tolist()

    return ECGResponse(predictions=preds, majority_vote=majority_vote, mean_probabilities=mean_prob)
