"""
================================================================================
  AEROTWIN B787 — UNIFIED MRO LIVE DASHBOARD (V5.0)
  FlightGear UDP -> ODE Physics -> CNN-LSTM -> Ollama LLM -> GUI

  This is a single, unified script. It replaces both api.py and the old
  dashboard. It loads the PyTorch model directly into memory, runs the physics
  upon touchdown, generates the diagnosis, and calls your local Ollama 3.2
  for the MRO explanation.

  Requirements:
    1. PyTorch model at: models/best_model.pt
    2. Scalers at: models/signal_scalers.pkl
    3. Ollama running in background: `ollama serve` (llama3.2 pulled)
    4. FlightGear streaming UDP to 127.0.0.1:16662
================================================================================
"""

import os
import time
import queue
import socket
import pickle
import threading
import requests
import numpy as np
import pandas as pd
import tkinter as tk
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, 'models', 'best_model.pt')
SCALER_PATH   = os.path.join(BASE_DIR, 'models', 'signal_scalers.pkl')
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

UDP_IP        = "127.0.0.1"
UDP_PORT      = 16662
SIM_TIME      = 4.0
N_PTS         = 100
SLUG_TO_KG    = 14.5939
FPS_TO_MS     = 0.3048

OLLAMA_URL    = 'http://localhost:11434'
OLLAMA_MODEL  = 'llama3.2'
OLLAMA_TIMEOUT= 120

RUL_MAX_HOURS = 2000.0

# ─── PHYSICS CONSTANTS ───────────────────────────────────────────────────────
K_NOM = 1500000.0; B_NOM = 60000.0; P0 = 20.0e6
N_POLY = 1.35; STROKE_LIM = 0.635; ARM_RATIO = 0.55
K_CONTACT = 1.0e6; B_CONTACT = 5.0e4; G = 9.81

# ─── MRO DICTIONARIES ────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "NORMAL",            1: "N2 LEAK",          2: "WORN SEAL",
    3: "EARLY DEGRAD",      4: "THERMAL DEGRAD",   5: "BRAKE FADE",
    6: "TIRE BURST",        7: "CORROSION",        8: "HARD LANDING",
    9: "N2 + SEAL",        10: "IMPENDING AOG",   11: "BOGIE PITCH"
}

SEVERITY_COLOR = {1: "#00c853", 2: "#ffd600", 3: "#ff6d00", 4: "#d50000"}
SEVERITY_LABEL = {1: "MONITOR", 2: "SCHEDULE", 3: "PRIORITY", 4: "GROUND A/C"}
DISPATCH_COLOR = {
    "CONTINUE": "#00c853", "MONITOR": "#4fc3f7",
    "SCHEDULE": "#ffd600", "PRIORITY": "#ff6d00",
    "GROUND": "#d50000"
}

AMM_REF = {
    0:  {'ata':'32-00','task':'32-00-00-200-001','dispatch':'CONTINUE',
         'action':'Continue operations. Log landing data per AMM 32-00-00-800-001.'},
    1:  {'ata':'32-31','task':'32-31-00-200-801','dispatch':'SCHEDULE',
         'action':'Inspect N2 strut servicing. Check extension per AMM 32-31-00-200-801.'},
    2:  {'ata':'32-32','task':'32-32-00-200-801','dispatch':'SCHEDULE',
         'action':'Inspect main gear seals per AMM 32-32-00-200-801. Replace packing if weeping.'},
    3:  {'ata':'32-00','task':'32-00-00-200-002','dispatch':'MONITOR',
         'action':'Schedule detailed inspection within 50 cycles. Increase monitoring.'},
    4:  {'ata':'32-61','task':'32-61-00-200-801','dispatch':'SCHEDULE',
         'action':'Inspect actuator thermal condition per AMM 32-61-00-200-801.'},
    5:  {'ata':'32-42','task':'32-42-00-200-801','dispatch':'SCHEDULE',
         'action':'Inspect brake assembly per AMM 32-42-00-200-801. Check wear pins.'},
    6:  {'ata':'32-41','task':'32-41-00-200-801','dispatch':'GROUND',
         'action':'Tire change required per AMM 32-41-00-200-801. Inspect rim. Do not dispatch.'},
    7:  {'ata':'32-31','task':'32-31-00-200-802','dispatch':'SCHEDULE',
         'action':'Corrosion treatment per AMM 32-31-00-200-802. Inspect chrome rod for pitting.'},
    8:  {'ata':'32-00','task':'32-00-00-200-801','dispatch':'GROUND',
         'action':'HARD LANDING INSPECTION per AMM 32-00-00-200-801. Must not depart until complete.'},
    9:  {'ata':'32-31','task':'32-31-00-200-801','dispatch':'PRIORITY',
         'action':'Combined N2+seal failure. Service N2 and replace seals per AMM 32-31/32.'},
    10: {'ata':'32-00','task':'32-00-00-200-900','dispatch':'GROUND',
         'action':'AOG INSPECTION. Do not dispatch. Raise defect per AMM 32-00-00-200-900.'},
    11: {'ata':'32-30','task':'32-30-00-200-801','dispatch':'SCHEDULE',
         'action':'Bogie pitch fault. Inspect semi-levered alignment per AMM 32-30-00-200-801.'}
}


# ─── PYTORCH MODEL DEFINITION & LOADING ──────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights

# ─── FIX APPLIED: REPLACED MODEL CLASS WITH CORRECT ARCHITECTURE ─────────────
class AEROTWIN_CNN_LSTM(nn.Module):
    def __init__(self, n_channels=6, n_timesteps=100, n_classes=12):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.attention = TemporalAttention(hidden_dim=256)
        self.trunk = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head_class    = nn.Linear(128, n_classes)
        self.head_severity = nn.Linear(128, 4)
        self.head_rul = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.cnn(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        context, _ = self.attention(lstm_out)
        shared = self.trunk(context)
        return self.head_class(shared), self.head_severity(shared), self.head_rul(shared)

print(f"Loading AI Model from {MODEL_PATH}...")
try:
    model = AEROTWIN_CNN_LSTM().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print("✅ Model loaded successfully.")
    
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
    print("✅ Scalers loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model or scalers. Error: {e}")
    print("Please ensure you have run the training script first.")
    os._exit(1)


# ─── OLLAMA LLM INTEGRATION ──────────────────────────────────────────────────
def check_ollama():
    """Returns (available, model_name)"""
    try:
        r = requests.get(f'{OLLAMA_URL}/api/tags', timeout=3)
        models = [m['name'].split(':')[0] for m in r.json().get('models', [])]
        if OLLAMA_MODEL in models:
            return True, OLLAMA_MODEL
        if models:
            return True, models[0]
        return False, 'none'
    except:
        return False, 'offline'

def build_prompt(class_id, severity, rul_hours, sink_rate, peak_g,
                 bogie_angle, seal_force, session_history):
    amm   = AMM_REF.get(class_id, AMM_REF[0])
    cname = CLASS_NAMES.get(class_id, f'Class {class_id}')

    cs25 = 'No limit load exceedance.'
    if (peak_g is not None and peak_g >= 2.0) or \
       (sink_rate is not None and sink_rate >= 3.05) or class_id == 8:
        cs25 = 'CS-25.473 LIMIT LOAD EXCEEDED. Mandatory hard landing inspection triggered.'

    if session_history:
        hlines = []
        for i, h in enumerate(session_history[-4:], 1):
            hlines.append(
                f'  Landing {i}: {CLASS_NAMES.get(h.get("class_id",0),"?")} | '
                f'Sev {h.get("severity","?")} | RUL {h.get("rul_hours",0):.0f}h'
            )
        hist = '\n'.join(hlines)
    else:
        hist = '  No prior landings this session.'

    sr = f'{sink_rate:.2f} m/s' if sink_rate is not None else 'not recorded'
    pg = f'{peak_g:.2f} g'      if peak_g is not None     else 'not recorded'
    ba = f'{bogie_angle:.1f} deg' if bogie_angle is not None else 'not recorded'
    sf = f'{seal_force:.0f} N'  if seal_force is not None  else 'not recorded'

    return f"""You are an MRO engineer specialising in Boeing 787-9 landing gear (ATA 32). Review this AI diagnosis.

DIAGNOSIS:
  Fault:    Class {class_id} — {cname}
  Severity: {severity}/4
  RUL:      {rul_hours:.0f} flight hours remaining
  ATA:      {amm['ata']}  |  AMM task: {amm['task']}
  Dispatch: {amm['dispatch']}

LANDING PARAMETERS:
  Sink rate:   {sr}
  Peak g:      {pg}
  Bogie angle: {ba}
  Seal force:  {sf}
  CS-25.473:   {cs25}

SESSION HISTORY (last 4 landings):
{hist}

Write exactly 4 sentences for a line maintenance engineer:
1. What the fault signature means physically for the strut condition now.
2. Whether the session trend is stable, worsening, or critical.
3. The specific AMM task required and urgency.
4. Precise dispatch decision — can this aircraft depart?

Technical language. No bullet points. No markdown. Plain text only."""

def llm_explain(class_id, severity, rul_hours, sink_rate, peak_g,
                bogie_angle, seal_force, session_history):
    available, model_name = check_ollama()
    if not available:
        return rule_based_explain(class_id, severity, rul_hours, sink_rate, peak_g), 'rule_based'
    try:
        prompt = build_prompt(class_id, severity, rul_hours, sink_rate, peak_g,
                              bogie_angle, seal_force, session_history)
        resp = requests.post(
            f'{OLLAMA_URL}/api/generate',
            json={
                'model':  model_name,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'num_predict': 320,
                    'top_p': 0.9
                }
            },
            timeout=OLLAMA_TIMEOUT
        )
        text = resp.json().get('response', '').strip()
        if text:
            return text, f'ollama/{model_name}'
        return rule_based_explain(class_id, severity, rul_hours, sink_rate, peak_g), 'rule_based'
    except Exception as e:
        print(f'Ollama call failed: {e}')
        return rule_based_explain(class_id, severity, rul_hours, sink_rate, peak_g), 'rule_based'

def rule_based_explain(class_id, severity, rul_hours, sink_rate, peak_g):
    amm   = AMM_REF.get(class_id, AMM_REF[0])
    cname = CLASS_NAMES.get(class_id, f'Class {class_id}')
    urgency = {1:'Monitor at next scheduled check.',
               2:'Schedule within 5 flight days.',
               3:'Priority action within 24 hours.',
               4:'GROUND AIRCRAFT. Do not dispatch.'}
    parts = [
        f'Fault: {cname} (Class {class_id}), Severity {severity}/4.',
        f'Estimated RUL: {rul_hours:.0f} flight hours remaining.',
        f'ATA {amm["ata"]} | Task {amm["task"]}. {amm["action"]}',
        urgency.get(severity, 'Refer to AMM.')
    ]
    if sink_rate and sink_rate >= 3.05:
        parts.append(f'CS-25.473 exceedance at {sink_rate:.2f} m/s. Hard landing inspection mandatory.')
    if peak_g and peak_g >= 2.0:
        parts.append(f'Peak {peak_g:.2f}g exceeds 2.0g CS-25.473 limit.')
    return ' '.join(parts)


# ─── ODE PHYSICS ENGINE ──────────────────────────────────────────────────────
def run_ode(mass, sink, tc):
    Tk  = tc + 273.15
    k   = K_NOM * (Tk / 288.15)
    vis = float(np.exp(float(np.clip(2500 * (1/Tk - 1/288.15), -10, 10))))
    b   = B_NOM * float(np.clip(vis, 0.3, 3.5))

    def ode(t, x):
        d, v = x
        pen  = max(0., d)
        Fc   = K_CONTACT * pen**1.5 + B_CONTACT * max(0., v) * pen
        fr   = min(0.80, abs(d) / STROKE_LIM)
        kp   = k * max(1., min(2., (1. / (1. - fr))**N_POLY))
        Fs   = kp / ARM_RATIO**2 * d
        Fd   = b * v
        va   = max(abs(v), 0.001)
        Fse  = 4000 + (8000 - 4000) * np.exp(-va / 0.05)
        return [v, (Fc - Fs - Fd - Fse - mass * G) / mass]

    def ev(t, x): return STROKE_LIM - x[0]
    ev.terminal = True; ev.direction = -1

    sol = solve_ivp(ode, [0, SIM_TIME], [0.001, -abs(sink)],
                    method="RK45", rtol=1e-3, atol=1e-4, max_step=0.02, events=ev)
    t_s, d_s, v_s = sol.t, sol.y[0], sol.y[1]
    tq = np.linspace(0, SIM_TIME, N_PTS)

    if len(t_s) >= 4:
        tn  = np.linspace(0, 1, len(t_s))
        tqn = np.linspace(0, 1, N_PTS)
        dr  = np.interp(tqn, tn, d_s)
        vr  = np.interp(tqn, tn, v_s)
    else:
        w  = np.sqrt(k / max(mass, 1))
        z  = min(b / (2 * np.sqrt(k * max(mass, 1))), 0.99)
        wd = w * np.sqrt(max(1 - z**2, 1e-6))
        dr = (sink / wd) * np.exp(-z * w * tq) * np.sin(wd * tq)
        vr = np.gradient(dr, tq)

    dt  = SIM_TIME / (N_PTS - 1)
    ar  = np.gradient(vr) / dt
    fr  = mass * (ar + G)
    fc  = np.clip(np.abs(dr) / STROKE_LIM, 0, 0.79)
    pr  = P0 * (1. / np.maximum(1 - fc, 0.01))**N_POLY
    bgr = 0.02 * np.random.randn(N_PTS)
    peak_g = float(np.max(np.abs(ar)) / G)
    
    bogie_angle = float(np.degrees(np.arcsin(np.clip(dr[-1] / (STROKE_LIM * 1.2), -1, 1))))
    seal_force = float(np.mean(4000 + (8000 - 4000) * np.exp(-np.abs(vr) / 0.05)))
    return dr, vr, fr, ar, pr, bgr, peak_g, bogie_angle, seal_force

def parse_pkt(raw):
    try:
        p = raw.strip().split(",")
        if len(p) < 5: return None
        return {
            "mass_kg":     float(p[0]) * SLUG_TO_KG,
            "sink_ms":     float(p[1]) * FPS_TO_MS,
            "temp_c":      float(p[2]),
            "wow_main":    int(float(p[3])),
            "wow_nose":    int(float(p[4])) if len(p) > 4 else 0,
            "alt_ft":      float(p[5])       if len(p) > 5 else 0,
            "airspeed_kt": float(p[8])       if len(p) > 8 else 0,
        }
    except:
        return None


# ─── INFERENCE WORKER (Replaces API Call) ────────────────────────────────────
def run_inference(dr, vr, fr, ar, pr, bgr, sink, peak_g, bogie_angle, seal_force, session_history):
    # 1. Stack and shape signals
    signals = np.array([dr, vr, fr, ar, pr, bgr], dtype=np.float32)
    
    # 2. Apply Scalers
    sig_norm = signals.copy()
    for ch in range(6):
        sig_norm[ch] = (sig_norm[ch] - scalers[ch].mean_[0]) / scalers[ch].scale_[0]

    # 3. PyTorch Inference
    tensor = torch.tensor(sig_norm).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        lc, ls, lr = model(tensor)
        class_id  = int(lc.argmax(1).item())
        severity  = int(ls.argmax(1).item()) + 1
        rul_hours = float(lr.item()) * RUL_MAX_HOURS

    # 4. Generate LLM Explanation
    explanation, expl_source = llm_explain(
        class_id, severity, rul_hours,
        sink, peak_g, bogie_angle, seal_force, session_history
    )

    amm = AMM_REF.get(class_id, AMM_REF[0])
    
    return {
        'class_id':    class_id,
        'severity':    severity,
        'rul_hours':   rul_hours,
        'ata_chapter': amm['ata'],
        'amm_task':    amm['task'],
        'dispatch':    amm['dispatch'],
        'explanation': explanation,
        'expl_source': expl_source,
        'sink_rate':   sink,
        'peak_g':      peak_g,
        'cs25_flag':   bool((peak_g >= 2.0) or (sink >= 3.05) or (class_id == 8))
    }


# ─── BACKGROUND UDP THREAD ───────────────────────────────────────────────────
def udp_worker(result_q, telemetry_q):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)

    wow_count      = 0
    landed         = False
    last_land      = 0
    td_mass        = 180000.
    td_sink        = 1.5
    td_temp        = 15.
    session_history = []

    while True:
        try:
            data, _ = sock.recvfrom(2048)
            pkt = parse_pkt(data.decode("utf-8", errors="ignore"))
            if pkt is None: continue

            telemetry_q.put(pkt)

            if pkt["wow_main"] == 1 and not landed:
                wow_count += 1
                if wow_count == 1:
                    td_mass = pkt["mass_kg"]
                    td_sink = max(abs(pkt["sink_ms"]), 0.10)
                    td_temp = pkt["temp_c"]

                if wow_count >= 3:
                    if time.time() - last_land < 20:
                        landed = True
                        continue
                    
                    landed    = True
                    last_land = time.time()

                    telemetry_q.put("PROCESSING")

                    # Run ODE Physics
                    dr, vr, fr, ar, pr, bgr, peak_g, bogie_angle, seal_force = run_ode(td_mass, td_sink, td_temp)

                    # Run Local Inference (PyTorch + Ollama)
                    result = run_inference(dr, vr, fr, ar, pr, bgr, 
                                           td_sink, peak_g, bogie_angle, seal_force, session_history)

                    # Update Session History
                    session_history.append({
                        "class_id": result.get("class_id",  0),
                        "severity": result.get("severity",  1),
                        "rul_hours": result.get("rul_hours", 0.0)
                    })
                    if len(session_history) > 10:
                        session_history = session_history[-10:]

                    result_q.put(result)

            elif pkt["wow_main"] == 0:
                wow_count = 0
                landed    = False

        except socket.timeout:
            continue
        except Exception as e:
            print(f"UDP Error/Inference Error: {e}")


# ─── DASHBOARD GUI ───────────────────────────────────────────────────────────
class MRODashboard:
    def __init__(self, root, result_q, telemetry_q):
        self.root        = root
        self.result_q    = result_q
        self.telemetry_q = telemetry_q
        self.landing_count = 0

        root.title("AEROTWIN  |  B787-9 Landing Gear Health Monitor")
        root.configure(bg="#0a0e1a")
        root.geometry("1160x760")
        root.resizable(False, False)

        mono_s = ("Courier New", 9)
        sml    = ("Courier New", 10, "bold")
        tiny   = ("Courier New", 8)

        # ── TOP BAR ──
        top = tk.Frame(root, bg="#0d1225", height=56)
        top.pack(fill="x"); top.pack_propagate(False)

        tk.Label(top, text="◈  AEROTWIN  |  B787-9  |  ATA 32 LANDING GEAR HEALTH MONITOR",
                 font=("Courier New", 12, "bold"),
                 bg="#0d1225", fg="#4fc3f7").pack(side="left", padx=20, pady=14)

        self.clock_lbl = tk.Label(top, text="", font=mono_s, bg="#0d1225", fg="#546e7a")
        self.clock_lbl.pack(side="right", padx=20)

        self.status_dot = tk.Label(top, text="●  LISTENING", font=sml, bg="#0d1225", fg="#00c853")
        self.status_dot.pack(side="right", padx=10)

        self.llm_lbl = tk.Label(top, text="LLM ○  offline", font=("Courier New", 8), bg="#0d1225", fg="#37474f")
        self.llm_lbl.pack(side="right", padx=12)

        # ── BODY ──
        body = tk.Frame(root, bg="#0a0e1a")
        body.pack(fill="both", expand=True, padx=16, pady=(8, 16))

        # LEFT 
        left = tk.Frame(body, bg="#0d1225", width=264)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        tk.Label(left, text="LIVE TELEMETRY", font=sml, bg="#0d1225", fg="#4fc3f7").pack(pady=(14, 6))
        tk.Frame(left, bg="#1e3a5f", height=1).pack(fill="x", padx=12)

        self.tele_vars = {}
        for label, unit in [
            ("MASS", "kg"), ("SINK RATE", "m/s"), ("TEMP", "°C"),
            ("AIRSPEED", "kt"), ("ALT AGL", "ft"),
            ("WOW MAIN", ""), ("WOW NOSE", "")
        ]:
            row = tk.Frame(left, bg="#0d1225")
            row.pack(fill="x", padx=14, pady=3)
            tk.Label(row, text=label, font=tiny, bg="#0d1225", fg="#546e7a", width=10, anchor="w").pack(side="left")
            var = tk.StringVar(value="---")
            tk.Label(row, textvariable=var, font=mono_s, bg="#0d1225", fg="#b0bec5", anchor="e", width=12).pack(side="right")
            tk.Label(row, text=unit, font=tiny, bg="#0d1225", fg="#37474f", width=4).pack(side="right")
            self.tele_vars[label] = var

        tk.Frame(left, bg="#1e3a5f", height=1).pack(fill="x", padx=12, pady=8)
        tk.Label(left, text="LANDING LOG", font=sml, bg="#0d1225", fg="#4fc3f7").pack(pady=(0, 6))

        self.log_text = tk.Text(left, font=("Courier New", 8), bg="#060b14", fg="#546e7a",
                                relief="flat", bd=0, state="disabled", wrap="word", width=28)
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(0, 14))

        # RIGHT 
        right = tk.Frame(body, bg="#0a0e1a")
        right.pack(side="left", fill="both", expand=True)

        # Standby
        self.standby_frame = tk.Frame(right, bg="#0a0e1a")
        self.standby_frame.pack(fill="both", expand=True)
        tk.Label(self.standby_frame, text="", bg="#0a0e1a").pack(expand=True)
        tk.Label(self.standby_frame, text="AWAITING TOUCHDOWN", font=("Courier New", 20, "bold"), bg="#0a0e1a", fg="#1e3a5f").pack()
        tk.Label(self.standby_frame, text="Fly an approach and land the 787-9\nDiagnosis fires instantly on touchdown detection",
                 font=("Courier New", 10), bg="#0a0e1a", fg="#263650", justify="center").pack(pady=8)
        tk.Label(self.standby_frame, text="", bg="#0a0e1a").pack(expand=True)

        # Processing
        self.proc_frame = tk.Frame(right, bg="#0a0e1a")
        tk.Label(self.proc_frame, text="", bg="#0a0e1a").pack(expand=True)
        self.proc_lbl = tk.Label(self.proc_frame, text="⟳  ODE SYNTHESIS → CNN-LSTM → LLM REASONING...",
                                 font=("Courier New", 13, "bold"), bg="#0a0e1a", fg="#4fc3f7")
        self.proc_lbl.pack()
        tk.Label(self.proc_frame, text="Physics engine → Fault classification → Ollama regulatory briefing",
                 font=("Courier New", 9), bg="#0a0e1a", fg="#263650").pack(pady=4)
        tk.Label(self.proc_frame, text="", bg="#0a0e1a").pack(expand=True)

        # Diagnosis card
        self.diag_frame = tk.Frame(right, bg="#0a0e1a")

        r1 = tk.Frame(self.diag_frame, bg="#0a0e1a")
        r1.pack(fill="x", pady=(0, 6))

        self.class_lbl = tk.Label(r1, text="NORMAL", font=("Courier New", 30, "bold"), bg="#0a0e1a", fg="#ffffff")
        self.class_lbl.pack(side="left")

        badges = tk.Frame(r1, bg="#0a0e1a")
        badges.pack(side="right", padx=8, pady=8)

        self.sev_badge = tk.Label(badges, text="MONITOR", font=("Courier New", 11, "bold"), bg="#00c853", fg="#000000", padx=12, pady=4)
        self.sev_badge.pack()

        self.dispatch_badge = tk.Label(badges, text="CONTINUE", font=("Courier New", 9, "bold"), bg="#1a2a1a", fg="#00c853", padx=10, pady=3)
        self.dispatch_badge.pack(pady=(3, 0))

        tk.Frame(self.diag_frame, bg="#1e3a5f", height=1).pack(fill="x")

        # Metrics
        metrics = tk.Frame(self.diag_frame, bg="#0a0e1a")
        metrics.pack(fill="x", pady=12)
        self.metric_vars = {}
        for col, (title, unit) in enumerate([("SINK RATE", "m/s"), ("PEAK G", "g"), ("RUL", "flight hrs"), ("SEVERITY", "/ 4")]):
            cell = tk.Frame(metrics, bg="#0d1225", padx=14, pady=10)
            cell.grid(row=0, column=col, padx=5, sticky="nsew")
            metrics.columnconfigure(col, weight=1)
            var = tk.StringVar(value="--")
            tk.Label(cell, textvariable=var, font=("Courier New", 24, "bold"), bg="#0d1225", fg="#4fc3f7").pack()
            tk.Label(cell, text=f"{title}  {unit}", font=("Courier New", 7), bg="#0d1225", fg="#546e7a").pack()
            self.metric_vars[title] = var

        self.ata_lbl = tk.Label(self.diag_frame, text="", font=("Courier New", 10, "bold"), bg="#0a0e1a", fg="#80cbc4", anchor="w")
        self.ata_lbl.pack(fill="x", padx=4, pady=(0, 4))

        self.expl_hdr = tk.Label(self.diag_frame, text="EXPLANATION  [AWAITING LLM]", font=("Courier New", 8, "bold"), bg="#0a0e1a", fg="#37474f", anchor="w")
        self.expl_hdr.pack(fill="x", padx=4)

        self.expl_text = tk.Text(self.diag_frame, font=("Courier New", 9), bg="#060b14", fg="#cfd8dc", relief="flat", bd=0, height=9, wrap="word", padx=10, pady=10)
        self.expl_text.pack(fill="x", pady=(2, 8))

        self.alert_banner = tk.Label(self.diag_frame, text="⚠  CS-25.473 — HARD LANDING — MANDATORY INSPECTION REQUIRED", font=("Courier New", 10, "bold"), bg="#b71c1c", fg="#ffffff", padx=12, pady=8)

        footer = tk.Frame(self.diag_frame, bg="#0a0e1a")
        footer.pack(fill="x")
        self.count_lbl = tk.Label(footer, text="LANDING #1", font=("Courier New", 9, "bold"), bg="#0a0e1a", fg="#263650")
        self.count_lbl.pack(side="left")
        self.ts_lbl = tk.Label(footer, text="", font=("Courier New", 8), bg="#0a0e1a", fg="#263650")
        self.ts_lbl.pack(side="right")

        self._check_llm_status()
        self._tick()
        self._poll()

    def _check_llm_status(self):
        available, model_name = check_ollama()
        if available:
            self.llm_lbl.config(text=f"LLM ●  {model_name}", fg="#00c853")
        else:
            self.llm_lbl.config(text="LLM ○  rule-based", fg="#ff6d00")
        self.root.after(12000, self._check_llm_status)

    def _tick(self):
        self.clock_lbl.config(text=time.strftime("UTC  %Y-%m-%d  %H:%M:%S"))
        self.root.after(1000, self._tick)

    def _poll(self):
        try:
            while True:
                pkt = self.telemetry_q.get_nowait()
                if pkt == "PROCESSING":
                    self._show_processing()
                else:
                    self._update_telemetry(pkt)
        except queue.Empty:
            pass
        try:
            self._show_diagnosis(self.result_q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _update_telemetry(self, pkt):
        self.tele_vars["MASS"].set(f"{pkt['mass_kg']:.0f}")
        self.tele_vars["SINK RATE"].set(f"{pkt['sink_ms']:+.3f}")
        self.tele_vars["TEMP"].set(f"{pkt['temp_c']:.1f}")
        self.tele_vars["AIRSPEED"].set(f"{pkt['airspeed_kt']:.1f}")
        self.tele_vars["ALT AGL"].set(f"{pkt['alt_ft']:.0f}")
        self.tele_vars["WOW MAIN"].set("ON GND"   if pkt["wow_main"] else "AIRBORNE")
        self.tele_vars["WOW NOSE"].set("ON GND"   if pkt["wow_nose"] else "AIRBORNE")

    def _show_processing(self):
        self.status_dot.config(text="●  ANALYSING", fg="#ffd600")
        self.standby_frame.pack_forget()
        self.diag_frame.pack_forget()
        self.proc_frame.pack(fill="both", expand=True)

    def _show_diagnosis(self, data):
        self.landing_count += 1
        self.proc_frame.pack_forget()
        self.standby_frame.pack_forget()
        self.diag_frame.pack(fill="both", expand=True)

        class_id     = data.get("class_id",    0)
        severity     = data.get("severity",    1)
        rul_hours    = data.get("rul_hours",    0.0)
        sink         = data.get("sink_rate",   0.0)
        peak_g       = data.get("peak_g",      0.0)
        expl         = data.get("explanation", "")
        expl_source  = data.get("expl_source", "rule_based")
        ata          = data.get("ata_chapter", "32-00")
        amm_task     = data.get("amm_task",    "")
        dispatch     = data.get("dispatch",    "CONTINUE")
        cs25         = data.get("cs25_flag",   False)

        name  = CLASS_NAMES.get(class_id, f"CLASS {class_id}")
        color = SEVERITY_COLOR.get(severity, "#00c853")
        slbl  = SEVERITY_LABEL.get(severity, "MONITOR")
        dcol  = DISPATCH_COLOR.get(dispatch, "#4fc3f7")

        self.class_lbl.config(text=name)
        self.sev_badge.config(text=slbl, bg=color, fg="#000000" if severity < 4 else "#ffffff")
        self.dispatch_badge.config(text=f"DISPATCH: {dispatch}", fg=dcol, bg="#0d1225")

        self.metric_vars["SINK RATE"].set(f"{sink:.2f}")
        self.metric_vars["PEAK G"].set(f"{peak_g:.2f}")
        self.metric_vars["RUL"].set(f"{rul_hours:.0f}")
        self.metric_vars["SEVERITY"].set(str(severity))

        self.ata_lbl.config(text=f"ATA {ata}  |  {amm_task}")

        if "ollama" in expl_source:
            model_name = expl_source.split("/")[-1]
            self.expl_hdr.config(text=f"EXPLANATION  [◈ LOCAL LLM — {model_name.upper()}]", fg="#00c853")
        else:
            self.expl_hdr.config(text="EXPLANATION  [RULE-BASED — run: ollama serve]", fg="#546e7a")

        if cs25:
            self.alert_banner.pack(fill="x", pady=(0, 6))
        else:
            self.alert_banner.pack_forget()

        self.expl_text.config(state="normal")
        self.expl_text.delete("1.0", "end")
        self.expl_text.insert("end", expl.strip())
        self.expl_text.config(state="disabled")

        self.count_lbl.config(text=f"LANDING #{self.landing_count}")
        self.ts_lbl.config(text=time.strftime("%Y-%m-%d  %H:%M:%S UTC"))

        self.log_text.config(state="normal")
        self.log_text.insert("1.0", f"#{self.landing_count}  {name}  sev={severity}  RUL={rul_hours:.0f}h  {dispatch}\n")
        self.log_text.config(state="disabled")
        self.status_dot.config(text="●  LISTENING", fg="#00c853")


# ─── MAIN EXECUTION ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rq = queue.Queue()
    tq = queue.Queue()
    
    print("Initializing UDP Background Worker...")
    threading.Thread(target=udp_worker, args=(rq, tq), daemon=True).start()
    
    print("Launching AeroTwin MRO Dashboard...")
    root = tk.Tk()
    MRODashboard(root, rq, tq)
    root.mainloop()