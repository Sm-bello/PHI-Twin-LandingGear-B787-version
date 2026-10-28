"""
================================================================================
  AEROTWIN B787 LIVE BRIDGE
  FlightGear UDP -> Python ODE -> Flask API -> Diagnosis
  
  Replaces MATLAB relay. MATLAB only needed for training, not live inference.
  
  Launch sequence:
    Terminal 1: python api.py
    Terminal 2: python live_bridge.py  
    FlightGear: --generic="socket,out,30,localhost,16662,udp,output"
                NOTE: use "output" not "output.xml" in the FG arg
================================================================================
"""

import socket
import time
import numpy as np
import requests
from scipy.integrate import solve_ivp

UDP_IP   = "127.0.0.1"
UDP_PORT = 16662
API_URL  = "http://localhost:5000/predict"
SIM_TIME = 4.0
N_PTS    = 100
SLUG_TO_KG = 14.5939
FPS_TO_MS  = 0.3048
K_NOM=1500000.; B_NOM=60000.; P0=20.0e6; V0=0.012; A_PISTON=0.018
N_POLY=1.35; STROKE_LIM=0.635; ARM_RATIO=0.55
K_CONTACT=1.0e6; B_CONTACT=5.0e4; G=9.81

def ode_landing(k, b, mass, sink, tc, health=1.0, b_ang=0.0):
    Tk  = tc + 273.15
    k   = k * (Tk / 288.15)
    vis = float(np.exp(float(np.clip(2500*(1/Tk - 1/288.15), -10, 10))))
    b   = b * float(np.clip(vis, 0.3, 3.5))

    def ode(t, x):
        d, v = x
        pen = max(0., d)
        Fc  = K_CONTACT * pen**1.5 + B_CONTACT * max(0., v) * pen
        frac = min(0.80, abs(d)/STROKE_LIM)
        kp   = k * max(1., min(2., (1./(1.-frac))**N_POLY))
        ke   = kp / ARM_RATIO**2
        Fs   = ke * d
        Fd   = b * v
        va   = max(abs(v), 0.001)
        Fs_st = 8000*(1+0.5*(1-health))
        Fs_co = 4000*(1+0.8*(1-health))
        Fse   = Fs_co + (Fs_st-Fs_co)*np.exp(-va/0.05)
        Fb    = Fs*(1-np.cos(np.deg2rad(b_ang)))*0.15
        acc   = (Fc - Fs - Fd - Fse - Fb - mass*G) / mass
        return [v, acc]

    def ev(t, x): return STROKE_LIM - x[0]
    ev.terminal=True; ev.direction=-1

    sol = solve_ivp(ode, [0, SIM_TIME], [0.001, -abs(sink)],
                    method="RK45", rtol=1e-3, atol=1e-4,
                    max_step=0.02, events=ev)
    t_s, d_s, v_s = sol.t, sol.y[0], sol.y[1]

    tq = np.linspace(0,SIM_TIME,N_PTS)
    if len(t_s) >= 4:
        tn  = np.linspace(0,1,len(t_s))
        tqn = np.linspace(0,1,N_PTS)
        dr = np.interp(tqn, tn, d_s)
        vr = np.interp(tqn, tn, v_s)
    else:
        w = np.sqrt(k/max(mass,1))
        z = min(b/(2*np.sqrt(k*max(mass,1))), 0.99)
        wd = w*np.sqrt(max(1-z**2,1e-6))
        dr = (sink/wd)*np.exp(-z*w*tq)*np.sin(wd*tq)
        vr = np.gradient(dr, tq)

    dt   = SIM_TIME/(N_PTS-1)
    ar   = np.gradient(vr)/dt
    fr   = mass*(ar+G)
    fc   = np.clip(np.abs(dr)/STROKE_LIM, 0, 0.79)
    pr   = P0*(1./np.maximum(1-fc,0.01))**N_POLY
    bgr  = b_ang*np.exp(-tq/0.8)+0.02*np.random.randn(N_PTS)
    return dr, vr, fr, ar, pr, bgr


def parse_packet(raw):
    try:
        p = raw.strip().split(",")
        if len(p) < 5: return None
        return {
            "mass_kg":   float(p[0]) * SLUG_TO_KG,
            "sink_ms":   float(p[1]) * FPS_TO_MS,
            "temp_c":    float(p[2]),
            "wow_main":  int(float(p[3])),
            "wow_nose":  int(float(p[4])) if len(p)>4 else 0,
            "alt_ft":    float(p[5]) if len(p)>5 else 0,
            "airspeed_kt": float(p[8]) if len(p)>8 else 0,
        }
    except Exception as e:
        print(f"  Parse error: {e} | raw={raw!r}")
        return None


def call_api(dr, vr, fr, ar, pr, bgr, sink, peak_g):
    signals = [dr.tolist(), vr.tolist(), fr.tolist(),
               ar.tolist(), pr.tolist(), bgr.tolist()]
    payload = {"signals": signals, "sink_rate": sink, "peak_g": peak_g}
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        return r.json()
    except Exception as e:
        print(f"  API error: {e}")
        return None


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)
    print(f"AEROTWIN Live Bridge")
    print(f"Listening UDP {UDP_IP}:{UDP_PORT}")
    print(f"API endpoint: {API_URL}")
    print(f"Waiting for FlightGear data...\n")

    wow_count = 0
    landed    = False
    td_mass   = 180000.
    td_sink   = 1.5
    td_temp   = 15.
    last_land = 0

    while True:
        try:
            data, addr = sock.recvfrom(2048)
            raw = data.decode("utf-8", errors="ignore")
            pkt = parse_packet(raw)
            if pkt is None:
                continue

            print(f"  RX: mass={pkt['mass_kg']:.0f}kg sink={pkt['sink_ms']:.2f}m/s "
                  f"temp={pkt['temp_c']:.1f}C wow={pkt['wow_main']} "
                  f"alt={pkt['alt_ft']:.0f}ft")

            if pkt["wow_main"] == 1 and not landed:
                wow_count += 1
                if wow_count == 1:
                    td_mass = pkt["mass_kg"]
                    td_sink = max(abs(pkt["sink_ms"]), 0.1)
                    td_temp = pkt["temp_c"]

                if wow_count >= 3:
                    if time.time() - last_land < 30:
                        print("  (Skipping — recent landing already processed)")
                        landed = True
                        continue

                    landed = True
                    last_land = time.time()
                    print(f"\nTOUCHDOWN DETECTED")
                    print(f"  mass={td_mass:.0f}kg | sink={td_sink:.2f}m/s | temp={td_temp:.1f}C")
                    print(f"  Running ODE simulation...")

                    dr, vr, fr, ar, pr, bgr = ode_landing(
                        K_NOM, B_NOM, td_mass, td_sink, td_temp)
                    peak_g = float(np.max(np.abs(ar)) / G)
                    print(f"  ODE complete. Peak g = {peak_g:.2f}")

                    print(f"  Calling AEROTWIN API...")
                    result = call_api(dr, vr, fr, ar, pr, bgr, td_sink, peak_g)

                    if result:
                        print(f"\n  AI DIAGNOSIS")
                        print(f"  Class:    {result.get('class_id', '?')} | "
                              f"Severity: {result.get('severity', '?')} | "
                              f"RUL: {result.get('rul_percent', 0):.1f}%")
                        print(f"\n  Explanation:")
                        print(f"  {result.get('explanation', 'No explanation')}")
                    else:
                        print("  No response from API — is api.py running?")

                    print(f"\n--- Ready for next landing ---\n")
            else:
                if pkt["wow_main"] == 0:
                    wow_count = 0
                    landed    = False

        except socket.timeout:
            continue
        except KeyboardInterrupt:
            print("\nStopped.")
            break

    sock.close()


if __name__ == "__main__":
    main()