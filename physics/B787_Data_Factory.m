% =========================================================================
%   🏭 AEROTWIN DATA FACTORY V2.0 — BOEING 787-9 DREAMLINER
%   Purpose  : 18,000 physics-based landing events for CNN-LSTM training
%   Aircraft : Boeing 787-9 | Semi-levered 6-wheel MLG bogie | MLW 193,000 kg
%
%   ╔══════════════════════════════════════════════════════════════════════╗
%   ║  V2.0 ARCHITECTURE DECISION — LANDING PHASE ONLY (4 seconds)       ║
%   ║  StopTime: 4 seconds — captures complete landing event              ║
%   ║  t=0.0–3.0s : Touchdown, oleo compression, peak load               ║
%   ║  t=3.0–4.0s : Rebound, rollout, weight transfer                    ║
%   ║                                                                      ║
%   ║  WHY 4 SECONDS (not 11):                                            ║
%   ║  The gear retract sequence (t=5–11s) requires the Side Stay Lock    ║
%   ║  Linkage over-centre toggle mechanism to fire. This is a            ║
%   ║  geometrically singular joint by design (mechanical lock at         ║
%   ║  dead-centre). No solver setting resolves a deliberate singularity. ║
%   ║  By stopping at t=4s, the lock never moves and the simulation       ║
%   ║  runs cleanly and fast for all 18,000 samples.                      ║
%   ║                                                                      ║
%   ║  6 SIGNALS (sim_Actuator_Load DROPPED):                             ║
%   ║  sim_Deflection | sim_Velocity | sim_Force | sim_Acceleration       ║
%   ║  sim_Chamber_Pressure | sim_Bogie_Angle                             ║
%   ║                                                                      ║
%   ║  Classes 11 (Retract Sticking), 13 (WOW Anomaly),                  ║
%   ║  14 (Lock Degradation) are handled by LAYER 3 LLM reasoning         ║
%   ║  using CS-25.729 / AMM 32-31/32/61 regulatory knowledge.           ║
%   ║  These classes do not appear in physics signal space.               ║
%   ╚══════════════════════════════════════════════════════════════════════╝
%
%   CERTIFICATION:
%   ► CS-25.473 / FAR 25.473  — Transport Category (787-9)
%   ► Hard landing trigger    : >=3.05 m/s OR >=2.0g (CS-25.473 limit)
%   ► Ultimate load trigger   : >=2.67g (1.5 × limit, FAR 25.305)
%
%   PHYSICS FUNCTIONS (8 active — actuatorLoad787 retained as scalar):
%   [U1] tireFriction787()      — hydroplaning + ice (B787 200 psi tire)
%   [U2] polytropicKRatio787()  — polytropic K with bogie arm correction
%   [U3] orificeB787()          — velocity-squared damping (Classes 6, 8)
%   [U4] sealFriction787()      — Stribeck seal friction (10× B787 scale)
%   [U5] bogieAngle()           — bogie pitch perturbation (Class 12 key)
%   [U6] actuatorLoad787()      — computed scalar only (Layer 3 use)
%   [U7] bogieStroke787()       — moment-arm corrected stroke
%   [U8] WOWdebounce()          — WOW sensor timing (scalar only)
%   [U9] brakeEnergy787()       — kinetic + potential brake energy (MJ)
%
%   CLASS MANIFEST — 12 classes, 18,000 samples:
%   Class 0  : Normal Operations           (3,500 samples)
%   Class 1  : Nitrogen Leak               (2,200 samples)
%   Class 2  : Worn Oleo Seal              (2,200 samples)
%   Class 3  : Early Degradation           (1,600 samples)
%   Class 4  : Thermal Degradation         (1,400 samples)
%   Class 5  : Brake Fade / Asymmetric     (1,100 samples)
%   Class 6  : Tire Burst                  (  900 samples)
%   Class 7  : Corrosion / Fatigue         (1,100 samples)
%   Class 8  : Hard Landing [CS-25.473]    (  900 samples)
%   Class 9  : Combined Faults N2+Seal     (  800 samples)
%   Class 10 : Impending Failure [AOG]     (  700 samples)
%   Class 12 : Bogie Pitch Fault [B787]    (1,600 samples) ← BogieAngle signal
%                                           ──────────────
%                                    TOTAL : 18,000 samples
%
%   NOTE: Classes 11, 13, 14 handled by Layer 3 LLM (regulatory knowledge):
%   Class 11 : Retract Actuator Sticking → CS-25.729 + AMM 32-31-00-810-801
%   Class 13 : WOW Sensor Anomaly        → ATA 32-60 + AMM 32-61-00-720-801
%   Class 14 : Uplock/Downlock Degrad.   → ATA 32-30 + AMM 32-32-00-810-801
%
%   COLUMN LAYOUT: 32 scalar + 6 signals × 100 timesteps = 632 total
%   Scalars (1–32):
%    1 SampleID          12 Health_Factor        23 ATA_Chapter_Index
%    2 Class             13 Max_Deflection_m      24 Bogie_Init_Angle_deg
%    3 RUL_Hours         14 Max_Velocity_ms       25 Actuator_Fric_N (computed)
%    4 Mass_kg           15 Max_Accel_g           26 Brake_Energy_MJ
%    5 SinkRate_ms       16 Max_Force_N           27 Gear_Cmd_Mode (=0 landing)
%    6 Friction_Coef     17 Max_Pressure_Pa       28 WOW_Delay_ms
%    7 Temp_C            18 Seal_Force_N          29 Bogie_Peak_Angle_deg
%    8 K_Stiffness_Nm    19 Max_Bogie_Angle_deg   30 Actuator_Peak_Load_N (computed)
%    9 B_Damping_Nsm     20 Actuator_Load_N (comp) 31 CS25_HardLanding_Flag
%   10 K_Normalized      21 Maint_Trigger         32 Retract_Flag (=0 all)
%   11 B_Normalized      22 Severity
%   Signals (6 × 100 timesteps = 600 columns):
%   Defl(33–132) | Velo(133–232) | Force(233–332) | Accel(333–432)
%   Pressure(433–532) | BogieAngle(533–632)
%
%   CHECKPOINT: Auto-saves every 1,000 samples to projPath folder.
%               If MATLAB crashes, reload last CHECKPOINT_sc*.mat and resume.
%
%   SOLVER (set these before running — apply manually to model too):
%   Solver: ode14x | RelTol: 1e-2 | AbsTol: 1e-2
%   MaxStep: 0.05  | ZeroCrossControl: DisableAll
%   (Manual joint fixes must already be saved in the Simscape model)
%
%   ⚠️  MODEL NAME: Confirm 'B787_DigitalTwin_Core' matches your file.
%   ⚠️  WORKSPACE VARS: k_strut, b_strut, mass_landing, sink_rate_input,
%       runway_friction, temperature, seal_friction, act_fric,
%       bogie_init_angle, wow_delay_ms, brake_torque_cmd, gear_lever_cmd
%   ⚠️  OUTPUT SIGNALS (To Workspace blocks in model):
%       sim_Deflection | sim_Velocity | sim_Force | sim_Acceleration
%       sim_Chamber_Pressure | sim_Bogie_Angle
%       (sim_Actuator_Load no longer required)
%
%   RUL FIX (added March 2026):
%   RUL_Percent removed – now RUL_Hours based on TTF_BASE and severity progression.
%   Each class sequenced fresh→degraded (severity 1→4) so LSTM sees real prognostic trend.
% =========================================================================

clear; clc; close all;
fprintf('🏭 AEROTWIN V2.0 — BOEING 787-9 | CS-25.473 | 6-SIGNAL LANDING PHASE\n');
fprintf('====================================================================\n');
fprintf('Mode     : 4-second landing phase | No retract singularity\n');
fprintf('Cert     : CS-25.473 FAR 25.473 Transport Category (787-9)\n');
fprintf('Trigger  : >=3.05 m/s OR >=2.0g — CS-25.473 limit load\n');
fprintf('Gear     : Semi-levered 6-wheel MLG bogie | Landing phase only\n');
fprintf('Dataset  : 18,000 samples | 12 classes | 6 signals | 632 columns\n');
fprintf('Physics  : tireFriction787 | polytropicK787 | orificeB787\n');
fprintf('           sealFriction787 | bogieAngle | bogieStroke787\n');
fprintf('           WOWdebounce | brakeEnergy787\n');
fprintf('Dropped  : sim_Actuator_Load (Layer 3 LLM handles Classes 11,13,14)\n');
fprintf('RUL FIX  : TTF-based RUL_Hours (column 3) – real prognostic target\n');
fprintf('====================================================================\n\n');

% =========================================================================
%   SECTION 1: REGULATORY THRESHOLDS — CS-25.473 / FAR 25.473
% =========================================================================
REG.SinkDesign       = 3.05;    % m/s  CS-25.473(d) design sink velocity
REG.SinkHardLanding  = 3.05;    % m/s  Mandatory ATA 32-00 inspection trigger
REG.SinkCaution      = 2.50;    % m/s  Precautionary inspection threshold
REG.AccelLimit       = 2.00;    % g    CS-25.473 limit load
REG.AccelExtreme     = 2.67;    % g    FAR 25.305 ultimate = 1.5 × limit
REG.MTOW             = 254000;  % kg   787-9 MTOW
REG.MLW              = 193000;  % kg   787-9 MLW
REG.FixedInterval    = 1000;    % cycles B787 A-check interval (approx)
REG.SimStopTime      = '4';     % seconds — landing phase only

% =========================================================================
%   RUL FIX: TIME-TO-FAILURE BASELINES (flight hours)
%   Engineering basis: MRO planning intervals + fault severity physics
% =========================================================================
TTF_BASE = containers.Map( ...
    {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  12}, ...
    {2000, 500, 300, 900, 600, 250,   5, 1200,  50, 180,   2, 400} ...
);
%  Class 0  Normal           2000h  Standard wear interval
%  Class 1  N2 Leak           500h  Seal dry-run accelerates wear
%  Class 2  Worn Seal         300h  Active leak path — rapid
%  Class 3  Early Degrad      900h  Initiation phase — slow
%  Class 4  Thermal Degrad    600h  Heat cycling fatigue
%  Class 5  Brake Fade        250h  Brake inspection interval
%  Class 6  Tire Burst          5h  Grounding — immediate change
%  Class 7  Corrosion        1200h  Slow chemical process
%  Class 8  Hard Landing       50h  CS-25.473 mandatory insp within 50h
%  Class 9  N2+Seal           180h  Compounded fault modes
%  Class 10 AOG                 2h  Terminal — aircraft grounded
%  Class 12 Bogie Pitch       400h  Alignment fatigue cycle

SEV_MULT = [1.00, 0.65, 0.30, 0.08];
%  sev=1: fault just detected  → 100% of TTF remains
%  sev=2: progressing          →  65% of TTF remains
%  sev=3: advanced             →  30% of TTF remains
%  sev=4: near-failure         →   8% of TTF remains

% =========================================================================
%   MAINTENANCE KNOWLEDGE BASE — 15 entries (all retained for Layer 3)
% =========================================================================
MAINT( 1).chapter='ATA 32-10'; MAINT( 1).category='Routine Line Check';
MAINT( 1).action='Visual inspection oleo strut, bogie beam, tire pressure, hydraulic leak check per AMM 32-11-00-200-801'; MAINT( 1).cost=850;

MAINT( 2).chapter='ATA 32-10'; MAINT( 2).category='Nitrogen System Service';
MAINT( 2).action='Oleo pre-charge pressure test, nitrogen recharge per AMM 32-11-00-600-801. Verify extension height per AMM Table 201'; MAINT( 2).cost=2200;

MAINT( 3).chapter='ATA 32-10'; MAINT( 3).category='Hydraulic Damper Service';
MAINT( 3).action='Damping performance test, oleo seal inspection/replacement, hydraulic fluid sampling per AMM 32-11-00-400-801'; MAINT( 3).cost=4500;

MAINT( 4).chapter='ATA 32-10'; MAINT( 4).category='Degradation Trending Inspection';
MAINT( 4).action='Parameter trending review, borescope if indicated, extension height measurement per AMM 32-11-00-200-803'; MAINT( 4).cost=5800;

MAINT( 5).chapter='ATA 32-10'; MAINT( 5).category='Thermal Stress Inspection';
MAINT( 5).action='Oleo seal hardness check, hydraulic fluid viscosity test, heat discolouration assessment per AMM 32-11-00-200-805'; MAINT( 5).cost=4800;

MAINT( 6).chapter='ATA 32-10'; MAINT( 6).category='Asymmetric Load / Bogie Inspection';
MAINT( 6).action='Bogie beam alignment, trunnion pin, sidestay, drag brace, attachment hardware torque check per AMM 32-11-00-820-801'; MAINT( 6).cost=7500;

MAINT( 7).chapter='ATA 32-40'; MAINT( 7).category='Wheel and Brake Assembly Inspection';
MAINT( 7).action='Tire serviceability, brake wear indicators, rim NDT, wheel bearing post-event inspection per AMM 32-41-00-200-801'; MAINT( 7).cost=12000;

MAINT( 8).chapter='ATA 32-10'; MAINT( 8).category='Corrosion Control Treatment';
MAINT( 8).action='NDT pitting assessment, cadmium plate inspection, protective treatment, torque verification per AMM 32-11-00-300-801'; MAINT( 8).cost=8500;

MAINT( 9).chapter='ATA 32-00'; MAINT( 9).category='Hard Landing Inspection [CS-25.473]';
MAINT( 9).action=['CS-25.473 TRIGGERED: Full structural NDT — oleo strut, bogie beam, trunnion, '...
                  'sidestay, drag brace, wing attach fittings. Per AMM 32-00-00-200-801. '...
                  'TRIGGER: sink >=3.05 m/s OR >=2.0g. Aircraft grounded until cleared.']; MAINT( 9).cost=28000;

MAINT(10).chapter='ATA 32-10'; MAINT(10).category='Component Overhaul';
MAINT(10).action='Full strut disassembly, complete seal kit replacement, cylinder honing, nitrogen recharge, pressure test per AMM 32-11-00-500-801'; MAINT(10).cost=65000;

MAINT(11).chapter='ATA 32-00'; MAINT(11).category='Emergency AOG Maintenance';
MAINT(11).action='AOG repair — expedited parts sourcing, mandatory ground until structural clearance per AMM 32-00-00-900-801 and Boeing SRM'; MAINT(11).cost=185000;

MAINT(12).chapter='ATA 32-31'; MAINT(12).category='Retract Actuator Inspection [Layer 3]';
MAINT(12).action='Actuator removal and bench test, seal kit, re-installation per AMM 32-31-00-810-801. CS-25.729 retraction system check.'; MAINT(12).cost=15000;

MAINT(13).chapter='ATA 32-10'; MAINT(13).category='Bogie Beam Structural Inspection [Class 12]';
MAINT(13).action='Bogie pitch angle measurement, bogie beam crack NDT (eddy current + mag particle), alignment check per AMM 32-11-00-820-803'; MAINT(13).cost=18000;

MAINT(14).chapter='ATA 32-61'; MAINT(14).category='WOW Proximity Sensor Rigging [Layer 3]';
MAINT(14).action='Weight-on-Wheels sensor gap check, actuation timing test, replace if outside tolerance per AMM 32-61-00-720-801'; MAINT(14).cost=6500;

MAINT(15).chapter='ATA 32-32'; MAINT(15).category='Lock Assembly Inspection [Layer 3]';
MAINT(15).action='Uplock/downlock mechanism inspection, hook wear check, spring pre-load, rigging per AMM 32-32-00-810-801'; MAINT(15).cost=12500;

ECON.Routine      = 850;
ECON.Detailed     = 5800;
ECON.HardLanding  = 28000;
ECON.AOG          = 185000;
ECON.PrevOverhaul = 65000;
ECON.Emergency    = 185000;

FAT.S_ut  = 1950e6;
FAT.S_e   = 0.5 * 1950e6;
FAT.b     = -0.12;
FAT.N_ref = 1e6;

% =========================================================================
%   SECTION 2: B787-9 PHYSICS CONSTANTS
% =========================================================================
MASS_NORM_MIN  = 150000;
MASS_NORM_MAX  = 193000;
MASS_MTOW      = 254000;
K_NOM          = 1500000;
B_NOM          = 60000;
STROKE_LIMIT   = 0.635;
BOGIE_ARM_RATIO= 0.55;
SINK_MIN       = 0.3;
SINK_MAX       = 3.5;
F_DRY          = 0.80;
F_WET          = 0.40;
F_ICE          = 0.20;
T_ISA          = 15;
T_HOT          = 45;
T_COLD         = -30;

% =========================================================================
%   SECTION 3: MODEL SETUP & TURBO SOLVER
%   ⚠️  Confirm model filename matches your Simscape file
% =========================================================================
mdl = 'B787_DigitalTwin_Core';
if ~bdIsLoaded(mdl), load_system(mdl); end

% Disable UDP / FlightGear live feed for batch run
try
    set_param([mdl '/UDP Receive1'], 'Commented', 'on');
    fprintf('✅ UDP block disabled\n');
catch
    fprintf('ℹ️  UDP block not found / already disabled\n');
end

% ── 1. FIX THE SOLVER (Force true Fixed-Step ode14x)
set_param(mdl, 'SolverType',           'Fixed-step'); 
set_param(mdl, 'Solver',               'ode14x');
set_param(mdl, 'FixedStep',            '1e-4'); % 1 millisecond steps
set_param(mdl, 'ZeroCrossControl',     'DisableAll');
set_param(mdl, 'ConsistencyChecking',  'none');

% ── 2. ENGAGE TURBO MODE (R2023a compatible)
set_param(mdl, 'SimulationMode',       'accelerator');
set_param(mdl, 'FastRestart',          'on');
set_param(mdl, 'SimMechanicsOpenEditorOnUpdate', 'off');

fprintf('✅ Turbo Mode Engaged: Accelerator + Fast Restart (ode14x at 1e-3)\n\n');

% =========================================================================
%   SECTION 4: VISUAL CONFIRMATION RUN (nominal landing only)
% =========================================================================
fprintf('🎬 Running nominal visual check (4s landing, gear stays down)...\n');
open_system(mdl);
set_param(mdl, 'SimMechanicsOpenEditorOnUpdate', 'on');
fprintf('🛬 Nominal: 180,000 kg | 1.8 m/s | K=1.5MN | B=60k | ISA 15°C\n\n');

seal_vis = sealFriction787(1.8, 1.0);
act_vis  = actuatorLoad787(1000, 1.8, 1.0);
assignAll787(mdl, K_NOM, B_NOM, 180000, 1.8, F_DRY, T_ISA, ...
             seal_vis, 1000, 0.0, 15.0, 50.0);
vOut = sim(mdl, 'StopTime', REG.SimStopTime);

if isempty(vOut.ErrorMessage)
    fprintf('✅ Visual nominal run OK — 4s landing cycle confirmed\n\n');
else
    fprintf('⚠️  Visual run warning: %s\n\n', vOut.ErrorMessage);
end
set_param(mdl, 'SimMechanicsOpenEditorOnUpdate', 'off');

% =========================================================================
%   SECTION 5: OUTPUT PATHS
% =========================================================================
projPath = 'C:\Users\User\Desktop\B787_AeroTwin';
outFile  = 'AEROTWIN_B787_V2p0_RULfixed.csv';   % new filename to avoid overwriting
outPath  = fullfile(projPath, outFile);
if ~exist(projPath, 'dir'), mkdir(projPath); end

fprintf('📂 Output  : %s\n', outPath);
fprintf('⏳ Est.    : 3–8 hours (18,000 × 4s sims) | run overnight\n');
fprintf('📊 Columns : 632 (32 scalar + 6 signals × 100 timesteps)\n');
fprintf('💾 Checkpt : Every 1,000 samples → CHECKPOINT_sc#####.mat\n');
fprintf('====================================================================\n\n');

% =========================================================================
%   SECTION 6: PRE-FLIGHT SANITY CHECK — physics functions
% =========================================================================
fprintf('🔍 Pre-flight CS-25.473 + B787 physics sanity check...\n');

tc_t = T_ISA; sink_t = 2.0; mass_t = 180000; k_t = K_NOM;

fric_t  = tireFriction787(F_DRY,  sink_t, tc_t);
krat_t  = polytropicKRatio787(sink_t, mass_t, k_t, 1.35, BOGIE_ARM_RATIO);
brat_t  = orificeB787(sink_t);
seal_t  = sealFriction787(sink_t, 1.0);
bang_t  = bogieAngle(0, 1.0, 0.0);
aload_t = actuatorLoad787(1000, sink_t, 1.0);
bstk_t  = bogieStroke787(sink_t, mass_t, k_t, BOGIE_ARM_RATIO);
wow_t   = WOWdebounce(20.0);
brkE_t  = brakeEnergy787(mass_t, sink_t, 72.0);

fprintf('   tireFriction787 (dry,2.0m/s,15°C)           = %.3f\n', fric_t);
fprintf('   polytropicKRatio787 (2.0m/s,180t,1.5MN/m)   = %.3f\n', krat_t);
fprintf('   orificeB787 (2.0 m/s, v_ref=2.0)            = %.3f  (=1.00 at ref)\n', brat_t);
fprintf('   sealFriction787 (2.0m/s, health=1.0)        = %.0f N\n', seal_t);
fprintf('   bogieAngle (class=0, health=1.0)             = %.2f deg\n', bang_t);
fprintf('   actuatorLoad787 (1000N, computed scalar)     = %.0f N\n', aload_t);
fprintf('   bogieStroke787 (2.0m/s, 180t, 1.5MN/m)      = %.4f m\n', bstk_t);
fprintf('   WOWdebounce (20ms)                           = %.1f ms\n', wow_t);
fprintf('   brakeEnergy787 (180t, 2.0m/s, 72m/s)        = %.1f MJ\n', brkE_t);

% Confirm 6 signals extractable
assignAll787(mdl, K_NOM, B_NOM, mass_t, sink_t, fric_t, tc_t, ...
             seal_t, aload_t, bang_t, wow_t, brkE_t);
tSim = sim(mdl, 'StopTime', REG.SimStopTime);

if ~isempty(tSim.ErrorMessage)
    fprintf('❌ PRE-FLIGHT SIM ERROR: %s\n', tSim.ErrorMessage);
    fprintf('   Check solver settings and joint fixes in your Simscape model.\n');
    return;
end

[dw,vw,fw,aw,pw,bgw,ok] = xWin787(tSim, 100);
if ~ok
    fprintf('❌ Signal extraction failed. Verify To Workspace block names:\n');
    fprintf('   sim_Deflection | sim_Velocity | sim_Force | sim_Acceleration\n');
    fprintf('   sim_Chamber_Pressure | sim_Bogie_Angle\n');
    return;
end

gPk = max(abs(aw)) / 9.81;
fprintf('\n✅ All 6 signals confirmed — 4-second landing cycle clean\n');
fprintf('   Peak accel   : %.3fg  (CS-25.473 limit: %.1fg)  ', gPk, REG.AccelLimit);
if gPk >= REG.AccelLimit
    fprintf('⚠️  TRIGGERS ATA 32-00 HARD LANDING\n');
else
    fprintf('✅ Within normal envelope\n');
end
fprintf('   Deflection   : %.3f m  (stroke limit: %.3f m)\n', max(abs(dw)), STROKE_LIMIT);
fprintf('   Force        : %.0f N\n', max(abs(fw)));
fprintf('   Pressure     : %.0f Pa\n', max(abs(pw)));
fprintf('   Bogie angle  : %.3f deg\n', max(abs(bgw)));

fprintf('\n🚀 Pre-flight passed. Starting 18,000-sample V2.0 batch (RUL fixed)...\n');
fprintf('====================================================================\n\n');

% =========================================================================
%   SECTION 7: STORAGE INITIALISATION
% =========================================================================
DS        = [];
META      = [];
MLOG      = {};
dmg       = zeros(1, 13);   % Index 1–11 = Classes 0–10, Index 12 = Class 12
costCBM   = 0;
costFIXED = 0;
aogSaved  = 0;
sc = 0; cc = 0;
t0 = tic;
rng(42);
TOTAL_TARGET = 18000;

% =========================================================================
%   CLASS 0 — NORMAL OPERATIONS  (3,500 samples)
%   K: 1.43M–1.57M | B: 57k–63k | Health: 85–100%
% =========================================================================
fprintf('--- ✅ CLASS 0: NORMAL OPERATIONS (3,500) ---\n');
tgt = 3500; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = 0.85 + 0.15 * rand();
    tc     = T_COLD + (T_HOT - T_COLD) * rand();
    mass   = MASS_NORM_MIN + (MASS_NORM_MAX - MASS_NORM_MIN) * rand();
    if rand() < 0.8
        sink = 0.3 + (2.0 - 0.3) * rand();
    else
        sink = 2.0 + (3.05 - 2.0) * rand();
    end

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * (0.953 + 0.094 * rand());
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * (0.950 + 0.100 * rand());

    fric_base = randsample([F_DRY, F_WET, F_ICE], 1, true, [0.70, 0.25, 0.05]);
    fric   = tireFriction787(fric_base, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(800 + 400 * rand(), sink, health);
    b_ang  = bogieAngle(0, health, 0.0);
    wow_d  = WOWdebounce(10 + 20 * rand());
    brk_e  = brakeEnergy787(mass, sink, 70 + 10 * rand());

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp = max(abs(aw)) / 9.81;
                kn = kNorm787(k, tc);
                [sev,tr,mi] = assess787(0,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                dp = dmgCalc787(sink, mass, max(abs(dw)), tc, FAT);
                dmg(1) = dmg(1) + dp;

                % ── RUL FIX: TTF-based flight hours ──────────────────────
                degrad_frac = cs / (tgt - 1);   % 0 at first sample, 1 at last
                rul_sev = min(4, 1 + floor(degrad_frac * 3));   % severity for RUL progression
                rul = TTF_BASE(0) * SEV_MULT(rul_sev) * (1 + 0.08 * randn());
                rul = max(0.5, rul);   % floor at 0.5h
                % ─────────────────────────────────────────────────────────

                c = mCost787(tr,sev,mi,MAINT); costCBM = costCBM + c;
                costFIXED = fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl = double(sink>=REG.SinkHardLanding || gp>=REG.AccelLimit);
                DS   = [DS;   bRow787(sc,0,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META = [META; sc,0,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1} = {sc,0,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs = cs + 1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,350)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 0: %d | Damage: %.6f | RUL min=%.0fh max=%.0fh mean=%.0fh\n\n', ...
    cs, dmg(1), min(META(META(:,2)==0,8)), max(META(META(:,2)==0,8)), mean(META(META(:,2)==0,8)));

% =========================================================================
%   CLASS 1 — NITROGEN LEAK  (2,200 samples)
%   K: 0.75M–1.15M | B: 57k–63k
% =========================================================================
fprintf('--- 🧪 CLASS 1: NITROGEN LEAK (2,200) | K:0.75M–1.15M ---\n');
tgt = 2200; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = rand();
    tc     = T_COLD + (T_HOT - T_COLD) * rand();
    mass   = MASS_NORM_MIN + (MASS_NORM_MAX - MASS_NORM_MIN) * rand();
    sink   = SINK_MIN + (SINK_MAX - SINK_MIN) * rand();

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * ((750000 + 400000 * health) / K_NOM);
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * (0.950 + 0.100 * rand());

    fric_base = randsample([F_DRY, F_WET, F_ICE], 1, true, [0.60, 0.30, 0.10]);
    fric   = tireFriction787(fric_base, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(800 + 400 * rand(), sink, health);
    b_ang  = bogieAngle(1, health, 0.0);
    wow_d  = WOWdebounce(10 + 20 * rand());
    brk_e  = brakeEnergy787(mass, sink, 70 + 10 * rand());

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp = max(abs(aw)) / 9.81; kn = kNorm787(k, tc);
                [sev,tr,mi] = assess787(1,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                dp = dmgCalc787(sink,mass,max(abs(dw)),tc,FAT);
                dmg(2)=dmg(2)+dp;

                degrad_frac = cs / (tgt - 1);
                rul_sev = min(4, 1 + floor(degrad_frac * 3));
                rul = TTF_BASE(1) * SEV_MULT(rul_sev) * (1 + 0.08*randn());
                rul = max(0.5, rul);

                c=mCost787(tr,sev,mi,MAINT); costCBM=costCBM+c;
                costFIXED=fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl=double(sink>=REG.SinkHardLanding||gp>=REG.AccelLimit);
                DS=[DS; bRow787(sc,1,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META=[META; sc,1,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1}={sc,1,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs=cs+1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,400)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 1: %d | Damage: %.6f | RUL min=%.0fh max=%.0fh mean=%.0fh\n\n', ...
    cs, dmg(2), min(META(META(:,2)==1,8)), max(META(META(:,2)==1,8)), mean(META(META(:,2)==1,8)));

% =========================================================================
%   CLASS 2 — WORN OLEO SEAL  (2,200 samples)
%   K: 1.43M–1.57M | B: 13.5k–30k
%   sealFriction787 Stribeck signature most distinctive here
% =========================================================================
fprintf('--- 🔧 CLASS 2: WORN OLEO SEAL (2,200) | B:13.5k–30k ---\n');
tgt = 2200; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = rand();
    tc     = T_COLD + (T_HOT - T_COLD) * rand();
    mass   = MASS_NORM_MIN + (MASS_NORM_MAX - MASS_NORM_MIN) * rand();
    sink   = SINK_MIN + (SINK_MAX - SINK_MIN) * rand();

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * (0.953 + 0.094 * rand());
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * ((13500 + 16500 * health) / B_NOM);

    fric_base = randsample([F_DRY, F_WET, F_ICE], 1, true, [0.60, 0.30, 0.10]);
    fric   = tireFriction787(fric_base, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(800 + 400 * rand(), sink, health);
    b_ang  = bogieAngle(2, health, 0.0);
    wow_d  = WOWdebounce(10 + 20 * rand());
    brk_e  = brakeEnergy787(mass, sink, 70 + 10 * rand());

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp=max(abs(aw))/9.81; kn=kNorm787(k,tc);
                [sev,tr,mi]=assess787(2,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                dp=dmgCalc787(sink,mass,max(abs(dw)),tc,FAT);
                dmg(3)=dmg(3)+dp;

                degrad_frac = cs / (tgt - 1);
                rul_sev = min(4, 1 + floor(degrad_frac * 3));
                rul = TTF_BASE(2) * SEV_MULT(rul_sev) * (1 + 0.08*randn());
                rul = max(0.5, rul);

                c=mCost787(tr,sev,mi,MAINT); costCBM=costCBM+c;
                costFIXED=fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl=double(sink>=REG.SinkHardLanding||gp>=REG.AccelLimit);
                DS=[DS; bRow787(sc,2,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META=[META; sc,2,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1}={sc,2,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs=cs+1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,400)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 2: %d | Damage: %.6f | RUL min=%.0fh max=%.0fh mean=%.0fh\n\n', ...
    cs, dmg(3), min(META(META(:,2)==2,8)), max(META(META(:,2)==2,8)), mean(META(META(:,2)==2,8)));

% =========================================================================
%   CLASS 3 — EARLY DEGRADATION  (1,600 samples)
%   K: 1.26M–1.41M | B: 51k–57k | Health: 70–85%
% =========================================================================
fprintf('--- ⚠️  CLASS 3: EARLY DEGRADATION (1,600) | K:1.26M–1.41M ---\n');
tgt = 1600; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = 0.70 + 0.15 * rand();
    tc     = T_COLD + (T_HOT - T_COLD) * rand();
    mass   = MASS_NORM_MIN + (MASS_NORM_MAX - MASS_NORM_MIN) * rand();
    sink   = SINK_MIN + (SINK_MAX - SINK_MIN) * rand();

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * ((1260000 + 1000000*(health-0.70)) / K_NOM);
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * ((51000 + 40000*(health-0.70)) / B_NOM);

    fric_base = randsample([F_DRY, F_WET], 1, true, [0.80, 0.20]);
    fric   = tireFriction787(fric_base, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(900 + 300*rand(), sink, health);
    b_ang  = bogieAngle(3, health, 0.5*rand());
    wow_d  = WOWdebounce(15 + 20*rand());
    brk_e  = brakeEnergy787(mass, sink, 68 + 12*rand());

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp=max(abs(aw))/9.81; kn=kNorm787(k,tc);
                [sev,tr,mi]=assess787(3,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                dp=dmgCalc787(sink,mass,max(abs(dw)),tc,FAT);
                dmg(4)=dmg(4)+dp;

                degrad_frac = cs / (tgt - 1);
                rul_sev = min(4, 1 + floor(degrad_frac * 3));
                rul = TTF_BASE(3) * SEV_MULT(rul_sev) * (1 + 0.08*randn());
                rul = max(0.5, rul);

                c=mCost787(tr,sev,mi,MAINT); costCBM=costCBM+c;
                costFIXED=fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl=double(sink>=REG.SinkHardLanding||gp>=REG.AccelLimit);
                DS=[DS; bRow787(sc,3,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META=[META; sc,3,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1}={sc,3,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs=cs+1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,300)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 3: %d | Damage: %.6f | RUL min=%.0fh max=%.0fh mean=%.0fh\n\n', ...
    cs, dmg(4), min(META(META(:,2)==3,8)), max(META(META(:,2)==3,8)), mean(META(META(:,2)==3,8)));

% =========================================================================
%   CLASS 4 — THERMAL DEGRADATION  (1,400 samples)
%   K: 1.38M–1.52M | B: 39k–54k | Forced hot ops 35–45°C
% =========================================================================
fprintf('--- 🌡️  CLASS 4: THERMAL DEGRADATION (1,400) | Forced hot ops ---\n');
tgt = 1400; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = 0.60 + 0.30 * rand();
    tc     = 35 + (T_HOT - 35) * rand();
    mass   = MASS_NORM_MIN + (MASS_NORM_MAX - MASS_NORM_MIN) * rand();
    sink   = SINK_MIN + (SINK_MAX - SINK_MIN) * rand();

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * ((1380000 + 140000*health)/K_NOM) * (0.97 + 0.06*rand());
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * ((39000 + 15000*health)/B_NOM) * (0.92 + 0.16*rand());

    fric   = tireFriction787(F_DRY, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(1000 + 500*rand(), sink, health);
    b_ang  = bogieAngle(4, health, 0.5*rand());
    wow_d  = WOWdebounce(10 + 25*rand());
    brk_e  = brakeEnergy787(mass, sink, 72+15*rand()) * (1.2 + 0.3*rand());

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp=max(abs(aw))/9.81; kn=kNorm787(k,tc);
                [sev,tr,mi]=assess787(4,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                dp=dmgCalc787(sink,mass,max(abs(dw)),tc,FAT);
                dmg(5)=dmg(5)+dp;

                degrad_frac = cs / (tgt - 1);
                rul_sev = min(4, 1 + floor(degrad_frac * 3));
                rul = TTF_BASE(4) * SEV_MULT(rul_sev) * (1 + 0.08*randn());
                rul = max(0.5, rul);

                c=mCost787(tr,sev,mi,MAINT); costCBM=costCBM+c;
                costFIXED=fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl=double(sink>=REG.SinkHardLanding||gp>=REG.AccelLimit);
                DS=[DS; bRow787(sc,4,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META=[META; sc,4,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1}={sc,4,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs=cs+1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,300)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 4: %d | Damage: %.6f | RUL min=%.0fh max=%.0fh mean=%.0fh\n\n', ...
    cs, dmg(5), min(META(META(:,2)==4,8)), max(META(META(:,2)==4,8)), mean(META(META(:,2)==4,8)));

% =========================================================================
%   CLASS 5 — BRAKE FADE / ASYMMETRIC LOAD  (1,100 samples)
%   K: 1.43M–1.57M | B: 21k–84k (ERRATIC — key LSTM signature)
% =========================================================================
fprintf('--- 🛑 CLASS 5: BRAKE FADE (1,100) | B: 21k–84k ERRATIC ---\n');
tgt = 1100; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = 0.75 + 0.20 * rand();
    tc     = T_ISA + 10*rand();
    mass   = MASS_NORM_MIN + (MASS_NORM_MAX-MASS_NORM_MIN)*(0.90+0.20*rand());
    sink   = 0.8 + (2.8-0.8)*rand();

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * (0.953 + 0.094*rand());
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * ((21000 + 63000*rand()) / B_NOM);

    fric_base = F_DRY * (0.85 + 0.30*rand());
    fric   = tireFriction787(fric_base, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(800 + 400*rand(), sink, health);
    b_ang  = bogieAngle(5, health, 1.0*rand());
    wow_d  = WOWdebounce(10 + 20*rand());
    brk_e  = brakeEnergy787(mass, sink, 74+8*rand()) * (1.3+0.4*rand());

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp=max(abs(aw))/9.81; kn=kNorm787(k,tc);
                [sev,tr,mi]=assess787(5,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                dp=dmgCalc787(sink,mass,max(abs(dw)),tc,FAT);
                dmg(6)=dmg(6)+dp;

                degrad_frac = cs / (tgt - 1);
                rul_sev = min(4, 1 + floor(degrad_frac * 3));
                rul = TTF_BASE(5) * SEV_MULT(rul_sev) * (1 + 0.08*randn());
                rul = max(0.5, rul);

                c=mCost787(tr,sev,mi,MAINT); costCBM=costCBM+c;
                costFIXED=fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl=double(sink>=REG.SinkHardLanding||gp>=REG.AccelLimit);
                DS=[DS; bRow787(sc,5,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META=[META; sc,5,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1}={sc,5,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs=cs+1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,200)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 5: %d | Damage: %.6f | RUL min=%.0fh max=%.0fh mean=%.0fh\n\n', ...
    cs, dmg(6), min(META(META(:,2)==5,8)), max(META(META(:,2)==5,8)), mean(META(META(:,2)==5,8)));

% =========================================================================
%   CLASS 6 — TIRE BURST  (900 samples)
%   K: 0.78M–1.05M | B: 90k–127.5k | orificeB787 active
% =========================================================================
fprintf('--- 💥 CLASS 6: TIRE BURST (900) | orificeB787 active ---\n');
tgt = 900; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = 0.40 + 0.20*rand();
    tc     = T_ISA + 5*rand();
    mass   = MASS_NORM_MAX * (0.80 + 0.30*rand());
    sink   = 2.0 + (3.5-2.0)*rand();

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * ((780000 + 270000*rand()) / K_NOM);
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * ((90000 + 37500*rand()) / B_NOM);
    b = b * orificeB787(sink);

    fric   = tireFriction787(F_DRY, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(1200 + 600*rand(), sink, health);
    b_ang  = bogieAngle(6, health, 1.5+1.5*rand());
    wow_d  = WOWdebounce(15 + 30*rand());
    brk_e  = brakeEnergy787(mass, sink, 76+10*rand());

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp=max(abs(aw))/9.81; kn=kNorm787(k,tc);
                [sev,tr,mi]=assess787(6,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                dp=dmgCalc787(sink,mass,max(abs(dw)),tc,FAT);
                dmg(7)=dmg(7)+dp;

                degrad_frac = cs / (tgt - 1);
                rul_sev = min(4, 1 + floor(degrad_frac * 3));
                rul = TTF_BASE(6) * SEV_MULT(rul_sev) * (1 + 0.08*randn());
                rul = max(0.5, rul);

                c=mCost787(tr,sev,mi,MAINT); costCBM=costCBM+c;
                costFIXED=fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl=double(sink>=REG.SinkHardLanding||gp>=REG.AccelLimit);
                DS=[DS; bRow787(sc,6,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META=[META; sc,6,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1}={sc,6,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs=cs+1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,200)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 6: %d | Damage: %.6f | RUL min=%.2fh max=%.2fh mean=%.2fh\n\n', ...
    cs, dmg(7), min(META(META(:,2)==6,8)), max(META(META(:,2)==6,8)), mean(META(META(:,2)==6,8)));

% =========================================================================
%   CLASS 7 — CORROSION / FATIGUE  (1,100 samples)
%   K: 1.05M–1.26M | B: 66k–84k | Tropical ops 20–35°C
% =========================================================================
fprintf('--- 🌊 CLASS 7: CORROSION/FATIGUE (1,100) | Tropical ops ---\n');
tgt = 1100; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = 0.50 + 0.50*rand();
    tc     = 20 + 15*rand();
    mass   = MASS_NORM_MIN + (MASS_NORM_MAX-MASS_NORM_MIN)*rand();
    sink   = SINK_MIN + (SINK_MAX-SINK_MIN)*rand();

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * ((1050000 + 210000*health)/K_NOM) * (0.97+0.06*rand());
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * ((66000 + 18000*(1-health))/B_NOM) * (0.95+0.10*rand());

    fric_base = randsample([F_DRY, F_WET], 1, true, [0.50, 0.50]);
    fric   = tireFriction787(fric_base, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(900+600*(1-health), sink, health);
    b_ang  = bogieAngle(7, health, 0.8*rand());
    wow_d  = WOWdebounce(15+25*rand());
    brk_e  = brakeEnergy787(mass, sink, 68+12*rand());

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp=max(abs(aw))/9.81; kn=kNorm787(k,tc);
                [sev,tr,mi]=assess787(7,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                dp=dmgCalc787(sink,mass,max(abs(dw)),tc,FAT);
                dmg(8)=dmg(8)+dp;

                degrad_frac = cs / (tgt - 1);
                rul_sev = min(4, 1 + floor(degrad_frac * 3));
                rul = TTF_BASE(7) * SEV_MULT(rul_sev) * (1 + 0.08*randn());
                rul = max(0.5, rul);

                c=mCost787(tr,sev,mi,MAINT); costCBM=costCBM+c;
                costFIXED=fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl=double(sink>=REG.SinkHardLanding||gp>=REG.AccelLimit);
                DS=[DS; bRow787(sc,7,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META=[META; sc,7,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1}={sc,7,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs=cs+1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,200)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 7: %d | Damage: %.6f | RUL min=%.0fh max=%.0fh mean=%.0fh\n\n', ...
    cs, dmg(8), min(META(META(:,2)==7,8)), max(META(META(:,2)==7,8)), mean(META(META(:,2)==7,8)));

% =========================================================================
%   CLASS 8 — HARD LANDING / OVERLOAD  (900 samples)
%   CS-25.473(d): sink >=3.05 m/s → mandatory ATA 32-00 inspection
%   orificeB787 applied — high-sink velocity-squared damping
% =========================================================================
fprintf('--- 💢 CLASS 8: HARD LANDING [CS-25.473] (900) | orificeB active ---\n');
tgt = 900; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = 0.40 + 0.40*rand();
    tc     = T_ISA + 20*rand();
    mass   = 170000 + (MASS_MTOW-170000)*rand();
    sink   = 3.0 + (REG.SinkDesign-3.0)*rand();

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * ((1120000 + 260000*health)/K_NOM);
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * ((45000 + 27000*rand())/B_NOM);
    b = b * orificeB787(sink);

    fric_base = randsample([F_DRY, F_WET], 1, true, [0.70, 0.30]);
    fric   = tireFriction787(fric_base, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(1500+1000*rand(), sink, health);
    b_ang  = bogieAngle(8, health, 1.0+1.0*rand());
    wow_d  = WOWdebounce(10+30*rand());
    brk_e  = brakeEnergy787(mass, sink, 75+10*rand()) * 1.4;

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp=max(abs(aw))/9.81; kn=kNorm787(k,tc);
                [sev,tr,mi]=assess787(8,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                dp=dmgCalc787(sink,mass,max(abs(dw)),tc,FAT)*1.5;
                dmg(9)=dmg(9)+dp;

                degrad_frac = cs / (tgt - 1);
                rul_sev = min(4, 1 + floor(degrad_frac * 3));
                rul = TTF_BASE(8) * SEV_MULT(rul_sev) * (1 + 0.08*randn());
                rul = max(0.5, rul);

                c=mCost787(tr,sev,mi,MAINT); costCBM=costCBM+c;
                costFIXED=fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl=double(sink>=REG.SinkHardLanding||gp>=REG.AccelLimit);
                DS=[DS; bRow787(sc,8,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META=[META; sc,8,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1}={sc,8,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs=cs+1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,200)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 8: %d | Damage: %.6f | RUL min=%.1fh max=%.1fh mean=%.1fh [CS-25.473 50h window]\n\n', ...
    cs, dmg(9), min(META(META(:,2)==8,8)), max(META(META(:,2)==8,8)), mean(META(META(:,2)==8,8)));

% =========================================================================
%   CLASS 9 — COMBINED FAULTS: N2 LEAK + WORN SEAL  (800 samples)
%   K: 0.65M–1.05M | B: 10.5k–30k — dual degraded signature
% =========================================================================
fprintf('--- 🔥 CLASS 9: COMBINED FAULTS N2+SEAL (800) ---\n');
tgt = 800; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = 0.20 + 0.60*rand();
    tc     = T_COLD + (T_HOT-T_COLD)*rand();
    mass   = MASS_NORM_MIN + (MASS_NORM_MAX-MASS_NORM_MIN)*rand();
    sink   = SINK_MIN + (SINK_MAX-SINK_MIN)*rand();

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * ((650000 + 400000*health)/K_NOM);
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * ((10500 + 19500*health)/B_NOM);

    fric_base = randsample([F_DRY, F_WET, F_ICE], 1, true, [0.50, 0.30, 0.20]);
    fric   = tireFriction787(fric_base, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(1000+500*rand(), sink, health);
    b_ang  = bogieAngle(9, health, 0.8*rand());
    wow_d  = WOWdebounce(15+30*rand());
    brk_e  = brakeEnergy787(mass, sink, 68+12*rand());

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp=max(abs(aw))/9.81; kn=kNorm787(k,tc);
                [sev,tr,mi]=assess787(9,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                dp=dmgCalc787(sink,mass,max(abs(dw)),tc,FAT);
                dmg(10)=dmg(10)+dp;

                degrad_frac = cs / (tgt - 1);
                rul_sev = min(4, 1 + floor(degrad_frac * 3));
                rul = TTF_BASE(9) * SEV_MULT(rul_sev) * (1 + 0.08*randn());
                rul = max(0.5, rul);

                c=mCost787(tr,sev,mi,MAINT); costCBM=costCBM+c;
                costFIXED=fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl=double(sink>=REG.SinkHardLanding||gp>=REG.AccelLimit);
                DS=[DS; bRow787(sc,9,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META=[META; sc,9,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1}={sc,9,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs=cs+1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,150)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 9: %d | Damage: %.6f | RUL min=%.0fh max=%.0fh mean=%.0fh\n\n', ...
    cs, dmg(10), min(META(META(:,2)==9,8)), max(META(META(:,2)==9,8)), mean(META(META(:,2)==9,8)));

% =========================================================================
%   CLASS 10 — IMPENDING FAILURE / AOG  (700 samples)
%   K: 0.50M–0.80M | B: 7.5k–22.5k | Health: 5–30%
% =========================================================================
fprintf('--- 🚨 CLASS 10: IMPENDING FAILURE [AOG] (700) ---\n');
tgt = 700; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = 0.05 + 0.25*rand();
    tc     = T_COLD + (T_HOT-T_COLD)*rand();
    mass   = MASS_NORM_MIN + (MASS_NORM_MAX-MASS_NORM_MIN)*rand();
    sink   = SINK_MIN + (SINK_MAX-SINK_MIN)*rand();

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * ((500000 + 300000*health)/K_NOM);
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * ((7500 + 15000*health)/B_NOM);

    fric_base = randsample([F_DRY, F_WET, F_ICE], 1, true, [0.40, 0.40, 0.20]);
    fric   = tireFriction787(fric_base, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(2000+1000*rand(), sink, health);
    b_ang  = bogieAngle(10, health, 1.0+1.5*rand());
    wow_d  = WOWdebounce(20+50*rand());
    brk_e  = brakeEnergy787(mass, sink, 70+10*rand());

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp=max(abs(aw))/9.81; kn=kNorm787(k,tc);
                [sev,tr,mi]=assess787(10,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                if tr && sev==4, aogSaved=aogSaved+1; end
                dp=dmgCalc787(sink,mass,max(abs(dw)),tc,FAT)*2.5;
                dmg(11)=dmg(11)+dp;

                degrad_frac = cs / (tgt - 1);
                rul_sev = min(4, 1 + floor(degrad_frac * 3));
                rul = TTF_BASE(10) * SEV_MULT(rul_sev) * (1 + 0.08*randn());
                rul = max(0.5, rul);

                c=mCost787(tr,sev,mi,MAINT); costCBM=costCBM+c;
                costFIXED=fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl=double(sink>=REG.SinkHardLanding||gp>=REG.AccelLimit);
                DS=[DS; bRow787(sc,10,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META=[META; sc,10,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1}={sc,10,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs=cs+1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,150)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 10: %d | Damage: %.6f | RUL min=%.2fh max=%.2fh mean=%.2fh (TERMINAL)\n\n', ...
    cs, dmg(11), min(META(META(:,2)==10,8)), max(META(META(:,2)==10,8)), mean(META(META(:,2)==10,8)));

% =========================================================================
%   CLASS 12 — BOGIE PITCH FAULT  (1,600 samples)
%   K: 1.30M–1.50M | B: 55k–65k | bogie_init_angle: 3°–12°
%   PRIMARY SIGNAL: sim_Bogie_Angle — double-peak Force in landing phase
%   Rear axle contacts first → pitch → forward axle hits → second peak
%   This is the key B787 geometry class retained in the 6-signal dataset.
%   AMM: B787 AMM Task 32-11-00-820-803 | ATA 32-10
%   NOTE: Class numbered 12 (not 11) to maintain research consistency.
%         Classes 11, 13, 14 reserved for Layer 3 LLM reasoning.
% =========================================================================
fprintf('--- 📐 CLASS 12: BOGIE PITCH FAULT (1,600) | Double-peak signature ---\n');
fprintf('    Key: sim_Bogie_Angle + Force double-peak | 3°–12° nose-up\n');
tgt = 1600; cs = 0;
while cs < tgt
    sc = sc + 1; cc = cc + 1;
    health = 0.50 + 0.45*rand();
    tc     = T_COLD + (T_HOT-T_COLD)*rand();
    mass   = MASS_NORM_MIN + (MASS_NORM_MAX-MASS_NORM_MIN)*rand();
    sink   = 0.5 + (2.8-0.5)*rand();

    [kt, bt] = therm787(K_NOM, B_NOM, tc);
    k = kt * ((1300000 + 200000*rand())/K_NOM);
    k = k * polytropicKRatio787(sink, mass, k, 1.35, BOGIE_ARM_RATIO);
    b = bt * (0.92 + 0.16*rand());

    fric_base = randsample([F_DRY, F_WET], 1, true, [0.65, 0.35]);
    fric   = tireFriction787(fric_base, sink, tc);
    F_seal = sealFriction787(sink, health);
    act_f  = actuatorLoad787(900+400*rand(), sink, health);

    % *** CLASS 12 KEY: Elevated bogie pitch angle ***
    bogie_fault = 3.0 + 9.0*rand();   % 3°–12° nose-up misalignment
    b_ang = bogieAngle(12, health, bogie_fault);

    wow_d  = WOWdebounce(15+20*rand());
    brk_e  = brakeEnergy787(mass, sink, 70+10*rand());

    assignAll787(mdl, k, b, mass, sink, fric, tc, F_seal, act_f, b_ang, wow_d, brk_e);
    try
        sO = sim(mdl, 'StopTime', REG.SimStopTime);
        if isempty(sO.ErrorMessage)
            [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, 100);
            if ok
                gp=max(abs(aw))/9.81; kn=kNorm787(k,tc);
                [sev,tr,mi]=assess787(12,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
                dp=dmgCalc787(sink,mass,max(abs(dw)),tc,FAT)*0.8;
                dmg(12)=dmg(12)+dp;

                degrad_frac = cs / (tgt - 1);
                rul_sev = min(4, 1 + floor(degrad_frac * 3));
                rul = TTF_BASE(12) * SEV_MULT(rul_sev) * (1 + 0.08*randn());
                rul = max(0.5, rul);

                c=mCost787(tr,sev,mi,MAINT); costCBM=costCBM+c;
                costFIXED=fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
                hl=double(sink>=REG.SinkHardLanding||gp>=REG.AccelLimit);
                DS=[DS; bRow787(sc,12,health,mass,sink,fric,tc,k,b,kn,b,...
                    dw,vw,fw,aw,pw,bgw,F_seal,act_f,b_ang,wow_d,brk_e,...
                    max(abs(bgw)),act_f,hl,tr,sev,dp,rul,c,mi)];
                META=[META; sc,12,tc,sink,fric,act_f,b_ang,rul];
                MLOG{end+1}={sc,12,MAINT(mi).chapter,MAINT(mi).category,sev,tr,c,rul};
                cs=cs+1;
                doCheckpoint(sc,projPath,DS,META,MLOG,dmg,costCBM,costFIXED,aogSaved,cc);
            end
        end
    catch, end
    if mod(cs,300)==0 && cs>0, pProg787(cs,tgt,sc,TOTAL_TARGET,t0); end
end
fprintf('✅ Class 12: %d | Damage: %.6f | RUL min=%.0fh max=%.0fh mean=%.0fh\n\n', ...
    cs, dmg(12), min(META(META(:,2)==12,8)), max(META(META(:,2)==12,8)), mean(META(META(:,2)==12,8)));

% =========================================================================
%   SECTION 8: SAVE ALL OUTPUTS — 632 total columns
% =========================================================================
totalTime = toc(t0);

if ~isempty(DS)
    % Build variable names — 32 scalars + 6×100 signals = 632 total
    vn = {'SampleID','Class','RUL_Hours','Mass_kg','SinkRate_ms',...
          'Friction_Coef','Temp_C','K_Stiffness_Nm','B_Damping_Nsm',...
          'K_Normalized','B_Normalized','Health_Factor',...
          'Max_Deflection_m','Max_Velocity_ms','Max_Accel_g',...
          'Max_Force_N','Max_Pressure_Pa','Seal_Force_N',...
          'Max_Bogie_Angle_deg','Actuator_Load_Computed_N',...
          'Maint_Trigger','Severity','ATA_Chapter_Index',...
          'Bogie_Init_Angle_deg','Actuator_Fric_N','Brake_Energy_MJ',...
          'Gear_Cmd_Mode','WOW_Delay_ms','Bogie_Peak_Angle_deg',...
          'Actuator_Peak_Computed_N','CS25_HardLanding_Flag','Retract_Flag'};

    for sg = {'Defl','Velo','Force','Accel','Pressure','BogieAngle'}
        for t = 1:100
            vn{end+1} = sprintf('%s_t%d', sg{1}, t);
        end
    end

    T = array2table(DS, 'VariableNames', vn);
    writetable(T, outPath);
    fprintf('✅ Main dataset saved: %s\n', outPath);

    % Metadata (now includes RUL_Hours)
    writetable(array2table(META, 'VariableNames', ...
        {'SampleID','Class','Temp_C','SinkRate','Friction','ActuatorFric_Computed','BogieAngle','RUL_Hours'}), ...
        fullfile(projPath, 'AEROTWIN_B787_V2p0_Metadata_RULfixed.csv'));

    % Maintenance log (now includes RUL_Hours)
    mL  = vertcat(MLOG{:});
    mLT = cell2table(mL, 'VariableNames', ...
        {'SampleID','Class','ATA_Chapter','Category','Severity','Triggered','Cost_USD','RUL_Hours'});
    writetable(mLT, fullfile(projPath, 'AEROTWIN_B787_V2p0_MaintenanceLog_RULfixed.csv'));

    % Train/val/test split: 70/15/15
    n = height(T);
    idx_all = randperm(n);
    sp.train_idx = idx_all(1 : round(0.70*n));
    sp.val_idx   = idx_all(round(0.70*n)+1 : round(0.85*n));
    sp.test_idx  = idx_all(round(0.85*n)+1 : end);
    sp.class_labels = {0,1,2,3,4,5,6,7,8,9,10,12};
    sp.signal_names = {'Deflection','Velocity','Force','Acceleration','Pressure','BogieAngle'};
    sp.n_timesteps  = 100;
    sp.n_signals    = 6;
    sp.n_classes    = 12;
    sp.sim_stop_s   = 4;
    sp.rul_column   = 'RUL_Hours';
    sp.rul_max      = 2000;
    sp.note = 'RUL fixed: TTF-based hours. Normalise by 2000 for training. Classes 11,13,14 handled by Layer3 LLM.';
    save(fullfile(projPath, 'train_val_split_B787_V2p0_RULfixed.mat'), 'sp');

    nt    = sum(T.Maint_Trigger);
    sav   = costFIXED - costCBM;
    savPct= 100 * sav / max(costFIXED, 1);

    econD = {'Metric','Value';
        'Version',       'AEROTWIN B787 V2.0 — RUL fixed (flight hours)';
        'Certification', 'CS-25.473 / FAR 25.473 Transport Category';
        'Hard Landing Threshold (m/s)', REG.SinkHardLanding;
        'Hard Landing Threshold (g)',   REG.AccelLimit;
        'Sim Duration per sample (s)',  REG.SimStopTime;
        'Total Samples',                n;
        'Total Columns',                width(T);
        'Signals',                      '6 (ActuatorLoad dropped)';
        'Classes in dataset',           12;
        'Classes in Layer 3 LLM',       'Classes 11,13,14';
        'Maintenance Triggers',         nt;
        'Trigger Rate (%)',             100*nt/n;
        'Fixed-Interval Cost (USD)',    costFIXED;
        'CBM Optimized Cost (USD)',     costCBM;
        'Savings (USD)',                sav;
        'Savings (%)',                   savPct;
        'AOG Prevented (Class10 Sev4)', aogSaved;
        'AOG Prevention Value (USD)',   aogSaved * ECON.AOG;
        'Runtime (hours)',               totalTime/3600};
    for ci = [0:10, 12]
        econD{end+1,1} = sprintf('Miner Damage Class%d', ci);
        econD{end,2}   = dmg(min(ci+1,12));
    end
    writecell(econD, fullfile(projPath, 'AEROTWIN_B787_V2p0_Economic_RULfixed.csv'));

    % =====================================================================
    %   CONSOLE SUMMARY with RUL distribution
    % =====================================================================
    fprintf('\n====================================================================\n');
    fprintf('✅ AEROTWIN B787 V2.0 RUL FIX COMPLETE\n');
    fprintf('   CS-25.473 / FAR 25.473 / ATA 32 | 6-Signal Landing Phase\n');
    fprintf('====================================================================\n');
    fprintf('📊 Samples  : %d | Columns : %d (32 scalar + 6×100)\n', n, width(T));
    fprintf('⏱️  Time     : %.1f hours\n', totalTime / 3600);
    fprintf('📂 File     : %s\n', outPath);
    fprintf('🔬 Signals  : Defl | Velo | Force | Accel | Pressure | BogieAngle\n');
    fprintf('📋 RUL      : Flight hours (column 3) — normalise /2000 for training\n');

    fprintf('\n📈 RUL DISTRIBUTION BY CLASS (prognostic target):\n');
    fprintf('Class | Name                | TTF_base | min(h)  max(h)  mean(h)  std(h) | Status\n');
    fprintf('------+---------------------+----------+-------------------------------+---------\n');
    cNames = {'Normal','N2 Leak','Worn Seal','EarlyDegrad','ThermalDegrad',...
              'BrakeFade','TireBurst','Corrosion','HardLanding','N2+Seal','AOG','BogieP'};
    cIds = [0,1,2,3,4,5,6,7,8,9,10,12];
    ttfVals = [2000,500,300,900,600,250,5,1200,50,180,2,400];
    for ci = 1:12
        cid = cIds(ci);
        rows = T.RUL_Hours(T.Class == cid);
        if isempty(rows)
            fprintf('%5d | %-19s | %8d | NO DATA\n', cid, cNames{ci}, ttfVals(ci));
            continue;
        end
        mn = min(rows); mx = max(rows); mu = mean(rows); sg = std(rows);
        if sg > 5
            status = 'OK';
        elseif sg > 1
            status = 'WARN - low variance';
        else
            status = 'FAIL - flat signal';
        end
        fprintf('%5d | %-19s | %8d | %7.1f  %7.1f  %7.1f  %6.1f | %s\n',...
            cid, cNames{ci}, ttfVals(ci), mn, mx, mu, sg, status);
    end
    fprintf('\nOverall RUL_Hours: min=%.1f  max=%.1f  mean=%.1f  std=%.1f\n',...
        min(T.RUL_Hours), max(T.RUL_Hours), mean(T.RUL_Hours), std(T.RUL_Hours));
    if std(T.RUL_Hours) > 100
        fprintf('OVERALL STATUS: RUL FIX CONFIRMED — sufficient variance for LSTM training\n');
    else
        fprintf('OVERALL STATUS: WARNING — variance too low, check TTF_BASE constants\n');
    end

    fprintf('\n🎯 K/B DISCRIMINABILITY (CNN-LSTM feature space):\n');
    fprintf('   Class  K_norm_mean    B_mean     Seal_mean  BogieAng_mean\n');
    for ci = [0:10, 12]
        km  = mean(T.K_Normalized(T.Class == ci));
        bm  = mean(T.B_Damping_Nsm(T.Class == ci));
        sm  = mean(T.Seal_Force_N(T.Class == ci));
        bam = mean(T.Bogie_Init_Angle_deg(T.Class == ci));
        fprintf('   %2d     %9.0f   %8.0f   %6.0fN   %5.2f°\n', ci, km, bm, sm, bam);
    end

    fprintf('\n💰 ECONOMIC ANALYSIS:\n');
    fprintf('   Fixed-Interval (every %d cycles): $%14.0f\n', REG.FixedInterval, costFIXED);
    fprintf('   CBM AI-Optimised:                 $%14.0f\n', costCBM);
    fprintf('   ─────────────────────────────────────────────\n');
    fprintf('   SAVINGS:                          $%14.0f (%.1f%%)\n', sav, savPct);
    fprintf('   AOG prevented (Class 10 Sev4):    %d  ($%.0f value)\n', ...
        aogSaved, aogSaved * ECON.AOG);

    % Optional: generate analysis plots (same as original, but you can run separately)
    % (Plot code omitted for brevity – can be run after dataset is created)

    fprintf('\n====================================================================\n');
    fprintf('🎓 NEXT STEPS:\n');
    fprintf('   Layer 2: CNN-LSTM training on 632-column dataset\n');
    fprintf('            Input: 6 signals × 100 timesteps = 600 features\n');
    fprintf('            y_rul = df[''RUL_Hours''].values / 2000.0  (normalise to 0-1)\n');
    fprintf('   Layer 3: LLM reasoning for Classes 11,13,14\n');
    fprintf('            Knowledge: CS-25.729 + AMM 32-31/32/61 + ATA 32-30/60\n');
    fprintf('====================================================================\n');
else
    fprintf('❌ No data collected — check model and workspace connections.\n');
end

% =========================================================================
%   FUNCTIONS (unchanged – all the helper functions from original V2.0)
% =========================================================================
function assignAll787(mdl, k, b, mass, sink, fric, tc, seal_f, ...
                      act_fric_in, bogie_ang, wow_delay, brake_e)
    assignin('base', 'k_strut',          k);
    assignin('base', 'b_strut',          b);
    assignin('base', 'mass_landing',     mass);
    assignin('base', 'sink_rate_input',  sink);
    assignin('base', 'runway_friction',  fric);
    assignin('base', 'temperature',      tc);
    assignin('base', 'seal_friction',    seal_f);
    assignin('base', 'act_fric',         act_fric_in);
    assignin('base', 'bogie_init_angle', bogie_ang);
    assignin('base', 'wow_delay_ms',     wow_delay);
    assignin('base', 'brake_torque_cmd', brake_e);
    t_cmd    = [0.0, 4.0];
    v_cmd    = [0,   0  ];
    gear_cmd = timeseries(v_cmd', t_cmd');
    assignin('base', 'gear_lever_cmd', gear_cmd);
end

function [dw,vw,fw,aw,pw,bgw,ok] = xWin787(sO, n)
    ok = false;
    dw=zeros(1,n); vw=zeros(1,n); fw=zeros(1,n);
    aw=zeros(1,n); pw=zeros(1,n); bgw=zeros(1,n);
    try
        Yd  = squeeze(sO.get('sim_Deflection'));
        Yv  = squeeze(sO.get('sim_Velocity'));
        Yf  = squeeze(sO.get('sim_Force'));
        Ya  = squeeze(sO.get('sim_Acceleration'));
        Yp  = squeeze(sO.get('sim_Chamber_Pressure'));
        Ybg = squeeze(sO.get('sim_Bogie_Angle'));
        Yd=Yd(:); Yv=Yv(:); Yf=Yf(:);
        Ya=Ya(:); Yp=Yp(:); Ybg=Ybg(:);
        if length(Yd) < 50, return; end
        xo = linspace(0,1,length(Yd));
        xn = linspace(0,1,n);
        dw  = rs787(Yd,  xo, xn);
        vw  = rs787(Yv,  xo, xn);
        fw  = rs787(Yf,  xo, xn);
        aw  = rs787(Ya,  xo, xn);
        pw  = rs787(Yp,  xo, xn);
        bgw = rs787(Ybg, xo, xn);
        ok  = true;
    catch ME
        fprintf('xWin787: %s\n', ME.message);
    end
end

function o = rs787(d, xo, xn)
    if length(d) == length(xn)
        o = d';
    else
        o = interp1(xo, d, xn, 'linear', 'extrap');
    end
end

function doCheckpoint(sc, projPath, DS, META, MLOG, dmg, ...
                      costCBM, costFIXED, aogSaved, cc)
    if mod(sc, 1000) == 0 && sc > 0
        chkPath = fullfile(projPath, sprintf('CHECKPOINT_sc%05d.mat', sc));
        try
            save(chkPath, 'DS', 'META', 'MLOG', 'dmg', 'costCBM', ...
                 'costFIXED', 'aogSaved', 'sc', 'cc', '-v7.3');
            fprintf('   💾 Checkpoint saved at sc=%d → %s\n', sc, chkPath);
        catch
            fprintf('   ⚠️  Checkpoint save failed at sc=%d\n', sc);
        end
    end
end

function mu = tireFriction787(mu_base, sink_rate, temp_c)
    v_td    = max(55.0, min(82.0, 77.0 - sink_rate * 2.5));
    v_hydro = 65.4;
    if mu_base < 0.60
        hydro_ratio  = max(0, (v_td - v_hydro) / max(82.0 - v_hydro, 1));
        hydro_factor = max(0.50, 1.0 - 0.35 * hydro_ratio);
    else
        hydro_factor = 1.0;
    end
    if temp_c < 0
        ice_factor = max(0.15, 1.0 - 0.010 * abs(temp_c));
    else
        ice_factor = 1.0;
    end
    mu = max(0.08, min(0.92, mu_base * hydro_factor * ice_factor));
end

function ratio = polytropicKRatio787(sink_rate, mass, k_current, n_poly, arm_ratio)
    if nargin < 4, n_poly    = 1.35; end
    if nargin < 5, arm_ratio = 0.55; end
    k_eff_bogie = k_current / (arm_ratio^2);
    stroke_est  = min(0.635, sqrt(mass / max(k_eff_bogie, 10000)) * sink_rate);
    frac  = min(0.80, stroke_est / 0.635);
    ratio = max(1.0, min(2.0, (1.0 / (1.0 - frac))^n_poly));
end

function ratio = orificeB787(sink_rate)
    ratio = max(0.50, min(2.50, sink_rate / 2.0));
end

function F_seal = sealFriction787(v_strut, health)
    F_static  = 8000  * (1 + 0.50 * (1 - health));
    F_coulomb = 4000  * (1 + 0.80 * (1 - health));
    v_strut   = max(abs(v_strut), 0.001);
    F_seal    = max(0, F_coulomb + (F_static - F_coulomb) * exp(-v_strut / 0.05));
end

function angle = bogieAngle(class_id, health, fault_angle_in)
    if class_id == 12
        angle = max(0.0, fault_angle_in);
    else
        angle = min(1.5, 0.5*(1-health) + 0.3*rand() + max(0, fault_angle_in));
    end
end

function act_load = actuatorLoad787(act_fric_in, sink_rate, health)
    base_force    = 800 + 600 * (1 - health);
    friction_comp = act_fric_in * (0.8 + 0.4*rand());
    pressure_comp = 200 * (1 + 0.2*sink_rate);
    act_load = max(500, base_force + friction_comp + pressure_comp);
end

function stroke = bogieStroke787(sink_rate, mass, k_current, arm_ratio)
    if nargin < 4, arm_ratio = 0.55; end
    stroke = min(0.635, sqrt(mass / max(k_current/(arm_ratio^2), 10000)) * sink_rate);
end

function delay = WOWdebounce(delay_ms)
    delay = max(5.0, min(300.0, delay_ms));
end

function E_MJ = brakeEnergy787(mass, sink_rate, v_approach_ms)
    E_MJ = max(0, min(500, (0.5*mass*v_approach_ms^2 + mass*9.81*sink_rate*5) / 1e6));
end

function [kt, bt] = therm787(k0, b0, tc)
    Tk  = tc + 273.15;
    kt  = k0 * (Tk / 288.15);
    vis = exp(2500 * (1/Tk - 1/288.15));
    bt  = b0 * max(0.3, min(3.5, vis));
    if tc < 0
        kt = kt * min(1.24, 1 + 0.008*abs(tc));
        bt = bt * min(1.15, 1 + 0.005*abs(tc));
    elseif tc > 40
        bt = bt * max(0.90, 1 - 0.005*(tc-40));
    end
end

function kn = kNorm787(k, tc)
    kn = k / ((tc + 273.15) / 288.15);
end

function [sev, tr, mi] = assess787(cID, sink, gp, health, act_f, b_ang, wow_d, REG, MAINT) %#ok<INUSL>
    sev=1; tr=0; mi=1;
    if sink>=REG.SinkHardLanding || gp>=REG.AccelLimit
        sev=3; tr=1; mi=9;
    end
    if gp>=REG.AccelExtreme || sink>4.5
        sev=4; tr=1; mi=11;
    end
    if sink>=REG.SinkCaution && sev<2
        sev=2; tr=1; mi=max(mi, min(cID+1,15));
    end
    if health<0.30,      sev=max(sev,4); tr=1; mi=11;
    elseif health<0.50,  sev=max(sev,3); tr=1; if mi<9, mi=10; end
    elseif health<0.75,  sev=max(sev,2); tr=1; if mi==1, mi=min(cID+1,15); end
    elseif cID>0,        sev=max(sev,2); tr=1; if mi==1, mi=min(cID+1,15); end
    end
    if cID==12 && b_ang>3.0
        sev=max(sev,2); tr=1; mi=13;
        if b_ang>8.0, sev=max(sev,3); end
    end
    mi = max(1, min(mi, 15));
end

function dp = dmgCalc787(sink, mass, mxD, tc, F)
    geq   = max(1.0, min(6.0, 1 + (sink^2)/(2*9.81*max(mxD,0.005))));
    stress= geq * mass * 9.81 / 0.15;
    if stress <= F.S_e, dp=0; return; end
    Nf    = max(1, F.N_ref * (F.S_ut/stress)^(1/abs(F.b)));
    dp    = 1/Nf;
    if tc>40,  dp=dp*(1+0.015*(tc-40));  end
    if tc<-20, dp=dp*(1+0.010*abs(tc+20)); end
end

function c = mCost787(tr, sev, mi, MAINT) %#ok<INUSL>
    if ~tr, c=0; return; end
    c = MAINT(mi).cost;
end

function cf = fixedCost787(cc, interval, sev, mi, MAINT, ECON, cf)
    if mod(cc, interval)==0
        if sev>=3, cf=cf+MAINT(mi).cost;
        else,      cf=cf+ECON.Routine; end
    end
end

function row = bRow787(id, cls, hf, mass, sink, fric, tc, k, b, kn, bn, ...
                        dw, vw, fw, aw, pw, bgw, ...
                        F_seal, act_f, b_ang, wow_d, brk_e, ...
                        bpeak, alpeak_comp, hl_flag, ...
                        tr, sev, dp, rul, c, mi)
    row = [id, cls, rul, mass, sink, fric, tc, k, b, kn, bn, hf, ...
           max(abs(dw)), max(abs(vw)), max(abs(aw))/9.81, ...
           max(abs(fw)), max(abs(pw)), F_seal, ...
           bpeak, alpeak_comp, ...
           tr, sev, mi, ...
           b_ang, act_f, brk_e, 0, wow_d, bpeak, alpeak_comp, hl_flag, 0, ...
           dw, vw, fw, aw, pw, bgw];
end

function pProg787(cs, tgt, sc, tot, t0)
    e    = toc(t0);
    r    = e / max(sc, 1);
    eta_h= (r * (tot - sc)) / 3600;
    fprintf('   %d/%d | Total: %d/%d | %.2fs/sim | ETA: %.1fh\n', ...
        cs, tgt, sc, tot, r, eta_h);
end