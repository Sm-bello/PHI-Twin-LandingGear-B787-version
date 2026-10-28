% =========================================================================
%   🏭 AEROTWIN DATA FACTORY V4.5 — BOEING 787-9 DREAMLINER
%   Engine   : Pure MATLAB ODE Synthesis (NO SIMSCAPE REQUIRED)
%   Purpose  : 18,000 physics-based landing events for CNN-LSTM training
%
%   V4.5 UPGRADES:
%   ► SPEED: Solves 18,000 landings in ~10 minutes using pure math.
%   ► AI LOGIC: Health is sorted Fresh -> Degraded across the batch.
%   ► RUL FIX: Uses Time-To-Failure (TTF_BASE) in Flight Hours.
%   ► BOGIE FIX: Class 12 correctly scales pitch fault from 3° to 12°.
%   ► PLOTS: 9-Subplot Dashboard restored.
% =========================================================================

clear; clc; close all;
fprintf('🏭 AEROTWIN V4.5 — BOEING 787-9 | PURE ODE ENGINE | TTF RUL\n');
fprintf('====================================================================\n');
fprintf('Engine   : Direct ODE45 synthesis (No Simscape/Simulink)\n');
fprintf('Dataset  : 18,000 samples | 12 classes | 6 signals | 632 columns\n');
fprintf('Logic    : Health Sorted (Fresh->Degraded) + TTF Flight Hours\n');
fprintf('Runtime  : ~10 to 15 minutes total\n');
fprintf('====================================================================\n\n');

% =========================================================================
%   SECTION 1: REGULATORY THRESHOLDS & RUL BASELINES
% =========================================================================
REG.SinkDesign       = 3.05;
REG.SinkHardLanding  = 3.05;
REG.SinkCaution      = 2.50;
REG.AccelLimit       = 2.00;
REG.AccelExtreme     = 2.67;
REG.MTOW             = 254000;
REG.MLW              = 193000;
REG.FixedInterval    = 1000;
REG.SimStopTime      = 4.0;

TTF_BASE = containers.Map( ...
    {0,    1,   2,   3,   4,   5,   6,    7,    8,   9,  10,  12}, ...
    {2000, 500, 300, 900, 600, 250,   5, 1200,  50, 180,   2, 400} ...
);
SEV_MULT = [1.00, 0.65, 0.30, 0.08];

% =========================================================================
%   SECTION 2: MAINTENANCE KNOWLEDGE BASE
% =========================================================================
MAINT( 1).chapter='ATA 32-10'; MAINT( 1).category='Routine Line Check'; MAINT( 1).cost=850;
MAINT( 2).chapter='ATA 32-10'; MAINT( 2).category='Nitrogen System Service'; MAINT( 2).cost=2200;
MAINT( 3).chapter='ATA 32-10'; MAINT( 3).category='Hydraulic Damper Service'; MAINT( 3).cost=4500;
MAINT( 4).chapter='ATA 32-10'; MAINT( 4).category='Degradation Trending Inspection'; MAINT( 4).cost=5800;
MAINT( 5).chapter='ATA 32-10'; MAINT( 5).category='Thermal Stress Inspection'; MAINT( 5).cost=4800;
MAINT( 6).chapter='ATA 32-10'; MAINT( 6).category='Asymmetric Load / Bogie Inspection'; MAINT( 6).cost=7500;
MAINT( 7).chapter='ATA 32-40'; MAINT( 7).category='Wheel and Brake Assembly Inspection'; MAINT( 7).cost=12000;
MAINT( 8).chapter='ATA 32-10'; MAINT( 8).category='Corrosion Control Treatment'; MAINT( 8).cost=8500;
MAINT( 9).chapter='ATA 32-00'; MAINT( 9).category='Hard Landing Inspection [CS-25.473]'; MAINT( 9).cost=28000;
MAINT(10).chapter='ATA 32-10'; MAINT(10).category='Component Overhaul'; MAINT(10).cost=65000;
MAINT(11).chapter='ATA 32-00'; MAINT(11).category='Emergency AOG Maintenance'; MAINT(11).cost=185000;
MAINT(12).chapter='ATA 32-31'; MAINT(12).category='Retract Actuator Inspection [Layer 3]'; MAINT(12).cost=15000;
MAINT(13).chapter='ATA 32-10'; MAINT(13).category='Bogie Beam Structural Inspection [Class 12]'; MAINT(13).cost=18000;
MAINT(14).chapter='ATA 32-61'; MAINT(14).category='WOW Proximity Sensor Rigging [Layer 3]'; MAINT(14).cost=6500;
MAINT(15).chapter='ATA 32-32'; MAINT(15).category='Lock Assembly Inspection [Layer 3]'; MAINT(15).cost=12500;

ECON.Routine=850; ECON.Detailed=5800; ECON.HardLanding=28000; 
ECON.AOG=185000; ECON.PrevOverhaul=65000; ECON.Emergency=185000;
FAT.S_ut=1950e6; FAT.S_e=0.5*1950e6; FAT.b=-0.12; FAT.N_ref=1e6;

% =========================================================================
%   SECTION 3: PHYSICS CONSTANTS
% =========================================================================
MASS_NORM_MIN=150000; MASS_NORM_MAX=193000; MASS_MTOW=254000;
K_NOM=1500000; B_NOM=60000; STROKE_LIMIT=0.635; BOGIE_ARM_RATIO=0.55;
SINK_MIN=0.3; SINK_MAX=3.5; F_DRY=0.80; F_WET=0.40; F_ICE=0.20;
T_ISA=15; T_HOT=45; T_COLD=-30;
K_CONTACT=1.0e6; B_CONTACT=5.0e4; P0=20.0e6; V0=0.012; A_PISTON=0.018;
N_POLY=1.35; g_acc=9.81; T_LAND=4.0; N_PTS=100;

% =========================================================================
%   SECTION 4: PRE-FLIGHT PHYSICS VALIDATION
% =========================================================================
fprintf('🔍 Pre-flight physics validation...\n');
[dw,vw,fw,aw,pw,bgw] = odeLanding787(...
    K_NOM, B_NOM, 180000, 1.8, F_DRY, T_ISA, 1.0, 0.0, ...
    K_CONTACT, B_CONTACT, P0, V0, A_PISTON, N_POLY, BOGIE_ARM_RATIO, ...
    STROKE_LIMIT, g_acc, T_LAND, N_PTS, false);
gPk = max(abs(aw)) / g_acc;
if gPk < 0.1 || gPk > 5.0 || max(abs(dw)) > STROKE_LIMIT*1.5
    fprintf('❌ Physics validation FAILED. Check ODE constants.\n');
    return;
end
fprintf('✅ ODE Engine Nominal. Peak Accel: %.2fg\n\n', gPk);

% =========================================================================
%   SECTION 5: STORAGE INITIALISATION
% =========================================================================
projPath = 'C:\Users\User\Desktop\B787_AeroTwin';
if ~exist(projPath,'dir'), mkdir(projPath); end
outPath = fullfile(projPath, 'AEROTWIN_B787_V4p5_Dataset.csv');

TOTAL = 18000;
DS = zeros(TOTAL, 632); 
META = zeros(TOTAL, 8); 
MLOG = cell(TOTAL, 1);
dmg = zeros(1,13); 
costCBM = 0; costFIXED = 0; aogSaved = 0; sc = 0; cc = 0;
t0 = tic; rng(42);

TARGETS = [3500, 2200, 2200, 1600, 1400, 1100, 900, 1100, 900, 800, 700, 1600];
CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12];
classNames = {'Normal','N2 Leak','Worn Seal','Early Degrad','Thermal Degrad',...
              'Brake Fade','Tire Burst','Corrosion','Hard Landing',...
              'N2+Seal','Impending AOG','','Bogie Pitch'};

% =========================================================================
%   MAIN BATCH LOOP
% =========================================================================
for ci_idx = 1:length(CLASS_IDS)
    cID = CLASS_IDS(ci_idx);
    tgt = TARGETS(ci_idx);
    cName = classNames{min(cID+1,13)};
    fprintf('--- CLASS %2d: %-20s (%d samples) ---\n', cID, cName, tgt);
    
    cs = 0;
    while cs < tgt
        % 1. Calculate Degradation Fraction (0.0 to 1.0)
        degrad_frac = cs / max(tgt-1, 1);
        
        % 2. Inject Sorted AI Logic based on Class
        bogie_fault_in = 0.0;
        
        switch cID
            case 0 % Normal
                health = 1.00 - (degrad_frac * 0.15);
                health = max(0.85, min(1.00, health + 0.02 * randn()));
            case 1 % N2 Leak
                health = 1.00 - degrad_frac;
                health = max(0.01, min(1.00, health + 0.02 * randn()));
            case 2 % Worn Seal
                health = 1.00 - degrad_frac;
                health = max(0.01, min(1.00, health + 0.02 * randn()));
            case 3 % Early Degrad
                health = 0.85 - (degrad_frac * 0.15);
                health = max(0.70, min(0.85, health + 0.01 * randn()));
            case 4 % Thermal Degrad
                health = 0.90 - (degrad_frac * 0.30);
                health = max(0.60, min(0.90, health + 0.02 * randn()));
            case 5 % Brake Fade
                health = 0.95 - (degrad_frac * 0.20);
                health = max(0.75, min(0.95, health + 0.02 * randn()));
            case 6 % Tire Burst
                health = 0.60 - (degrad_frac * 0.20);
                health = max(0.40, min(0.60, health + 0.02 * randn()));
            case 7 % Corrosion
                health = 1.00 - (degrad_frac * 0.50);
                health = max(0.50, min(1.00, health + 0.02 * randn()));
            case 8 % Hard Landing
                health = 0.80 - (degrad_frac * 0.40);
                health = max(0.40, min(0.80, health + 0.02 * randn()));
            case 9 % Combined N2+Seal
                health = 0.80 - (degrad_frac * 0.60);
                health = max(0.20, min(0.80, health + 0.02 * randn()));
            case 10 % AOG
                health = 0.30 - (degrad_frac * 0.25);
                health = max(0.05, min(0.30, health + 0.02 * randn()));
            case 12 % Bogie Pitch
                health = 0.95 - (degrad_frac * 0.45);
                health = max(0.50, min(0.95, health + 0.02 * randn()));
                bogie_fault_in = 3.0 + (degrad_frac * 9.0) + 0.5*randn(); % 3 to 12 degrees
        end
        
        % 3. Generate Physics Parameters
        tc = genTemp(cID, T_ISA, T_HOT, T_COLD);
        mass = genMass(cID, MASS_NORM_MIN, MASS_NORM_MAX, MASS_MTOW);
        sink = genSink(cID, SINK_MIN, SINK_MAX, REG.SinkDesign);
        [kt, bt] = therm787(K_NOM, B_NOM, tc);
        [k, b] = genKB(cID, kt, bt, K_NOM, B_NOM, health, sink, BOGIE_ARM_RATIO, N_POLY);
        
        fric_base = genFric(cID, F_DRY, F_WET, F_ICE);
        fric = tireFriction787(fric_base, sink, tc);
        F_seal = sealFriction787(sink, health);
        act_f = actuatorLoad787(genActFric(cID), sink, health);
        b_ang = bogieAngle(cID, health, bogie_fault_in);
        wow_d = WOWdebounce(genWOW(cID));
        brk_e = brakeEnergy787(mass, sink, 70+10*rand()) * genBrkMult(cID);

        % Apply velocity-squared damping to classes 6 & 8
        if cID == 6 || cID == 8, b = b * orificeB787(sink); end

        % 4. Run Fast ODE
        try
            [dw,vw,fw,aw,pw,bgw] = odeLanding787(...
                k, b, mass, sink, fric, tc, health, b_ang, ...
                K_CONTACT, B_CONTACT, P0, V0, A_PISTON, N_POLY, BOGIE_ARM_RATIO,...
                STROKE_LIMIT, g_acc, T_LAND, N_PTS, cID==12);
                
            gp = max(abs(aw)) / g_acc;
            if max(abs(dw)) < 0.001 || any(isnan(dw)), continue; end
            
            sc = sc + 1; cc = cc + 1;
            kn = kNorm787(k, tc);
            [sev,tr,mi] = assess787(cID,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT);
            
            if cID==10 && tr && sev==4, aogSaved=aogSaved+1; end
            
            % 5. Calculate Sorted TTF RUL (Flight Hours)
            rul_sev = min(4, 1 + floor(degrad_frac * 3));
            rul = TTF_BASE(cID) * SEV_MULT(rul_sev) * (1 + 0.08 * randn());
            rul = max(0.5, rul);
            
            % Miner Damage (for economic calculation only)
            dp_mult = getDmgMult(cID);
            dp = dmgCalc787(sink,mass,max(abs(dw)),tc,FAT) * dp_mult;
            dmg(min(cID+1,13)) = dmg(min(cID+1,13)) + dp;
            
            c = mCost787(tr,sev,mi,MAINT); costCBM = costCBM + c;
            costFIXED = fixedCost787(cc,REG.FixedInterval,sev,mi,MAINT,ECON,costFIXED);
            hl = double(sink>=REG.SinkHardLanding || gp>=REG.AccelLimit);
            
            scalars = [sc, cID, rul, mass, sink, fric, tc, k, b, kn, b,...
                       health, max(abs(dw)), max(abs(vw)), gp,...
                       max(abs(fw)), max(abs(pw)), F_seal,...
                       max(abs(bgw)), act_f,...
                       tr, sev, mi,...
                       b_ang, act_f, brk_e, 0, wow_d,...
                       max(abs(bgw)), act_f, hl, 0];
            DS(sc,:)   = [scalars, dw, vw, fw, aw, pw, bgw];
            META(sc,:) = [sc, cID, tc, sink, fric, act_f, b_ang, rul];
            MLOG{sc}   = {sc, cID, MAINT(mi).chapter, MAINT(mi).category, sev, tr, c, rul};
            
            cs = cs + 1;
            
        catch
            continue;
        end
        
        if mod(cs, round(tgt/4))==0 && cs>0
            e=toc(t0); r=e/max(sc,1);
            fprintf('   %d/%d | %.3fs/sim | ETA: %.1f min\n', cs, tgt, r, r*(TOTAL-sc)/60);
        end
    end
end

% =========================================================================
%   SECTION 6: SAVE & PLOT
% =========================================================================
totalTime = toc(t0);
DS   = DS(1:sc, :);
META = META(1:sc, :);
MLOG = MLOG(1:sc);

vn = {'SampleID','Class','RUL_Hours','Mass_kg','SinkRate_ms',...
      'Friction_Coef','Temp_C','K_Stiffness_Nm','B_Damping_Nsm',...
      'K_Normalized','B_Normalized','Health_Factor',...
      'Max_Deflection_m','Max_Velocity_ms','Max_Accel_g',...
      'Max_Force_N','Max_Pressure_Pa','Seal_Force_N',...
      'Max_Bogie_Angle_deg','Actuator_Load_N',...
      'Maint_Trigger','Severity','ATA_Chapter_Index',...
      'Bogie_Init_Angle_deg','Actuator_Fric_N','Brake_Energy_MJ',...
      'Gear_Cmd_Mode','WOW_Delay_ms','Bogie_Peak_Angle_deg',...
      'Actuator_Peak_Load_N','CS25_HardLanding_Flag','Retract_Flag'};
for sg = {'Defl','Velo','Force','Accel','Pressure','BogieAngle'}
    for t = 1:100, vn{end+1} = sprintf('%s_t%d', sg{1}, t); end
end
T = array2table(DS, 'VariableNames', vn);
writetable(T, outPath);

writetable(array2table(META, 'VariableNames',...
    {'SampleID','Class','Temp_C','SinkRate','Friction','ActuatorFric','BogieAngle','RUL_Hours'}),...
    fullfile(projPath,'AEROTWIN_B787_V4p5_Metadata.csv'));

MLOG_valid = MLOG(~cellfun(@isempty, MLOG));
mL  = vertcat(MLOG_valid{:});
mLT = cell2table(mL, 'VariableNames',...
    {'SampleID','Class','ATA_Chapter','Category','Severity','Triggered','Cost_USD','RUL_Hours'});
writetable(mLT, fullfile(projPath,'AEROTWIN_B787_V4p5_MaintenanceLog.csv'));

n = height(T);
train_idx=[]; val_idx=[]; test_idx=[];
for ci = CLASS_IDS
    idx_c = find(T.Class == ci);
    idx_c = idx_c(randperm(length(idx_c)));
    n1 = round(0.70*length(idx_c)); n2 = round(0.85*length(idx_c));
    train_idx = [train_idx; idx_c(1:n1)];
    val_idx   = [val_idx;   idx_c(n1+1:n2)];
    test_idx  = [test_idx;  idx_c(n2+1:end)];
end
sp.train_idx    = train_idx; sp.val_idx      = val_idx; sp.test_idx     = test_idx;
sp.class_labels = {0,1,2,3,4,5,6,7,8,9,10,12};
sp.signal_names = {'Deflection','Velocity','Force','Acceleration','Pressure','BogieAngle'};
sp.n_signals    = 6; sp.n_timesteps  = 100; sp.n_classes    = 12;
sp.rul_column   = 'RUL_Hours'; sp.rul_max      = 2000.0;
save(fullfile(projPath,'train_val_split_B787_V4p5.mat'), 'sp');

sav = costFIXED - costCBM; savPct = 100*sav/max(costFIXED,1);

% RUL DISTRIBUTION
fprintf('\n====================================================================\n');
fprintf('V4.5 RUL DISTRIBUTION — VERIFY AI LOGIC\n');
fprintf('====================================================================\n');
fprintf('Class | Name                | TTF_Base | min(h)  max(h)  mean(h)  std(h)\n');
fprintf('------+---------------------+----------+--------------------------------\n');
for ci=1:12
    cid=CLASS_IDS(ci); rows=T.RUL_Hours(T.Class==cid);
    if isempty(rows), continue; end
    fprintf('%5d | %-19s | %8d | %7.1f  %7.1f  %7.1f  %6.1f \n',...
        cid,classNames{min(cid+1,13)},TTF_BASE(cid),min(rows),max(rows),mean(rows),std(rows));
end

fprintf('\n====================================================================\n');
fprintf('✅ AEROTWIN B787 V4.5 COMPLETE — PURE ODE SYNTHESIS\n');
fprintf('====================================================================\n');
fprintf('📊 Samples  : %d | Columns: %d\n', n, width(T));
fprintf('⏱️  Time     : %.1f minutes\n', totalTime/60);
fprintf('💰 SAVINGS  : $%14.0f (%.1f%%)\n', sav, savPct);
fprintf('🎓 NEXT STEP: Python -> y_rul = df[''RUL_Hours''].values / 2000.0\n');
fprintf('====================================================================\n');

% --- PLOTS ---
figure('Name','AEROTWIN B787 V4.5 — ODE Dataset Analysis','Position',[30 30 1800 1000]);
t_ax = linspace(0,4,N_PTS);
subplot(3,3,1);
cnts=arrayfun(@(c)sum(T.Class==c),CLASS_IDS); bar(1:12,cnts,'FaceColor',[0.2 0.5 1.0]);
set(gca,'XTick',1:12,'XTickLabel',arrayfun(@(c)sprintf('C%d',c),CLASS_IDS,'uni',0));
title('Class Distribution (12 classes)'); grid on;

subplot(3,3,2);
scatter(T.SinkRate_ms,T.Max_Accel_g,6,T.Class,'filled'); hold on;
yline(REG.AccelLimit,'r--','LineWidth',2); yline(REG.AccelExtreme,'m--','LineWidth',1.5);
xline(REG.SinkHardLanding,'k--','LineWidth',2);
title('CS-25.473 Hard Landing Envelope'); xlabel('Sink (m/s)'); ylabel('Peak Accel (g)'); colormap(jet); grid on;

subplot(3,3,3);
scatter(T.K_Stiffness_Nm,T.B_Damping_Nsm,6,T.Class,'filled'); hold on;
title('K-B Feature Space (12 classes)'); xlabel('K (N/m)'); ylabel('B (N·s/m)'); colormap(jet); grid on;

subplot(3,3,4);
idx0=find(T.Class==0,1,'first'); idx1=find(T.Class==1,1,'first'); idx10=find(T.Class==10,1,'first');
if ~isempty(idx0) && ~isempty(idx1) && ~isempty(idx10)
    plot(t_ax,T{idx0,33:132},'b-','LineWidth',1.8); hold on;
    plot(t_ax,T{idx1,33:132},'g--','LineWidth',1.8);
    plot(t_ax,T{idx10,33:132},'r:','LineWidth',1.8);
    title('Deflection: C0 vs C1(N2) vs C10(Fail)'); xlabel('Time (s)'); ylabel('Deflection (m)'); grid on;
end

subplot(3,3,5);
idx12=find(T.Class==12,1,'first');
if ~isempty(idx0) && ~isempty(idx12)
    plot(t_ax,T{idx0,233:332},'b-','LineWidth',1.8); hold on;
    plot(t_ax,T{idx12,233:332},'r--','LineWidth',1.8);
    title('Force: C0 vs C12 (bogie double-peak)'); xlabel('Time (s)'); ylabel('Force (N)'); grid on;
end

subplot(3,3,6);
if ~isempty(idx0) && ~isempty(idx12)
    plot(t_ax,T{idx0,533:632},'b-','LineWidth',1.8); hold on;
    plot(t_ax,T{idx12,533:632},'r--','LineWidth',1.8);
    title('BogieAngle: C0 vs C12 — KEY DISCRIMINATOR'); xlabel('Time (s)'); ylabel('Bogie Angle (deg)'); grid on;
end

subplot(3,3,7);
ba_mean=arrayfun(@(c)mean(T.Bogie_Init_Angle_deg(T.Class==c)),CLASS_IDS);
bar(1:12,ba_mean,'FaceColor',[0.3 0.7 0.4]);
set(gca,'XTick',1:12,'XTickLabel',arrayfun(@(c)sprintf('C%d',c),CLASS_IDS,'uni',0));
title('Mean Bogie Init Angle by Class'); xlabel('Class'); ylabel('Angle (deg)'); grid on;

subplot(3,3,8);
trigR=arrayfun(@(c)100*sum(T.Class==c&T.Maint_Trigger==1)/max(sum(T.Class==c),1),CLASS_IDS);
bar(1:12,trigR,'FaceColor',[0.85 0.35 0.25]);
set(gca,'XTick',1:12,'XTickLabel',arrayfun(@(c)sprintf('C%d',c),CLASS_IDS,'uni',0));
title('Maintenance Trigger Rate (%)'); xlabel('Class'); ylabel('%'); yline(50,'k--'); grid on;

subplot(3,3,9);
bar([costFIXED/1e6,costCBM/1e6],'FaceColor','flat','CData',[0.85 0.25 0.25;0.25 0.75 0.35]);
set(gca,'XTickLabel',{'Fixed-Interval','CBM AI'}); ylabel('Total Cost ($M)');
title(sprintf('Savings: $%.1fM (%.0f%%)',sav/1e6,savPct)); grid on;
saveas(gcf,fullfile(projPath,'AEROTWIN_B787_V4p5_Analysis.png'));

% =========================================================================
%   CORE ODE ENGINE
% =========================================================================
function [dw,vw,fw,aw,pw,bgw] = odeLanding787(k, b, mass, sink, fric, tc, health, b_ang,...
    K_c, B_c, P0, V0, A_p, n_poly, arm_ratio, stroke_lim, g, T_end, N, is_bogie_fault)

    function dxdt = gearODE(~, x)
        defl = x(1); vel = x(2);
        penetration = max(0, defl);
        F_contact = K_c * penetration^1.5 + B_c * max(0, vel) * penetration;
        frac = min(0.80, abs(defl) / stroke_lim);
        k_poly = k * max(1.0, min(2.0, (1/(1-frac))^n_poly));
        k_eff = k_poly / (arm_ratio^2);
        F_spring = k_eff * defl;
        F_damp = b * vel;
        F_s = F_seal_inline(vel, health);
        bogie_rad = deg2rad(b_ang);
        F_bogie_corr = F_spring * (1 - cos(bogie_rad)) * 0.15;
        F_grav = mass * g;
        accel = (F_contact - F_spring - F_damp - F_s - F_bogie_corr - F_grav) / mass;
        dxdt = [vel; accel];
    end

    function Fs = F_seal_inline(v, h)
        F_static = 8000*(1 + 0.50*(1-h));
        F_coulomb = 4000*(1 + 0.80*(1-h));
        v_abs = max(abs(v), 0.001);
        Fs = F_coulomb + (F_static-F_coulomb)*exp(-v_abs/0.05);
        Fs = Fs * sign(v + 1e-10);
    end

    x0 = [0.001; -sink];
    opts = odeset('RelTol',1e-3,'AbsTol',1e-4,'MaxStep',0.02);
    [t_sol, X] = ode45(@gearODE, [0, T_end], x0, opts);
    
    if length(t_sol) < 10
        t_sol = linspace(0, T_end, 500)';
        omega = sqrt(k/mass); zeta = b/(2*sqrt(k*mass)); zeta = min(zeta, 0.99);
        wd = omega*sqrt(1-zeta^2);
        defl_ana = (sink/wd)*exp(-zeta*omega*t_sol).*sin(wd*t_sol);
        vel_ana = gradient(defl_ana, t_sol);
        X = [defl_ana, vel_ana];
    end

    t_uni = linspace(0, T_end, N);
    t_norm= linspace(0, 1, length(t_sol));
    t_q = linspace(0, 1, N);
    defl_r = interp1(t_norm, X(:,1), t_q, 'pchip');
    vel_r = interp1(t_norm, X(:,2), t_q, 'pchip');
    accel_r = gradient(vel_r) / (T_end/(N-1));
    force_r = mass * (accel_r + g);
    frac_r = min(0.79, abs(defl_r) / stroke_lim);
    press_r = P0 * (1 ./ max(1-frac_r, 0.01)).^1.35;
    
    if is_bogie_fault
        [~, peak_idx] = max(abs(force_r));
        bgw_sig = zeros(1, N);
        bgw_sig = bgw_sig + b_ang * exp(-((1:N)-peak_idx).^2 / (N*0.005));
        sec_idx = min(peak_idx + round(N*0.08), N);
        bgw_sig = bgw_sig + b_ang*0.6 * exp(-((1:N)-sec_idx).^2 / (N*0.003));
        bgw_sig = bgw_sig + 0.05*randn(1,N);
    else
        bgw_sig = b_ang * exp(-t_uni/0.8) + 0.02*randn(1,N);
    end

    noise_scale = 0.005;
    dw = defl_r + noise_scale * max(abs(defl_r)) * randn(1,N);
    vw = vel_r + noise_scale * max(abs(vel_r)) * randn(1,N);
    fw = force_r + noise_scale * max(abs(force_r)) * randn(1,N);
    aw = accel_r + noise_scale * max(abs(accel_r)) * randn(1,N);
    pw = press_r + noise_scale * max(abs(press_r)) * randn(1,N);
    bgw = bgw_sig;
end

% =========================================================================
%   HELPER FUNCTIONS
% =========================================================================
function tc = genTemp(cID, T_ISA, T_HOT, T_COLD)
    if cID == 4, tc = 35 + (T_HOT-35)*rand();
    elseif cID == 7, tc = 20 + 15*rand();
    else, tc = T_COLD + (T_HOT-T_COLD)*rand(); end
end
function mass = genMass(cID, MASS_MIN, MASS_MAX, MASS_MTOW)
    if cID == 8, mass = 170000 + (MASS_MTOW-170000)*rand();
    elseif cID == 6, mass = MASS_MAX*(0.80 + 0.30*rand());
    else, mass = MASS_MIN + (MASS_MAX-MASS_MIN)*rand(); end
end
function sink = genSink(cID, SINK_MIN, SINK_MAX, SinkDesign)
    if cID == 8, sink = 3.0 + (SinkDesign-3.0)*rand();
    elseif cID == 6, sink = 2.0 + (3.5-2.0)*rand();
    elseif cID == 0
        if rand() < 0.8, sink = 0.3 + (2.0-0.3)*rand(); else, sink = 2.0 + (SinkDesign-2.0)*rand(); end
    else, sink = SINK_MIN + (SINK_MAX-SINK_MIN)*rand(); end
end
function [k, b] = genKB(cID, kt, bt, K_NOM, B_NOM, health, sink, arm, npoly)
    function kp = polyCorr(k_in, s, m_est)
        frac = min(0.80, sqrt(m_est/max(k_in/(arm^2),10000))*s / 0.635);
        kp = k_in * max(1.0, min(2.0, (1/(1-frac))^npoly));
    end
    m_est = 180000;
    switch cID
        case 0, k=kt*(0.953+0.094*rand()); b=bt*(0.950+0.100*rand());
        case 1, k=kt*((750000+400000*health)/K_NOM); b=bt*(0.950+0.100*rand());
        case 2, k=kt*(0.953+0.094*rand()); b=bt*((13500+16500*health)/B_NOM);
        case 3, k=kt*((1260000+1000000*(health-0.70))/K_NOM); b=bt*((51000+40000*(health-0.70))/B_NOM);
        case 4, k=kt*((1380000+140000*health)/K_NOM)*(0.97+0.06*rand()); b=bt*((39000+15000*health)/B_NOM)*(0.92+0.16*rand());
        case 5, k=kt*(0.953+0.094*rand()); b=bt*((21000+63000*rand())/B_NOM);
        case 6, k=kt*((780000+270000*rand())/K_NOM); b=bt*((90000+37500*rand())/B_NOM);
        case 7, k=kt*((1050000+210000*health)/K_NOM)*(0.97+0.06*rand()); b=bt*((66000+18000*(1-health))/B_NOM)*(0.95+0.10*rand());
        case 8, k=kt*((1120000+260000*health)/K_NOM); b=bt*((45000+27000*rand())/B_NOM);
        case 9, k=kt*((650000+400000*health)/K_NOM); b=bt*((10500+19500*health)/B_NOM);
        case 10,k=kt*((500000+300000*health)/K_NOM); b=bt*((7500+15000*health)/B_NOM);
        case 12,k=kt*((1300000+200000*rand())/K_NOM); b=bt*(0.92+0.16*rand());
        otherwise, k=kt*(0.953+0.094*rand()); b=bt*(0.950+0.100*rand());
    end
    k = polyCorr(k, sink, m_est);
end
function fb = genFric(cID, F_DRY, F_WET, F_ICE)
    if cID == 4, fb = F_DRY;
    elseif cID == 0, fb = randsample([F_DRY,F_WET,F_ICE],1,true,[0.70,0.25,0.05]);
    elseif cID == 9 || cID == 10, fb = randsample([F_DRY,F_WET,F_ICE],1,true,[0.40,0.40,0.20]);
    elseif cID == 6 || cID == 8, fb = randsample([F_DRY,F_WET],1,true,[0.70,0.30]);
    elseif cID == 7, fb = randsample([F_DRY,F_WET],1,true,[0.50,0.50]);
    else, fb = randsample([F_DRY,F_WET,F_ICE],1,true,[0.60,0.30,0.10]); end
end
function af = genActFric(cID)
    if cID == 6 || cID == 8, af = 1200+600*rand();
    elseif cID == 10, af = 2000+1000*rand();
    else, af = 800+400*rand(); end
end
function wd = genWOW(cID)
    if cID == 10, wd = 20+50*rand(); else, wd = 10+20*rand(); end
end
function m = genBrkMult(cID)
    switch cID, case 4, m=1.2+0.3*rand(); case 5, m=1.3+0.4*rand(); case 8, m=1.4; otherwise, m=1.0; end
end
function m = getDmgMult(cID)
    switch cID, case 8, m=1.5; case 10, m=2.5; case 6, m=1.2; case 12, m=0.8; otherwise, m=1.0; end
end
function mu = tireFriction787(mu_base, sink_rate, temp_c)
    v_td = max(55.0, min(82.0, 77.0-sink_rate*2.5)); v_hydro = 65.4;
    if mu_base < 0.60, hydro_factor = max(0.50, 1.0-0.35*max(0,(v_td-v_hydro)/max(82-v_hydro,1)));
    else, hydro_factor = 1.0; end
    if temp_c < 0, ice_factor = max(0.15, 1.0-0.010*abs(temp_c)); else, ice_factor = 1.0; end
    mu = max(0.08, min(0.92, mu_base*hydro_factor*ice_factor));
end
function ratio = polytropicKRatio787(sink_rate, mass, k_current, n_poly, arm_ratio)
    k_eff = k_current/(arm_ratio^2); stroke = min(0.635, sqrt(mass/max(k_eff,10000))*sink_rate);
    frac = min(0.80, stroke/0.635); ratio = max(1.0, min(2.0, (1/(1-frac))^n_poly));
end
function ratio = orificeB787(sink_rate), ratio = max(0.50, min(2.50, sink_rate/2.0)); end
function F_seal = sealFriction787(v_strut, health)
    F_static = 8000*(1+0.50*(1-health)); F_coulomb = 4000*(1+0.80*(1-health));
    F_seal = max(0, F_coulomb+(F_static-F_coulomb)*exp(-max(abs(v_strut), 0.001)/0.05));
end
function angle = bogieAngle(class_id, health, fault_in)
    if class_id==12, angle = max(0, fault_in); else, angle = min(1.5, 0.5*(1-health)+0.3*rand()+max(0,fault_in)); end
end
function act_load = actuatorLoad787(act_fric_in, sink_rate, health)
    act_load = max(500, 800+600*(1-health) + act_fric_in*(0.8+0.4*rand()) + 200*(1+0.2*sink_rate));
end
function delay = WOWdebounce(delay_ms), delay = max(5.0, min(300.0, delay_ms)); end
function E_MJ = brakeEnergy787(mass, sink_rate, v_approach_ms)
    E_MJ = max(0, min(500, (0.5*mass*v_approach_ms^2 + mass*9.81*sink_rate*5)/1e6));
end
function [kt, bt] = therm787(k0, b0, tc)
    Tk = tc+273.15; kt = k0*(Tk/288.15); vis= exp(2500*(1/Tk-1/288.15)); bt = b0*max(0.3,min(3.5,vis));
    if tc<0, kt=kt*min(1.24,1+0.008*abs(tc)); bt=bt*min(1.15,1+0.005*abs(tc));
    elseif tc>40, bt=bt*max(0.90,1-0.005*(tc-40)); end
end
function kn = kNorm787(k, tc), kn = k/((tc+273.15)/288.15); end
function [sev,tr,mi] = assess787(cID,sink,gp,health,act_f,b_ang,wow_d,REG,MAINT)
    sev=1; tr=0; mi=1;
    if sink>=REG.SinkHardLanding||gp>=REG.AccelLimit, sev=3;tr=1;mi=9; end
    if gp>=REG.AccelExtreme||sink>4.5, sev=4;tr=1;mi=11; end
    if sink>=REG.SinkCaution&&sev<2, sev=2;tr=1;mi=max(mi,min(cID+1,15)); end
    if health<0.30,     sev=max(sev,4);tr=1;mi=11;
    elseif health<0.50, sev=max(sev,3);tr=1;if mi<9,mi=10;end
    elseif health<0.75, sev=max(sev,2);tr=1;if mi==1,mi=min(cID+1,15);end
    elseif cID>0,       sev=max(sev,2);tr=1;if mi==1,mi=min(cID+1,15);end
    end
    if cID==12&&b_ang>3.0, sev=max(sev,2);tr=1;mi=13; if b_ang>8.0, sev=max(sev,3); end; end
    mi=max(1,min(mi,15));
end
function dp = dmgCalc787(sink,mass,mxD,tc,F)
    geq = max(1.0,min(6.0,1+(sink^2)/(2*9.81*max(mxD,0.005)))); stress= geq*mass*9.81/0.15;
    if stress<=F.S_e, dp=0; return; end
    Nf = max(1, F.N_ref*(F.S_ut/stress)^(1/abs(F.b))); dp = 1/Nf;
    if tc>40, dp=dp*(1+0.015*(tc-40)); end; if tc<-20, dp=dp*(1+0.010*abs(tc+20)); end
end
function c = mCost787(tr,sev,mi,MAINT), if ~tr, c=0; return; end; c=MAINT(mi).cost; end
function cf = fixedCost787(cc,interval,sev,mi,MAINT,ECON,cf)
    if mod(cc,interval)==0, if sev>=3, cf=cf+MAINT(mi).cost; else, cf=cf+ECON.Routine; end; end
end