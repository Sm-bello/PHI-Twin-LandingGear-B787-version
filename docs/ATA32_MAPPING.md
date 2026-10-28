# Boeing 787-8 ATA Chapter 32 Fault Taxonomy & Maintenance Mapping

The **PHI-Twin** framework maps neural prognostic tensors into deterministic Aircraft Maintenance Manual (AMM) tasks adhering to **ATA Chapter 32 (Landing Gear)** and European Aviation Safety Agency (EASA) **CS-25.473** limit load certification specifications.

---

## 1. Class Taxonomy & AMM Task Mapping

| Class ID | Damage Category | ATA Code | AMM Task Reference | Dispatch Action | Physical Phenomenon & Operational Urgency |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | **Normal Touchdown** | 32-00 | `32-00-00-200-001` | **CONTINUE** | Nominal landing impact within elastic limits. Log touchdown parameters per standard ACARS/QAR routine. |
| **1** | **Nitrogen (N2) Micro-Leak** | 32-31 | `32-31-00-200-801` | **SCHEDULE** | Pre-charge gas pressure drop ($P_0 < 17.5$ MPa). Inspect strut servicing and recharge nitrogen chamber. |
| **2** | **Worn Dynamic Seals** | 32-32 | `32-32-00-200-801` | **SCHEDULE** | Chevron packing extrusion and scraper ring wear. Inspect chrome rod for hydraulic weeping; replace packing. |
| **3** | **Early Internal Degradation** | 32-00 | `32-00-00-200-002` | **MONITOR** | Incipient metering pin wear or fluid aeration. Schedule detailed borescopic inspection within 50 cycles. |
| **4** | **Thermal Actuator Degradation** | 32-61 | `32-61-00-200-801` | **SCHEDULE** | Viscous fluid thinning and seal hardening due to extreme operating temperature excursions. |
| **5** | **Brake Fade Friction Drop** | 32-42 | `32-42-00-200-801` | **SCHEDULE** | Carbon-carbon brake disk glazing or thermal fade during runway rollout (deliberately scoped out of vertical impact). |
| **6** | **Tire Delamination / Burst** | 32-41 | `32-41-00-200-801` | **GROUND** | Sudden loss of tire carcass compliance and asymmetric bogie force distribution. Immediate wheel change required. |
| **7** | **Cylinder Pitting Corrosion** | 32-31 | `32-31-00-200-802` | **SCHEDULE** | Environmental pitting and chrome micro-cracking on outer cylinder rod. Apply corrosion inhibiting compound. |
| **8** | **CS-25.473 Hard Landing** | 32-00 | `32-00-00-200-801` | **GROUND** | Limit load exceedance ($\dot{z} > 3.05$ m/s or peak acceleration $> 2.0g$). Mandatory structural NDT teardown. |
| **9** | **Combined N2 Leak + Seal Wear**| 32-31 | `32-31-00-200-801` | **PRIORITY** | Accelerated dual-mode degradation: rapid stroke bottoming risk. Service nitrogen and replace seals within 24h. |
| **10** | **Impending AOG Seizure** | 32-00 | `32-00-00-200-900` | **GROUND** | Severe internal structural galling and mechanical bottoming risk. Immediate aircraft grounding. |
| **11** | **Semi-Levered Bogie Misalign** | 32-30 | `32-30-00-200-801` | **SCHEDULE** | Bogie pitch trim actuator mechanical misalignment. Asymmetric double-peak ground impact signature. |

---

## 2. Regulatory Airworthiness Thresholds (CS-25.473)

* **Maximum Design Sink Rate (Normal Landing):** $v_z \le 3.05$ m/s (10.0 ft/s) at Maximum Landing Weight (MLW = 192,777 kg).
* **Maximum Design Acceleration:** $n_z \le 2.0g$ vertical limit load factor.
* **AOG Grounding Horizon:** Triggered automatically whenever Remaining Useful Life (RUL) falls below 200 flight hours or when catastrophic classes (Class 8, 10) are detected with $>95\%$ neural confidence.
