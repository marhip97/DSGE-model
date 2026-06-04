"""
================================================================================
NEMO FASE II v3 — ADAPTIV MCMC MED KONVERGENSOVERVÅKNING (v2)
================================================================================
Forbedringer fra v1:
  - sigma_A kalibreres fast til K&M-verdi (0.006) — svakt identifisert
  - Starter fra data/results/posterior_v3_final.json hvis tilgjengelig
  - Starter med kalibrert scale fra forrige kjøring
  - 200 000 produksjonstrekkninger for bedre ESS
  - Strengere konvergenskrav: PSRF < 1.10 (ned fra 1.15)

Krever i samme mappe:
    equations.py, parameters.py, blanchard_kahn.py
    nemo_data_faktisk_v2.csv
    T_v3.npy, R_v3.npy
    data/results/posterior_v3_final.json  (valgfritt — brukes som startpunkt)
    chain_baseline_v3.npy    (valgfritt — fallback startpunkt)
================================================================================
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import (solve as sp_solve, cholesky, LinAlgError,
                           solve_discrete_lyapunov)
from scipy.special import betaln, gammaln

from nemo.model.equations import (
    build_matrices_v3, build_matrices_pi4chain, build_matrices_altB,
    build_matrices_v3_forward, build_matrices_v3_plt,
    NZ, NZ_PI4, NZ_ALTB, NZ_PLT, NZ_GEORG, NE,
    Y, C, INV, INV_H, X, M, PI, W, I_R, RER, S, PO, YS,
    Q_H, B_NW, C_NW, I_D, I_L_NW, L, MC,
    E_A, E_C, E_P, E_O, E_Ys, E_rp, E_i, E_H, E_phi_h
)
from nemo.solver.blanchard_kahn import solve as bk_solve
from nemo.model.parameters import Parameters

# sigma_A kalibreres fast — svakt identifisert
SIGMA_A_FIXED = 0.006
# phi_I1 var fast (PE-godkjent 2026-05-17), men frigjort igjen (PE-godkjent 2026-05-20):
# B5-benchmark viste at phi_I1=4.0 gir BNP-respons 0.4× NB; fase2v2 med phi_I1≈0.5 traff NB eksakt.
PHI_I1_FIXED  = 4.0  # beholdt som konstant for bakoverkompatibilitet; ikke lenger brukt i log_posterior
# sigma_rp: fikseres til K&M-verdi (kjøring 10, PE-godkjent 2026-05-24).
# sigma_rp=0.016 i kj9 (2.7× K&M) presset psi_R→0.911, effektiv KPI-koeff = 0.012 (for lav).
# C3-eksperiment viste psi_P1→0.248 og KPI-ratio≈1× da sigma_rp=0.006.
SIGMA_RP_FIXED = 0.006
# psi_R: fikseres til K&M-verdi (kjøring 11, PE-godkjent 2026-05-24).
# psi_R=0.911 i kj10 treffer priorbegrensningen (0.92) og holder effektiv
# KPI-koeff = (1-0.911)×0.167 = 0.015. K&M=0.667 frigjør psi_P1 til å stige.
PSI_R_FIXED = 0.667
# h_c kalibreres fast — PE-godkjent 2026-05-18 (C2 Alt A): posterior treffer alltid
# 0.9995-grensen og dreper konsumkanalen (a3_W→0.006). K&M-verdi 0.938 gir
# a3_W=0.032 og BNP-ratio 1.8× vs NB Memo 3/2024 (mot 6-8× med h_c estimert).
H_C_FIXED = 0.938
# phi_I1 kalibreres fast — PE-godkjent 2026-05-26 (kj20-diagnose): sweep viser
# phi_I1≈0.5 gir BNP-ratio 1.16× NB ([0.8,1.5]×). kj19 estimerte 0.103 (for lav → BNP-eksplosjon).
PHI_I1_FIXED = 0.50
# phi_u kalibreres fast — PE-godkjent 2026-05-28 (kj23-diagnose): posterior konvergerer til
# phi_u=1.72 (8× K&M=0.2192). Med phi_I1=0.50 gir dette BNP=2.33× NB (mål [0.8,1.5]×).
# K&M Tabell 8: phi_u=0.2192 kalibrert fra mikrodata (kapitalutnyttingselastisitet).
# Svakt identifisert fra makroobservablene; fast på K&M-verdi som phi_I1 og sigma_A.
PHI_U_FIXED = 0.2192
# phi_PQ kalibreres fast — PE-godkjent 2026-05-28 (kj25-diagnose): kj24 viste KPI q4=0.48× NB
# (mål ≥0.35×, men full kvartalsmatch gir RMSE=0.258). phi_PQ=300 gir κ_P=0.10 (vs K&M 0.0448),
# som forbedrer KPI-match substansielt. RMSE-sweepdiagnose: phi_PQ=300+rho_s=0.50+psi_R=0.93
# gir total RMSE=0.170 (34% bedre enn baseline=0.258). K&M φ_PQ=669 er Rotemberg-kalibrering;
# norsk prisjustering per kvartal kan være hyppigere enn antatt.
PHI_PQ_FIXED = 300.0   # kappa_P = ε(ε-1)/φ_PQ = 30/300 = 0.100
# psi_R kalibreres fast — PE-godkjent 2026-05-28 (kj25-diagnose): posterior=0.969 gir halvtid
# 18 kvartaler (NB: ~4 kv). Full RMSE domineres av rente-profil-avvik. psi_R=0.90 halverer
# RMSE fra 0.258→0.170. B5 bestått: BNP_ratio=0.84×✅, KPI_ratio=0.81×✅ ved psi_R=0.90.
PSI_R_KJ25_FIXED = 0.90
# kj26 — K&M-korreksjon (PE-godkjent 2026-05-29):
# nemo_complete_documentation_2019.pdf avdekket at φ_I1=0.50 er 25× for lav vs K&M=12.54,
# og φ_PQ=300 er 2× for lav vs K&M=669. Disse korrigeres i kj26 for å eliminere systematisk feil.
# Dessuten: rho_s=0 hardkoding i log_posterior (linje 334) gjorde at kj25 aldri estimerte rho_s
# fra data (posterior var prior-dominert). kj26 fjerner dette og estimerer rho_s genuint.
PHI_I1_KJ26_FIXED = 0.50    # kj49: B5-passing verdi (kj31/kj41 posterior ≈ 0.50); K&M=12.54 kollapser estimation
PHI_PQ_KJ26_FIXED = 669.0   # K&M Tabell 8 (complete doc. s.59): kj25 brukte 300 (2× for lav)
PHI_PQ_KJ38_FIXED = 200.0   # Sandkasse kj38+: kappa_P=0.15, forbedrer PI.q4≈NB
# lambda_pi4: vekt på samtid π i hybrid Taylor-regel (0=ren E_t[π_{t+4}], 1=samtid)
LAMBDA_PI4_FIXED = 0.0
# psi_R: fryses til K&M=0.667 for kj21-diagnose — test om psi_R→0.956 er rotårsak
# til KPI q4-ratio 0.183× NB. Fjern PSI_R_FIXED og reaktiver 'psi_R' i PARAM_PRIORS
# dersom kj21 viser at K&M-kalibrering er OK.
PSI_R_FIXED = 0.667

# ══════════════════════════════════════════════════════════════════════════════
# OBSERVASJONSLIKNING
# ══════════════════════════════════════════════════════════════════════════════

N_OBS = 14
OBS_NAMES = [
    'dy_obs', 'dc_obs', 'dinv_obs', 'dx_obs', 'dm_obs',
    'pi_obs', 'dw_obs', 'i_R_obs', 'i_3m_obs',
    'ds_obs', 'dpO_obs', 'dyS_obs', 'dh_obs', 'db_obs',
]

def build_H():
    H = np.zeros((N_OBS, NZ))
    H[0,Y]=1.0; H[1,C]=1.0; H[2,INV]=1.0; H[3,X]=1.0; H[4,M]=1.0
    H[5,PI]=4.0; H[6,W]=1.0; H[7,I_R]=4.0; H[8,I_R]=4.0; H[9,S]=1.0   # A12.1: ds_obs→S(19) ikke RER(15)
    H[10,PO]=1.0; H[11,YS]=1.0; H[12,Q_H]=1.0; H[13,B_NW]=1.0
    return H

def build_Sv():
    sme = {'dy_obs':0.005,'dc_obs':0.008,'dinv_obs':0.015,'dx_obs':0.010,
           'dm_obs':0.012,'pi_obs':0.008,'dw_obs':0.004,'i_R_obs':0.0005,
           'i_3m_obs':0.0005,'ds_obs':0.010,'dpO_obs':0.050,'dyS_obs':0.006,
           'dh_obs':0.004,'db_obs':0.002}
    return np.diag([sme[n]**2 for n in OBS_NAMES])

# Alt. 4 (PE-godkjent 2026-05-19): utelat RER fra observasjonssettet.
# ds_obs absorberer sigma_rp-dominans; uten RER testes om sigma_rp faller mot K&M.
OBS_NAMES_NO_RER = [n for n in OBS_NAMES if n != 'ds_obs']
N_OBS_NO_RER = len(OBS_NAMES_NO_RER)  # 13

def build_H_no_rer() -> np.ndarray:
    """Observasjonsmatrise uten RER (ds_obs). PE-godkjent Alt. 4, 2026-05-19."""
    H = np.zeros((N_OBS_NO_RER, NZ))
    obs = OBS_NAMES_NO_RER
    mapping = {
        'dy_obs': (Y, 1.0), 'dc_obs': (C, 1.0), 'dinv_obs': (INV, 1.0),
        'dx_obs': (X, 1.0), 'dm_obs': (M, 1.0), 'pi_obs': (PI, 4.0),
        'dw_obs': (W, 1.0), 'i_R_obs': (I_R, 4.0), 'i_3m_obs': (I_R, 4.0),
        'dpO_obs': (PO, 1.0), 'dyS_obs': (YS, 1.0),
        'dh_obs': (Q_H, 1.0), 'db_obs': (B_NW, 1.0),
    }
    for i, nm in enumerate(obs):
        col, scale = mapping[nm]
        H[i, col] = scale
    return H

def build_Sv_no_rer() -> np.ndarray:
    """Målefeil-kovarians uten RER. PE-godkjent Alt. 4, 2026-05-19."""
    sme = {'dy_obs':0.005,'dc_obs':0.008,'dinv_obs':0.015,'dx_obs':0.010,
           'dm_obs':0.012,'pi_obs':0.008,'dw_obs':0.004,'i_R_obs':0.0005,
           'i_3m_obs':0.0005,'dpO_obs':0.050,'dyS_obs':0.006,
           'dh_obs':0.004,'db_obs':0.002}
    return np.diag([sme[n]**2 for n in OBS_NAMES_NO_RER])

# Test A (kj15, PE-godkjent 2026-05-24): fjern i_3m_obs for å eliminere dobbelvekting av I_R.
# H[7,I_R]=4 (i_R_obs) OG H[8,I_R]=4 (i_3m_obs) → I_R over-identifisert → psi_R≈0.95 → psi_P1 lav.
OBS_NAMES_NO_I3M = [n for n in OBS_NAMES if n != 'i_3m_obs']  # 13 obs
N_OBS_NO_I3M = len(OBS_NAMES_NO_I3M)

def build_H_no_i3m() -> np.ndarray:
    """Observasjonsmatrise uten i_3m_obs — fjerner dobbelvekting av I_R. Test A (kj15)."""
    H = np.zeros((N_OBS_NO_I3M, NZ))
    mapping = {
        'dy_obs': (Y, 1.0), 'dc_obs': (C, 1.0), 'dinv_obs': (INV, 1.0),
        'dx_obs': (X, 1.0), 'dm_obs': (M, 1.0), 'pi_obs': (PI, 4.0),
        'dw_obs': (W, 1.0), 'i_R_obs': (I_R, 4.0), 'ds_obs': (S, 1.0),
        'dpO_obs': (PO, 1.0), 'dyS_obs': (YS, 1.0),
        'dh_obs': (Q_H, 1.0), 'db_obs': (B_NW, 1.0),
    }
    for i, nm in enumerate(OBS_NAMES_NO_I3M):
        col, scale = mapping[nm]
        H[i, col] = scale
    return H

def build_Sv_no_i3m() -> np.ndarray:
    """Målefeil-kovarians uten i_3m_obs (13×13 diagonal). Test A (kj15)."""
    sme = {'dy_obs':0.005,'dc_obs':0.008,'dinv_obs':0.015,'dx_obs':0.010,
           'dm_obs':0.012,'pi_obs':0.008,'dw_obs':0.004,'i_R_obs':0.0005,
           'ds_obs':0.010,'dpO_obs':0.050,'dyS_obs':0.006,
           'dh_obs':0.004,'db_obs':0.002}
    return np.diag([sme[n]**2 for n in OBS_NAMES_NO_I3M])

# Test B (kj16): KPI-JAE som pi-observasjon i stedet for total KPI.
# H-matrisen er identisk med build_H() — PI-rad mapper til PI(0) uendret.
# Forskjellen er i data: pre/post-arrayene bruker pi_core_obs-kolonnen (fra SSB 10235).
# Krever at nemo_data-CSVen inneholder pi_core_obs (kjør pipeline med kpi_jae=True).
# NB: SSB-API (data.ssb.no) er ikke tilgjengelig fra dette skymiljøet — kjøres lokalt.
def build_H_core() -> np.ndarray:
    """Observasjonsmatrise for KPI-JAE-test (identisk med build_H()). Test B (kj16)."""
    return build_H()

def build_Sv_core() -> np.ndarray:
    """Målefeil-kovarians for KPI-JAE-test (identisk med build_Sv()). Test B (kj16)."""
    return build_Sv()


def build_H_pi4chain() -> np.ndarray:
    """Observasjonsmatrise for pi4chain (NZ_PI4=53 kolonner, identisk mapping som build_H)."""
    H = np.zeros((N_OBS, NZ_PI4))
    H[0,Y]=1.0; H[1,C]=1.0; H[2,INV]=1.0; H[3,X]=1.0; H[4,M]=1.0
    H[5,PI]=4.0; H[6,W]=1.0; H[7,I_R]=4.0; H[8,I_R]=4.0; H[9,S]=1.0
    H[10,PO]=1.0; H[11,YS]=1.0; H[12,Q_H]=1.0; H[13,B_NW]=1.0
    return H


def build_H_altB() -> np.ndarray:
    """
    Observasjonsmatrise for Alt B (NZ_ALTB=51 kolonner).

    Endringer fra build_H (NZ=49):
    - dinv_obs (rad 2): mappes til IY*INV + IHY*INV_H (vektet total investering)
      IY=0.20, IHY=0.10 → INV-vekt=2/3, INV_H-vekt=1/3
    - Alle andre rader identiske med build_H
    """
    from nemo.model.parameters import Parameters as _P
    IY  = _P.IY   # 0.20
    IHY = _P.IHY  # 0.10
    tot = IY + IHY
    H = np.zeros((N_OBS, NZ_ALTB))
    H[0,Y]=1.0; H[1,C]=1.0
    H[2, INV]   = IY  / tot   # kapitalandel av total investering
    H[2, INV_H] = IHY / tot   # boligandel av total investering
    H[3,X]=1.0; H[4,M]=1.0
    H[5,PI]=4.0; H[6,W]=1.0; H[7,I_R]=4.0; H[8,I_R]=4.0; H[9,S]=1.0
    H[10,PO]=1.0; H[11,YS]=1.0; H[12,Q_H]=1.0; H[13,B_NW]=1.0
    return H


def build_H_plt() -> np.ndarray:
    """Observasjonsmatrise for PLT-modellen (NZ_PLT=51 kolonner).
    P_STAR_GAP (index 50) er ikke direkte observert — ekstra null-kolonne.
    """
    H_50 = build_H()
    H = np.zeros((N_OBS, NZ_PLT))
    H[:, :NZ] = H_50
    return H


def build_H_georg() -> np.ndarray:
    """Observasjonsmatrise for GEORG-modellen (NZ_GEORG=64 kolonner).
    De 14 GEORG-tilstandene (lagg + AR(1)-Z) er ikke direkte observert —
    ekstra null-kolonner. Første NZ kolonner matcher build_H().
    """
    H_50 = build_H()
    H = np.zeros((N_OBS, NZ_GEORG))
    H[:, :NZ] = H_50
    return H


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERE OG PRIOR
# sigma_A er fjernet fra estimering — kalibreres fast
# ══════════════════════════════════════════════════════════════════════════════

PARAM_PRIORS = {
    # PE-godkjent 2026-05-16: Beta(2,2) — symmetrisk; data støtter ikke K&M rho_A=0.804
    'rho_A':   ('beta',     2.0,  2.0,  0.01, 0.9995),
    'rho_C':   ('beta',     2.0,  0.5,  0.01, 0.9995),
    # rho_O: kj47: prior tilbake til original Beta(2,0.5,[0.01,0.9995]) etter at
    # strammet Beta(6,1.5,[0.50,0.9995]) ga PSRF=11.8 (likelihood-klippe ved 0.50).
    # Data vil ha rho_O≈0.24 — dette er en genuin modell-egenskap, ikke misidentifikasjon.
    # Alt C (stram rho_O-prior) forlatt; phi_O-estimering (Alt A) beholdes.
    'rho_O':   ('beta',     2.0,  0.5,  0.01, 0.9995),
    'rho_Ys':  ('beta',     2.0,  0.5,  0.01, 0.9995),
    'rho_rp':  ('beta',     2.0,  0.5,  0.01, 0.9995),
    'rho_H':   ('beta',     5.0,  3.0,  0.30, 0.95),   # kj28: fikset fra Beta(2,0.5) → mode=0.667≈K&M=0.694
    # PE-godkjent 2026-05-21 (A6): sigma_A fristilles. K&M Tabell 9 estimerer sigma_A.
    # Bayesiansk faktor sigma_A=0.012 vs 0.006: ~10^27. rho_A=0.39 i kj10 er identifikasjonsartefakt.
    # PE-godkjent 2026-05-28 (kj21): sigma_A fryses igjen. kj20 drev sigma_A→0.049 (tak=0.050),
    # og phi_u→0.012 (gulv=0.010), noe som ga 0% aksept i kj21. Svakt identifisert.
    # 'sigma_A':  ('normal', 0.010, 0.004, 0.002, 0.050),  # DEAKTIVERT — kalibreres fast SIGMA_A_FIXED
    'sigma_C':  ('inv_gamma', 2.0, 0.0182, 1e-5, 0.5),
    'sigma_O':  ('inv_gamma', 2.0, 0.0475, 1e-5, 1.0),
    'sigma_Ys': ('inv_gamma', 2.0, 0.0067, 1e-5, 0.5),
    # sigma_rp fjernet fra estimering — kalibreres fast til SIGMA_RP_FIXED=0.006 (PE-godkjent 2026-05-24).
    'sigma_i':  ('inv_gamma', 2.0, 0.0002, 1e-5, 0.1),
    'sigma_P':  ('inv_gamma', 2.0, 0.0027, 1e-5, 0.5),
    'sigma_H':  ('inv_gamma', 2.0, 0.0500, 1e-5, 1.0),
    # psi_R: kj25 festet til 0.90. kj26 reaktiverer: med φ_I1=12.54 (K&M) er investeringene
    # mye tregere og B5-grensen for psi_R kan ligge lavere. Beta(2,2,[0.50,0.95]) sentrert ~0.73.
    # K&M mimicking rule: ω_R=0.6663. K&M Taylor rule: ikke spesifisert; estimeres.
    # psi_R kj27: tak hevet fra 0.95→0.99. kj26 traff 0.9486 (std=0.001) — klart prior-tak.
    # Med K&M φ_I1=12.54 trenger modellen høy renteglattning for tilstrekkelig BNP-transmisjon.
    'psi_R':   ('beta',   2.0, 2.0, 0.50, 0.99),   # kj27: utvidet fra [0.50,0.95] (PE-godkjent 2026-05-29)
    # psi_R2: AR(2) 2-periodes lagg (Alt. A2, PE-godkjent 2026-06-02).
    # DEAKTIVERT etter kj45 (2026-06-02): estimert til -0.0003 (sd=0.0003) — presset mot
    # øvre grense 0.0. Data forkaster mean-reversion entydig; AR(2)-leddet er en død tilstand
    # (modellen oppfører seg eksakt som AR(1)). I_R.q12 forble 0.848 (NB: -0.15).
    # Kalibreres fast = 0.0 (Parameters.psi_R2). NZ=50-infrastrukturen beholdt som exit-mulighet.
    # Exit: gjenaktiver linjen under for å estimere psi_R2 på nytt.
    # 'psi_R2':  ('normal', -0.10, 0.05, -0.40, 0.00),  # DEAKTIVERT etter kj45
    'psi_P1':  ('normal', 0.29, 0.10, 0.05, 1.50),
    'psi_Y':   ('normal', 0.24, 0.05, 0.01, 0.80),
    # gamma_p: Calvo-prisindeksasjon i hybrid NK Phillips-kurve (PE-godkjent 2026-05-24).
    # K&M Tabell 8: γ_p ≈ 0.35. Beta(3,3) sentrert ~0.5, tillater [0, 0.95].
    'gamma_p': ('beta',   3.0, 3.0,  0.0,  0.95),
    # h_c er fjernet fra estimering — kalibreres fast til H_C_FIXED=0.938 (PE-godkjent 2026-05-18, C2 Alt A).
    # Posterior traff alltid 0.9995-grensen og drepte konsumkanalen. K&M-verdi gjenoppretter a3_W=0.032.
    # phi_I1 kj49: DEAKTIVERT — kalibreres fast=0.50 (PE-godkjent 2026-06-03).
    # kj47/kj48: phi_I1 kollapset til nedre grense 0.10 (std=0.0001) — likelihood-drag
    # overveldende (~800+ log-enheter). LogNormal(log(12.54),0.5)-prior (kj48) holdt ikke.
    # Beste baseline kj41 (RMSE=0.277) brukte phi_I1≈0.50 effektivt fast.
    # B5-passing region: phi_I1∈[0.30,0.75]. 0.50 er sentrum, strukturelt validert.
    # Exit: gjenaktiver med LogNormal(log(0.50),0.15,[0.30,0.75]) dersom identifikasjonsproblem løses.
    # 'phi_I1':  ('lognormal', np.log(12.54), 0.5, 0.1, 40.0),  # DEAKTIVERT kj49
    # phi_I2: kj25 prior Normal(8,4,[0.5,40]) truncerte K&M=165.66. kj26 åpner prioren:
    # Normal(50,50,[1,400]) lar data velge mellom kj25-estimat (~12) og K&M (166).
    'phi_I2':  ('normal', 50.0, 50.0, 1.0, 400.0),
    # Fase 2v2 (2026-05-15): kapitalutnyttelseselastisitet (Alt. A, K&M Tabell 8)
    # phi_u kj23: posterior=1.72 (8× K&M) → BNP=2.33× NB med phi_I1=0.50. Kalibreres fast=0.2192.
    # Svakt identifisert fra makrodata; K&M Tabell 8 (mikrodata). PE-godkjent 2026-05-28.
    # Exit-mulighet: gjenaktiver med Normal(0.22,0.10,[0.01,2.0]) ved behov.
    # 'phi_u':   ('normal', 0.22, 0.10, 0.01, 2.0),  # DEAKTIVERT etter kj23 → fast=PHI_U_FIXED
    # phi_PQ kj13: svakt identifisert [104,1089] → KPI 0.21× NB. Ikke estimer på nytt.
    # 'phi_PQ':  ('normal', 669.0, 300.0, 50.0, 2000.0),  # DEAKTIVERT etter kj13
    # kappa_M kj14: data vil ha LAVERE kappa_M (0.0175 < K&M=0.030) → KPI 0.13× NB. Ikke estimer på nytt.
    # 'kappa_M': ('normal', 0.03, 0.03, 0.005, 0.20),   # DEAKTIVERT etter kj13
    # rho_s: DEAKTIVERT kj47 (2026-06-03, PE-godkjent). kj46 estimerte 0.003±0.003 — degenerert
    # posterior nær null. Kalibreres fast = 0.00 i parameters.py. ESS-bottleneck eliminert.
    # Exit: gjenaktiver Beta(2,2,[0.05,0.90]) ved ny diagnose.
    # 'rho_s': ('beta', 2.0, 2.0, 0.05, 0.90),  # DEAKTIVERT kj47
    # phi_O: frigjort for estimering kj47 (PE-godkjent 2026-06-03, Alt. A).
    # Kalibrert fast=0.15 (K&M Tabell 8) i alle kjøringer t.o.m. kj46. Frigjøres for å la
    # data velge olje→RER-styrke. Normal(0.15,0.10,[0.01,0.80]) — sentrert på K&M med brede haler.
    'phi_O':  ('normal', 0.15, 0.10, 0.01, 0.80),  # Aktivert kj47
    # phi_H1 kj27 (Alt B, PE-godkjent 2026-05-29): boliginvesteringsjusteringskost.
    # K&M Tabell 8: 60.73. phi_H1-sweep viser at K&M-verdi gir BNP q4=0.33× (mål 0.8×).
    # Med φ_I1=12.54 mangler vår forenklede modell NB-kanalene — phi_H1 estimeres for å
    # la data avgjøre kompensasjonsgraden. Prior Normal(60.73, 40, [0.5, 200]) — bredt.
    'phi_H1': ('normal', 60.73,  5.0, 30.0, 100.0),  # kj28: strammet fra (40,[0.5,200]) → eliminerer bimodal
    # psi_PL: PLT prisnivåmål-koeffisient (Fase 2, 2026-06-02, kj46).
    # Normal(0.10, 0.05, [0.00, 0.50]): sentrert over typisk PLT-respons.
    # psi_PL=0 → exitstrategi (ren inflasjonsmål). Gjenaktiver for kj46.
    # 'psi_PL': ('normal', 0.10, 0.05, 0.00, 0.50),  # DEAKTIVERT — aktiver for kj46
}
PARAM_NAMES = list(PARAM_PRIORS.keys())
N_PARAMS    = len(PARAM_NAMES)

PARAM_NAMES_V3_FULL = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
                        'sigma_A','sigma_C','sigma_O','sigma_Ys','sigma_rp',
                        'sigma_i','sigma_P','sigma_H','psi_R','psi_P1','psi_Y','h_c']

KM = {'rho_A':0.804,'rho_C':0.725,'rho_O':0.874,'rho_Ys':0.783,
      'rho_rp':0.737,'rho_H':0.694,'sigma_A':0.006,'sigma_C':0.030,
      'sigma_O':0.079,'sigma_Ys':0.011,'sigma_rp':0.006,'sigma_i':0.0003,
      'sigma_P':0.003,'sigma_H':0.050,'psi_R':0.666,'psi_P1':0.292,
      'psi_Y':0.242,'h_c':0.938,'gamma_p':0.35,
      'psi_R2':0.0,  # AR(2)-lagg; 0.0 = AR(1)-exit (Alt. A2, PE-godkjent 2026-06-02)
      'psi_PL':0.0,  # PLT-koeffisient; 0.0 = ren inflasjonsmål (exitstrategi, Fase 2 2026-06-02)
      'phi_I1':12.54,'phi_I2':165.66,'phi_u':0.2192,  # K&M complete doc. s.59: phi_I1=12.54, phi_I2=165.66
      'phi_PQ':669.0,'kappa_M':0.03,'rho_s':0.00,  # rho_s: kj47 fast=0.00
      'phi_O':0.15,   # K&M Tabell 8: olje→RER-kanal; frigjort kj47
      'phi_H1':60.73}  # K&M Tabell 8: boliginvesteringsjusteringskost.

def log_prior(theta, overrides=None):
    """
    Log-prior. overrides: valgfri dict {param_name: spec_tuple} som erstatter
    PARAM_PRIORS lokalt — uten å endre global PARAM_PRIORS (exit-mulighet).
    """
    priors = PARAM_PRIORS if overrides is None else {**PARAM_PRIORS, **overrides}
    lp = 0.0
    for i, name in enumerate(PARAM_NAMES):
        x = theta[i]; spec = priors[name]; lb,ub = spec[-2],spec[-1]
        if x < lb or x > ub: return -np.inf
        pt = spec[0]
        if pt == 'beta':
            a,b = spec[1],spec[2]; xn=(x-lb)/(ub-lb)
            if xn<=0 or xn>=1: return -np.inf
            lp += (a-1)*np.log(xn)+(b-1)*np.log(1-xn)-betaln(a,b)
        elif pt == 'normal':
            mu,sig = spec[1],spec[2]
            lp += -0.5*((x-mu)/sig)**2-np.log(sig)-0.5*np.log(2*np.pi)
        elif pt == 'inv_gamma':
            sh,sc = spec[1],spec[2]
            if x<=0: return -np.inf
            lp += sh*np.log(sc)-(sh+1)*np.log(x)-sc/x-gammaln(sh)
        elif pt == 'lognormal':
            # spec: ('lognormal', mu_log, sig_log, lb, ub). mu_log/sig_log i log-rom.
            mu,sig = spec[1],spec[2]
            if x<=0: return -np.inf
            lp += -0.5*((np.log(x)-mu)/sig)**2-np.log(x*sig)-0.5*np.log(2*np.pi)
    return lp


# ══════════════════════════════════════════════════════════════════════════════
# KALMAN-FILTER MED COVID-HULL
# ══════════════════════════════════════════════════════════════════════════════

def build_Q(theta):
    # sigma_rp og sigma_A faste — ikke i theta, bruker _fixed-oppslag.
    smap = {E_A:'sigma_A',E_C:'sigma_C',E_P:'sigma_P',E_O:'sigma_O',
            E_Ys:'sigma_Ys',E_rp:'sigma_rp',E_i:'sigma_i',E_H:'sigma_H'}
    _fixed = {'sigma_rp': SIGMA_RP_FIXED, 'sigma_A': SIGMA_A_FIXED}
    Q = np.zeros((NE,NE))
    for idx,pn in smap.items():
        if pn in PARAM_NAMES:
            s = theta[PARAM_NAMES.index(pn)]
        elif pn in _fixed:
            s = _fixed[pn]
        else:
            s = getattr(Parameters, pn, 0.01)
        Q[idx,idx] = s**2
    return Q

def _kf_block(T_mat, R_mat, H, Q, Sv, Y_obs):
    NZ_l=T_mat.shape[0]; RQR=R_mat@Q@R_mat.T
    try:    P=solve_discrete_lyapunov(T_mat,RQR)
    except: P=np.eye(NZ_l)*0.01
    z=np.zeros(NZ_l); ll=0.0
    for t in range(len(Y_obs)):
        zp=T_mat@z; Pp=T_mat@P@T_mat.T+RQR
        yt=Y_obs[t]; ms=np.isnan(yt)
        if ms.all(): z=zp; P=Pp; continue
        Ht=H[~ms]; yo=yt[~ms]; Sv_t=Sv[np.ix_(~ms,~ms)]
        inn=yo-Ht@zp; S=Ht@Pp@Ht.T+Sv_t; S=(S+S.T)/2
        try:
            Lc=cholesky(S,lower=True)
            ll -= 0.5*(2*np.sum(np.log(np.diag(Lc)))
                      +inn@sp_solve(S,inn,assume_a='pos')
                      +len(inn)*np.log(2*np.pi))
        except LinAlgError: return -np.inf
        Kg=Pp@Ht.T@np.linalg.inv(S); z=zp+Kg@inn
        P=(np.eye(NZ_l)-Kg@Ht)@Pp; P=(P+P.T)/2
    return ll

def kalman_hull(T_mat, R_mat, H, Q, Sv, Y_pre, Y_post):
    ll1=_kf_block(T_mat,R_mat,H,Q,Sv,Y_pre)
    if not np.isfinite(ll1): return -np.inf
    ll2=_kf_block(T_mat,R_mat,H,Q,Sv,Y_post)
    return ll1+ll2 if np.isfinite(ll2) else -np.inf

def log_posterior(theta, H, Sv, Y_pre, Y_post, build_fn=None, prior_overrides=None):
    """
    Log-posterior for MCMC.

    Parametere
    ----------
    theta           : parametervektor
    H, Sv           : observasjons- og støymatriser
    Y_pre, Y_post   : data (pre/post COVID)
    build_fn        : valgfri funksjon(p, theta_H) → (G0,G1,Psi,Pi).
                      None = standard v3/pi4chain logikk (bakoverkompatibel).
                      Sett til build_matrices_altB for kj27+ (Alt B).
    prior_overrides : valgfri dict {param_name: (spec_tuple)} som overstyrer
                      PARAM_PRIORS lokalt. Brukes for å endre enkeltprior uten
                      global endring (f.eks. psi_R: [0.50,0.99] i kj27).
    """
    if prior_overrides:
        lp = log_prior(theta, overrides=prior_overrides)
    else:
        lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    try:
        class Pt(Parameters): pass
        for i,n in enumerate(PARAM_NAMES): setattr(Pt,n,float(theta[i]))
        setattr(Pt,'h_c',       H_C_FIXED)        # fast — PE-godkjent 2026-05-18 (C2 Alt A)
        setattr(Pt,'sigma_rp',  SIGMA_RP_FIXED)   # fast — PE-godkjent 2026-05-24 (kj10)
        setattr(Pt,'sigma_A',   SIGMA_A_FIXED)    # fast=0.006 — PE-godkjent 2026-05-28 (kj20: tak-problem)
        setattr(Pt,'kappa_M',   KM['kappa_M'])    # fast K&M=0.030 — kj14 viste estimering forverrer KPI
        if 'phi_I1' not in PARAM_NAMES:
            setattr(Pt,'phi_I1',    PHI_I1_KJ26_FIXED)  # fast K&M=12.54 hvis ikke estimert
        setattr(Pt,'phi_u',     PHI_U_FIXED)
        setattr(Pt,'phi_PQ',    PHI_PQ_KJ26_FIXED)  # K&M=669
        setattr(Pt,'lambda_pi4',LAMBDA_PI4_FIXED)
        if build_fn is not None:
            G0,G1,Psi,Pi = build_fn(Pt, theta_H=0.05)
        else:
            use_pi4 = H.shape[1] == NZ_PI4
            if use_pi4:
                G0,G1,Psi,Pi=build_matrices_pi4chain(Pt,theta_H=0.05)
            else:
                G0,G1,Psi,Pi=build_matrices_v3(Pt,theta_H=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T_n,R_n,d=bk_solve(G0,G1,Psi,Pi,verbose=False)
        if not d['stable']: return -np.inf
        ll=kalman_hull(T_n,R_n,H,build_Q(theta),Sv,Y_pre,Y_post)
        return ll+lp if np.isfinite(ll) else -np.inf
    except: return -np.inf


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIKK
# ══════════════════════════════════════════════════════════════════════════════

def compute_psrf(chain):
    if len(chain)<400: return np.full(N_PARAMS,99.0)
    n=len(chain)//4
    segs=[chain[k*n:(k+1)*n] for k in range(4)]
    means=np.array([s.mean(0) for s in segs]); grand=means.mean(0)
    B=n/3*np.sum((means-grand)**2,axis=0)
    W=np.mean([np.var(s,axis=0,ddof=1) for s in segs],axis=0)
    vp=(n-1)/n*W+B/n
    return np.sqrt(vp/np.where(W>0,W,np.nan))

def compute_ess(x):
    n=len(x); x_=x-x.mean(); var=np.var(x_)
    if var<1e-14: return 1.0
    iat=1.0
    for k in range(1,min(500,n//2)):
        ac=np.sum(x_[k:]*x_[:-k])/(n*var)
        if ac<0.05: break
        iat+=2*ac
    return n/max(iat,1.0)

def check_convergence(chain, psrf_thr=1.10, ess_pct=0.02):
    if len(chain)<400: return False,99.0,0.0,PARAM_NAMES
    ps=compute_psrf(chain)
    ess=[compute_ess(chain[:,i]) for i in range(N_PARAMS)]
    ess_min=min(ess); psrf_max=float(np.nanmax(ps))
    problems=[PARAM_NAMES[i] for i in range(N_PARAMS)
              if ps[i]>psrf_thr or ess[i]/len(chain)<ess_pct]
    ok = psrf_max<=psrf_thr and ess_min/len(chain)>=ess_pct
    return ok, psrf_max, ess_min, problems

def print_diagnostics(chain, label=""):
    ps=compute_psrf(chain); ess=[compute_ess(chain[:,i]) for i in range(N_PARAMS)]
    print(f"\n  {'='*72}")
    if label: print(f"  DIAGNOSTIKK — {label}")
    print(f"  sigma_A kalibrert fast til {SIGMA_A_FIXED}")
    print(f"  {'Parameter':<12} {'K&M':>7} {'Mean':>8} {'Std':>7} "
          f"{'[5%,95%]':>14} {'ESS':>6} {'PSRF':>6}")
    print(f"  {'─'*68}")
    for i,name in enumerate(PARAM_NAMES):
        s=chain[:,i]; k=KM.get(name,float('nan')); flag="!" if ps[i]>1.10 else " "
        print(f"  {name:<12} {k:>7.4f} {s.mean():>8.4f} {s.std():>7.4f} "
              f"[{np.percentile(s,5):.3f},{np.percentile(s,95):.3f}] "
              f"{ess[i]:>6.0f} {ps[i]:>5.3f}{flag}")
    print(f"\n  Konvergens: {sum(p<=1.10 for p in ps)}/{N_PARAMS} OK  "
          f"max PSRF={np.nanmax(ps):.3f}  min ESS={min(ess):.0f}")
    print(f"  {'='*72}")


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIV MCMC MED KONVERGENSOVERVÅKNING
# ══════════════════════════════════════════════════════════════════════════════

def adaptive_mcmc_with_monitoring(
        Y_pre, Y_post, H, Sv, theta_init, post_std_init,
        n_production=200000, burnin=20000, adapt_every=500,
        check_every=10000, max_recalib=5,
        psrf_thr=1.10, ess_pct_thr=0.02,
        scale_init=0.676, seed=42, verbose=True,
        save_prefix="chain_v3_v2", use_reparam=False,
        block_indices=None,
        build_fn=None, prior_overrides=None):
    """
    Adaptiv MCMC med konvergensovervåkning.

    build_fn        : valgfri modellbygger (None = standard v3/pi4chain).
                      Sett build_matrices_altB for kj27+ (Alt B).
    prior_overrides : valgfri dict {param: spec} for lokal prior-overstyring.
                      Endrer IKKE global PARAM_PRIORS (exit-mulighet bevares).
    """
    import functools
    rng=np.random.default_rng(seed); N=N_PARAMS
    scale=scale_init; post_std=post_std_init.copy()

    # Logit-reparametrisering (Fase 2, C5 §2): h_c og psi_R sampler i ubegrenset rom.
    # Chain lagres internt i ubegrenset rom; transformeres tilbake ved diagnostikk/lagring.
    if use_reparam:
        from nemo.estimation.reparam import (
            to_natural, to_unconstrained, wrap_log_posterior, REPARAM_PARAMS,
        )
        _base = functools.partial(log_posterior, build_fn=build_fn, prior_overrides=prior_overrides)
        log_post_fn = wrap_log_posterior(_base)
        theta_internal = to_unconstrained(theta_init)
        # post_std for reparametriserte parametre tolkes nå i unc-rom — sett til 1.0
        # som default skala (logit-rom; presis verdi tunes av sampleren).
        for n in REPARAM_PARAMS:
            i = PARAM_NAMES.index(n)
            post_std[i] = 1.0
        if verbose:
            print(f"  Logit-reparametrisering AKTIV for: {REPARAM_PARAMS}")
    else:
        if build_fn is not None or prior_overrides is not None:
            log_post_fn = functools.partial(log_posterior,
                                            build_fn=build_fn,
                                            prior_overrides=prior_overrides)
        else:
            log_post_fn = log_posterior
        theta_internal = theta_init.copy()
        to_natural = None  # ikke brukt

    def _chain_natural(ch_internal):
        """Transformer chain tilbake til naturlig rom hvis reparam er aktiv."""
        if not use_reparam:
            return ch_internal
        out = np.empty_like(ch_internal)
        for i in range(len(ch_internal)):
            out[i] = to_natural(ch_internal[i])
        return out

    # Start med diagonal proposal; bytt til empirisk kovarians etter burn-in
    # (Haario adaptive Metropolis — fanger sigma_C/h_c-korrelasjon automatisk)
    C_prop=np.diag((scale*2.38/np.sqrt(N)*post_std)**2+1e-12)
    use_empirical_cov=False  # settes True etter burn-in
    theta=theta_internal
    lp_cur=log_post_fn(theta,H,Sv,Y_pre,Y_post)
    if not np.isfinite(lp_cur):
        raise ValueError(f"Startverdi gir ikke-endelig log-posterior: {lp_cur}")
    recalib_log=[]; t_total=time.time()

    def _run(n_steps, phase, adapt=False, monitor=False):
        nonlocal theta,lp_cur,scale,C_prop
        ch=np.zeros((n_steps,N)); lp_v=np.zeros(n_steps)
        acc=0; acc_win=0; t0=time.time()
        # Blokksampling (PE-godkjent 2026-05-16): Metropolis-within-Gibbs per blokk.
        # Brukes kun etter burn-in (use_empirical_cov=True) der sub-kovariansen er meningsfull.
        # Proposal per blokk: C_block = C_prop[blk,blk] * (N/nb) — justerer for blokkdimensjon.
        use_blocks = block_indices is not None and use_empirical_cov
        n_moves = len(block_indices) if use_blocks else 1
        for i in range(n_steps):
            if use_blocks:
                for blk in block_indices:
                    tp=theta.copy(); nb=len(blk)
                    C_blk=C_prop[np.ix_(blk,blk)]*(N/nb)
                    tp[blk]+=rng.multivariate_normal(np.zeros(nb),C_blk)
                    lpp=log_post_fn(tp,H,Sv,Y_pre,Y_post)
                    if np.log(rng.uniform())<lpp-lp_cur:
                        theta=tp; lp_cur=lpp; acc+=1; acc_win+=1
            else:
                tp=theta+rng.multivariate_normal(np.zeros(N),C_prop)
                lpp=log_post_fn(tp,H,Sv,Y_pre,Y_post)
                if np.log(rng.uniform())<lpp-lp_cur:
                    theta=tp; lp_cur=lpp; acc+=1; acc_win+=1
            ch[i]=theta; lp_v[i]=lp_cur
            if adapt and (i+1)%adapt_every==0:
                rate=acc_win/(adapt_every*n_moves); acc_win=0
                if   rate<0.10: scale*=0.60
                elif rate<0.15: scale*=0.75
                elif rate<0.20: scale*=0.88
                elif rate>0.40: scale*=1.40
                elif rate>0.32: scale*=1.20
                elif rate>0.28: scale*=1.10
                scale=float(np.clip(scale,0.005,10.0))
                if use_empirical_cov and i+1>=2*adapt_every:
                    # Empirisk kovarians fra siste segment (Haario AM)
                    seg=ch[max(0,i+1-5000):i+1]
                    C_emp=np.cov(seg.T)+1e-10*np.eye(N)
                    C_prop=(scale*2.38)**2/N*C_emp
                else:
                    C_prop=np.diag((scale*2.38/np.sqrt(N)*post_std)**2+1e-12)
            if monitor and (i+1)%check_every==0:
                ok,pmax,emin,_=check_convergence(ch[:i+1],psrf_thr,ess_pct_thr)
                rem=(time.time()-t0)/(i+1)*(n_steps-i-1)
                status="OK" if ok else f"PSRF={pmax:.2f} ESS={emin:.0f}"
                if verbose:
                    print(f"  [{i+1:7d}/{n_steps}] acc={acc/(i+1):.3f}  "
                          f"lp={lp_cur:.1f}  scale={scale:.4f}  "
                          f"konv={status}  gjenstår≈{rem/60:.1f}min")
                # Løpende lagring — flush til disk umiddelbart
                np.save(f"{save_prefix}_partial.npy", ch[:i+1])
            elif save_prefix and (i+1)%1000==0:
                # Ekstra sikkerhet: lagre hvert 1000 trekk i produksjon
                np.save(f"{save_prefix}_partial.npy", ch[:i+1])
            if verbose and (i+1)%5000==0 and not (monitor and (i+1)%check_every==0):
                rem=(time.time()-t0)/(i+1)*(n_steps-i-1)
                print(f"  [{i+1:7d}/{n_steps}] acc={acc/(i+1):.3f}  "
                      f"lp={lp_cur:.1f}  scale={scale:.4f}  "
                      f"gjenstår≈{rem/60:.1f}min  [{phase}]")
        return ch, lp_v, acc/(n_steps*n_moves)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  NEMO v3 — ADAPTIV MCMC v2 (sigma_A fast={SIGMA_A_FIXED})")
        print(f"  T_pre={len(Y_pre)} kv  T_post={len(Y_post)} kv  N={N} param")
        print(f"  Produksjon={n_production:,}  Burn-in={burnin:,}")
        print(f"  Startscale={scale_init:.4f}  PSRF-krav<{psrf_thr}")
        if block_indices is not None:
            blk_str = " | ".join(
                "{"+",".join(PARAM_NAMES[j] for j in blk)+"}" for blk in block_indices
            )
            print(f"  Blokksampling: {blk_str}")
        print(f"{'='*65}")
        print(f"\n  Startverdi log-posterior: {lp_cur:.2f}")
        print(f"\n--- FASE 1: Adaptiv burn-in ({burnin:,} trekk) ---")

    ch_bi,_,acc_bi=_run(burnin,"BURN-IN",adapt=True)
    if verbose: print(f"  Ferdig. acc={acc_bi:.3f}  scale={scale:.4f}")

    # Aktiver empirisk kovariansproposal — fanger korrelasjoner (sigma_C/h_c)
    # NB: Reset scale=1.0 siden empirisk cov allerede enkoder posterior-størrelse
    # (Haario et al. 2001: C_prop = 2.38²/N · Σ, scale-tuning gjøres deretter)
    use_empirical_cov=True
    scale=1.0
    C_emp_init=np.cov(ch_bi[-min(burnin,5000):].T)+1e-10*np.eye(N)
    C_prop=(scale*2.38)**2/N*C_emp_init
    if verbose:
        # Vis sterkeste korrelasjoner i empirisk C
        D=np.sqrt(np.diag(C_emp_init))
        R_emp=C_emp_init/np.outer(D,D)
        idx=np.argsort(np.abs(R_emp).flatten())[::-1]
        print(f"  Empirisk kovarians aktivert. Sterkeste korrelasjoner:")
        seen=set()
        for k in idx[:20]:
            i,j=k//N,k%N
            if i==j or (j,i) in seen: continue
            seen.add((i,j))
            if len(seen)>3: break
            print(f"    {PARAM_NAMES[i]:10s} ↔ {PARAM_NAMES[j]:10s}: r={R_emp[i,j]:+.3f}")

    if verbose: print(f"\n--- FASE 2: Konvergensovervåkning ---")
    monitor_ch=ch_bi.copy(); n_recalib=0

    for rd in range(max_recalib+1):
        ok,pmax,emin,probs=check_convergence(monitor_ch,psrf_thr,ess_pct_thr)
        if verbose:
            status="KONVERGERT" if ok else f"IKKE OK — PSRF={pmax:.3f}, ESS={emin:.0f}"
            print(f"\n  Sjekk runde {rd}: {status}")
            if not ok and probs: print(f"  Problemer: {probs[:6]}")
        if ok or rd==max_recalib:
            if not ok and verbose:
                print(f"  Maks rekalibreringer nådd. Fortsetter.")
            break
        actual_std=monitor_ch.std(axis=0)
        actual_std=np.where(actual_std<1e-8,post_std_init,actual_std)
        scale=float(np.clip(scale*3.0,0.01,10.0))
        post_std=actual_std.copy()
        # Empirisk kovarians fra siste segment av monitor_ch
        seg=monitor_ch[-min(5000,len(monitor_ch)):]
        C_emp=np.cov(seg.T)+1e-10*np.eye(N)
        C_prop=(scale*2.38)**2/N*C_emp
        n_recalib+=1
        recalib_log.append({'round':rd,'psrf_max':float(pmax),
                            'ess_min':float(emin),'scale_new':float(scale)})
        if verbose: print(f"  Rekalibrering {n_recalib}: scale→{scale:.4f}. "
                          f"Ekstra burn-in ({burnin//2:,} trekk)...")
        ex_ch,_,acc_ex=_run(burnin//2,f"REKALIB-{n_recalib}",adapt=True)
        monitor_ch=np.concatenate([monitor_ch[-5000:],ex_ch])
        if verbose: print(f"  Ekstra burn-in ferdig. acc={acc_ex:.3f}")

    if verbose:
        print(f"\n--- FASE 3: Produksjonskjøring ({n_production:,} trekk) ---")
        print(f"  Endelig scale: {scale:.4f}")

    prod_ch,lp_prod,acc_prod=_run(n_production,"PROD",monitor=True)

    # Hvis reparametrisering: konverter chain tilbake til naturlig rom før
    # diagnostikk og lagring. _unc.npy beholdes for posterior-analyse i unc-rom.
    prod_ch_internal = prod_ch
    if use_reparam:
        np.save(f"{save_prefix}_unc.npy", prod_ch_internal)
        prod_ch = _chain_natural(prod_ch_internal)

    if verbose:
        print(f"\n  Produksjon ferdig. acc={acc_prod:.3f}  "
              f"Total tid: {(time.time()-t_total)/60:.1f} min")
        print_diagnostics(prod_ch, label="PRODUKSJONSKJØRING")

    np.save(f"{save_prefix}.npy", prod_ch)
    np.save(f"{save_prefix}_lp.npy", lp_prod)
    ps_final=compute_psrf(prod_ch)
    meta={'n_production':n_production,'burnin':burnin,
          'acc_rate':float(acc_prod),'final_scale':float(scale),
          'n_recalibrations':n_recalib,'recalib_log':recalib_log,
          'psrf_final':float(ps_final.max()),
          'sigma_A_fixed':SIGMA_A_FIXED,
          'use_reparam':bool(use_reparam),
          'ess_min':float(min(compute_ess(prod_ch[:,i]) for i in range(N_PARAMS)))}
    with open(f"{save_prefix}_meta.json",'w') as f: json.dump(meta,f,indent=2)

    # Posterior-sammendrag
    summ={}
    for i,name in enumerate(PARAM_NAMES):
        s=prod_ch[:,i]
        summ[name]={'mean':float(s.mean()),'std':float(s.std()),
                    'p5':float(np.percentile(s,5)),'p50':float(np.median(s)),
                    'p95':float(np.percentile(s,95)),'km':KM.get(name,float('nan')),
                    'psrf':float(ps_final[i]),'ess':float(compute_ess(s))}
    summ['sigma_A'] = {'mean':SIGMA_A_FIXED,'std':0.0,'p5':SIGMA_A_FIXED,
                       'p50':SIGMA_A_FIXED,'p95':SIGMA_A_FIXED,
                       'km':0.006,'psrf':1.0,'ess':float('inf'),'fixed':True}
    with open(f"{save_prefix}_posterior.json",'w') as f:
        json.dump({'summary':summ,'meta':meta},f,indent=2)

    if verbose:
        print(f"\n  Lagret: {save_prefix}.npy")
        print(f"          {save_prefix}_lp.npy")
        print(f"          {save_prefix}_meta.json")
        print(f"          {save_prefix}_posterior.json")
    return prod_ch, lp_prod, meta


# ══════════════════════════════════════════════════════════════════════════════
# KJØRING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("Laster data...")
    # Foretrekker pipeline-generert fil; faller tilbake til v2 hvis ikke tilgjengelig
    _data_fil = (
        "data/processed/nemo_data.csv"
        if os.path.exists("data/processed/nemo_data.csv")
        else "data/processed/nemo_data_faktisk_v2.csv"
    )
    print(f"  Datafil: {_data_fil}")
    obs_df=pd.read_csv(_data_fil,index_col=0,parse_dates=True)
    # COVID-hull: ekskluder 2020Q1–2021Q4 (PE-godkjent, Alt A 2026-05-23)
    pre =obs_df[obs_df.index<="2019-12-31"][OBS_NAMES].values
    post=obs_df[obs_df.index>="2022-01-01"][OBS_NAMES].values
    print(f"  Pre={len(pre)} kv  Post={len(post)} kv")
    H=build_H(); Sv=build_Sv()

    # Startverdi: foretrekker data/results/posterior_v3_final.json
    scale_init = 0.676   # fra forrige kjøring
    if os.path.exists("data/results/posterior_v3_final.json"):
        print("Laster startverdi fra data/results/posterior_v3_final.json...")
        with open("data/results/posterior_v3_final.json") as f:
            prev = json.load(f)
        summ_prev = prev['summary']
        theta_start=np.zeros(N_PARAMS); post_std=np.zeros(N_PARAMS)
        for i,name in enumerate(PARAM_NAMES):
            if name in summ_prev:
                theta_start[i] = summ_prev[name]['mean']
                post_std[i]    = max(summ_prev[name]['std'], 1e-4)
            else:
                theta_start[i] = KM.get(name,0.5)
                # Parameter-spesifikke startstandardavvik (Fase 2)
                post_std[i] = {'phi_I1':1.0,'phi_I2':2.0}.get(name,0.05)
        scale_init = prev.get('final_scale', 0.676)
        print(f"  Startscale fra forrige kjøring: {scale_init:.4f}")

    elif os.path.exists("chain_baseline_v3.npy"):
        print("Laster startverdi fra chain_baseline_v3.npy...")
        cp=np.load("chain_baseline_v3.npy")
        NAMES_V2=['rho_A','rho_C','rho_O','rho_Ys','rho_rp','sigma_A',
                  'sigma_C','sigma_O','sigma_Ys','sigma_rp','sigma_i',
                  'sigma_P','psi_R','psi_P1','psi_Y','h_c']
        theta_start=np.zeros(N_PARAMS); post_std=np.zeros(N_PARAMS)
        for i,name in enumerate(PARAM_NAMES):
            if name in NAMES_V2:
                j=NAMES_V2.index(name)
                theta_start[i]=cp.mean(axis=0)[j]; post_std[i]=max(cp.std(axis=0)[j],1e-4)
            elif name=='rho_H': theta_start[i]=0.694; post_std[i]=0.08
            elif name=='sigma_H': theta_start[i]=0.050; post_std[i]=0.015
            else: theta_start[i]=KM.get(name,0.5); post_std[i]=0.05
    else:
        print("Bruker K&M-verdier som startpunkt...")
        theta_start=np.array([KM.get(n,0.5) for n in PARAM_NAMES])
        post_std=np.array([0.05]*N_PARAMS)

    # Test startpunkt
    lp_test=log_posterior(theta_start,H,Sv,pre,post)
    print(f"Startverdi log-posterior: {lp_test:.2f}")
    if not np.isfinite(lp_test):
        print("ADVARSEL: Startverdi ugyldig. Bruker K&M-verdier.")
        theta_start=np.array([KM.get(n,0.5) for n in PARAM_NAMES])
        post_std=np.array([0.05]*N_PARAMS); scale_init=1.0

    # Kjør
    chain,lp_vec,meta=adaptive_mcmc_with_monitoring(
        Y_pre=pre, Y_post=post, H=H, Sv=Sv,
        theta_init=theta_start, post_std_init=post_std,
        n_production=200000, burnin=20000, adapt_every=500,
        check_every=10000, max_recalib=5,
        psrf_thr=1.10, ess_pct_thr=0.02,
        scale_init=scale_init, seed=42, verbose=True,
        save_prefix="data/results/chain_v3_v2",
    )
    print("\nEstimering fullfort.")
