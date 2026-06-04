"""
================================================================================
NEMO FASE II — LIKNINGSSYSTEM
Γ₀ z_t = Γ₁ z_{t-1} + Ψ ε_t + Π η_t
 
Tilstandsvektor (NZ = 49, Alt. A 2026-05-15):
  HUSHOLDNINGER OG KONSUM:
    0  pi        KPI-inflasjon
    1  c_W       Konsum, sparere (W = workers / optimizers)
    2  c_NW      Konsum, låntakere (NW = non-optimizers / borrowers)
    3  c         Aggregert konsum
    4  pi_W      Lønnsinflasjon
    5  w         Reallønn (aggregert)
    6  q_H       Boligpris (Tobin's Q for bolig)
    7  h_W       Boligbeholdning, sparere
    8  h_NW      Boligbeholdning, låntakere
 
  PRODUKSJON OG KAPITAL:
    9  y         BNP (fastland)
    10 l         Sysselsetting
    11 k         Kapital
    12 inv        Investering
    13 mc         Marginal kostnad
    14 q_K        Kapital Tobin's Q
 
  VALUTA OG HANDEL:
    15 rer        Reell valutakurs
    16 x          Eksport
    17 m          Import
    18 pM         Importpris
    19 s          Nominell valutakursendring
 
  FINANSIELL SEKTOR:
    20 i_R        Styringsrente (nominell)
    21 i_D        Innskuddsrente
    22 i_L_W      Utlånsrente husholdninger (sparere)
    23 i_L_NW     Utlånsrente låntakere
    24 b_W        Gjeld sparere (begrenset av LTV)
    25 b_NW       Gjeld låntakere (LTV-bindende)
    26 nb         Bankkapital (net worth bank)
 
  OFFENTLIG SEKTOR:
    27 g          Offentlig konsum
    28 pO         Oljepris (real, AR(1))
 
  LAGG-TILSTANDER:
    29 k_lag      k_{t-1}
    30 inv_lag    inv_{t-1}
    31 h_W_lag    h_W_{t-1}
    32 h_NW_lag   h_NW_{t-1}
    33 i_R_lag    i_{t-1}
    34 rer_lag    rer_{t-1}
    35 w_lag      w_{t-1}
    36 pi_lag     pi_{t-1}  (for mimicking rule)
 
  EKSOGENE AR(1)-PROSESSER:
    37 a          TFP
    38 eps_C      Konsumpreferanse
    39 eps_H      Boligpreferanse
    40 eps_G      Offentlig forbruk
    41 pO         (allerede i 28)  — ikke duplisert
    42 yS         Utenlandsk BNP
    43 eps_rp     Risikopremie
    44 pi_star    Utenlandsk inflasjon
    45 i_star     Utenlandsk rente
    46 eps_phi_h  LTV-sjokk husholdninger
    47 eps_prem   Pengemarkedspremie

  ALT. A (2026-05-15) — VARIABEL KAPITALUTNYTTELSE:
    48 u_K        Kapitalutnyttelse (utilization rate), K&M §2.7

Sjokk (NE = 13):
    0  E_A       TFP
    1  E_C       Konsumpreferanse
    2  E_H       Boligpreferanse
    3  E_G       Offentlig forbruk
    4  E_O       Oljepris
    5  E_Ys      Utenlandsk etterspørsel
    6  E_rp      Risikopremie
    7  E_i       Pengepolitikk
    8  E_P       Prismarkup
    9  E_phi_h   LTV-sjokk husholdninger
    10 E_prem    Pengemarkedspremie
    11 E_I       Investeringsjusteringskost.
    12 E_pi_star Utenlandsk inflasjonssjokk
 
================================================================================
"""
 
import warnings

import numpy as np

from nemo.model.parameters import Parameters
 
# ── Dimensjoner ───────────────────────────────────────────────────────────────
# Alt. A (2026-05-15): NZ 48→49 — variabel kapitalutnyttelse u_t lagt til
# Alt. A2 (2026-06-02, PE-godkjent): NZ 49→50 — AR(2) Taylor-regel (psi_R2)
#   I_R_LL = i_{t-2} er andre lagg av styringsrenten.
#   Exitstrategi: psi_R2=0.0 gir eksakt AR(1)-atferd (NZ=50 beholdes, tom ledd).
NZ = 50
NE = 13

# ── Variabelindekser ─────────────────────────────────────────────────────────
PI=0; C_W=1; C_NW=2; C=3; PIW=4; W=5; Q_H=6; H_W=7; H_NW=8
Y=9; L=10; K=11; INV=12; MC=13; Q_K=14
RER=15; X=16; M=17; PM=18; S=19
I_R=20; I_D=21; I_L_W=22; I_L_NW=23; B_W=24; B_NW=25; NB=26
G=27; PO=28
K_L=29; INV_L=30; H_W_L=31; H_NW_L=32; I_R_L=33; RER_L=34; W_L=35; PI_L=36
A=37; EPS_C=38; EPS_H=39; EPS_G=40
YS=41; EPS_RP=42; PI_STAR=43; I_STAR=44
EPS_PHI_H=45; EPS_PREM=46; EPS_I_ADJ=47  # siste plass: investeringssjokk
U_K=48  # Alt. A: kapitalutnyttelse (utilization rate)
I_R_LL=49  # Alt. A2: 2-periodes lagg av styringsrenten (i_{t-2}) for AR(2) Taylor

# PLT (Fase 2, 2026-06-02): akkumulert prisnivå-gap for prisnivåmål-kanal (NZ 50→51)
# p_gap_t = p_gap_{t-1} + π_t  →  mean-reversion i styringsrenten via psi_PL > 0
# Exitstrategi: psi_PL=0 → eksakt v3_forward-atferd (NZ_PLT beholdes, gap er dead state)
P_STAR_GAP = 50
NZ_PLT     = 51

# Alt B (PE-godkjent 2026-05-29): boliginvesteringskanal — separat INV_H + lagg (NZ 49→51)
INV_H   = 49   # boliginvestering (Euler-ligning med phi_H1)
INV_H_L = 50   # lagg av boliginvestering
NZ_ALTB = 51   # ny tilstandsromdimensjon

# A9 (PE-godkjent 2026-05-22): 7 hjelpetilstander for RE-forventninger (NZ 49→56)
PI_E=49; C_W_E=50; Q_H_E=51; PIW_E=52; INV_E=53; Q_K_E=54; RER_E=55
NZ_V4 = 56

# Alt B (PE-godkjent 2026-05-23): 4-periodes inflasjonsforventningskjede (NZ 49→53)
# Taylor-regel reagerer på E_t[π_{t+4}] — NB NEMO-konvensjon (inflasjonsmål 4Q frem)
PI_E1=49; PI_E2=50; PI_E3=51; PI_E4=52
NZ_PI4 = 53

# GEORG (PE-godkjent 2026-06-04): NBs enkle optimale regel (Staff Memo 15/2025).
# Bygger ved siden av v3_forward (NZ=50). 14 nye tilstander for regelens indikatorer:
#   - π_{t-2}-lagg (4-kv. inflasjon; pi_lag=36 gir π_{t-1})
#   - 2 πW-lagg + 3 a-lagg (4-kv. lønnskostnadsvekst-gap ϕ̂)
#   - 7 s-lagg (8-kv. valutakursvekst-gap Ŝ — full window, PE-valg 2026-06-04)
#   - AR(1) pengepolitikksjokk Z_t (λ_Z), erstatter i.i.d. E_i i regelen
# Exitstrategi: use_georg=False → build_matrices_v3_forward (nye states er dead).
GEORG_PI_L2  = 50   # π_{t-2}
GEORG_PIW_L1 = 51   # πW_{t-1}
GEORG_PIW_L2 = 52   # πW_{t-2}
GEORG_A_L1   = 53   # a_{t-1}
GEORG_A_L2   = 54   # a_{t-2}
GEORG_A_L3   = 55   # a_{t-3}
GEORG_S_L1   = 56   # s_{t-1}
GEORG_S_L2   = 57   # s_{t-2}
GEORG_S_L3   = 58   # s_{t-3}
GEORG_S_L4   = 59   # s_{t-4}
GEORG_S_L5   = 60   # s_{t-5}
GEORG_S_L6   = 61   # s_{t-6}
GEORG_S_L7   = 62   # s_{t-7}
GEORG_Z      = 63   # AR(1) pengepolitikksjokk Z_t
NZ_GEORG     = 64

# Endogen risikopremie i UIP (PE-godkjent 2026-06-04). Bygger på v3_forward (NZ=50),
# +1 tilstand for persistent premie. Exit: kappa_rp_endo=0 → v3_forward.
RP_ENDO   = 50      # endogen risikopremie (AR(1) drevet av rentedifferanse)
NZ_RPENDO = 51

# ── Sjokk-indekser ───────────────────────────────────────────────────────────
E_A=0; E_C=1; E_H=2; E_G=3; E_O=4; E_Ys=5; E_rp=6
E_i=7; E_P=8; E_phi_h=9; E_prem=10; E_I=11; E_piS=12
 
VAR_NAMES = [
    'pi','c_W','c_NW','c','piW','w','q_H','h_W','h_NW',
    'y','l','k','inv','mc','q_K',
    'rer','x','m','pM','s',
    'i_R','i_D','i_L_W','i_L_NW','b_W','b_NW','nb',
    'g','pO',
    'k_lag','inv_lag','h_W_lag','h_NW_lag','i_R_lag','rer_lag','w_lag','pi_lag',
    'a','eps_C','eps_H','eps_G',
    'yS','eps_rp','pi_star','i_star','eps_phi_h','eps_prem','eps_I_adj',
    'u_K',  # Alt. A: kapitalutnyttelse
    # A9: hjelpetilstander for RE-forventninger
    'pi_E','c_W_E','q_H_E','piW_E','inv_E','q_K_E','rer_E',
]
 
SHOCK_NAMES = [
    'TFP','Konsum','Bolig','Off.forbruk','Oljepris',
    'Utenl.ettersp.','Risikopremie','Pengepolitikk','Prismarkup',
    'LTV husholdning','Pengemarkedspremie','Inv.just.kost.','Utenl.inflasjon'
]
 
 
def build_matrices(p=None):
    """
    Bygger G0, G1, Psi, Pi for Fase II-modellen.
 
    Returnerer
    ----------
    G0, G1, Psi, Pi : (NZ×NZ), (NZ×NZ), (NZ×NE), (NZ×NZ)
    """
    if p is None:
        p = Parameters
 
    beta   = p.beta
    h_c    = p.h_c
    phi_L  = p.phi_L
    sigma  = p.sigma
    alpha_K = p.alpha_K
    delta  = p.delta
    delta_H = p.delta_H
    mu_M   = p.mu_M
    mu_X   = p.mu_X
    phi_B  = p.phi_B
    kP     = p.kappa_P()
    kW     = p.kappa_W()
    CY, IY, GY, XY, MY = p.CY, p.IY, p.GY, p.XY, p.MY
    IHY    = p.IHY
    omega  = p.omega_NW        # andel låntakere
    m_H    = p.m_H             # LTV
    gamma_G = p.gamma_G
    kappa_M = p.kappa_M        # importpriskanal (A14.9: fra parameters.py)
 
    # Avledede størrelser
    a1_W = h_c / (1 + h_c)
    a2_W = 1.0 / (1 + h_c)
    a3_W = (1 - h_c) / (sigma * (1 + h_c))
    sigma_tilde = sigma + phi_L / (1 - alpha_K)
 
    # Pengepolitikk: mimicking rule-koeffisienter
    psi_R  = p.psi_R
    psi_P1 = p.psi_P1
    psi_Y  = p.psi_Y
    psi_S  = p.psi_S
    psi_W  = p.psi_W
 
    G0  = np.zeros((NZ, NZ))
    G1  = np.zeros((NZ, NZ))
    Psi = np.zeros((NZ, NE))
    Pi  = np.zeros((NZ, NZ))
 
    # ════════════════════════════════════════════════════════════════════════
    # BLOKK A: PRISSETTING OG LØNN
    # ════════════════════════════════════════════════════════════════════════
 
    # A1. NK Pris-Phillips-kurve med importpriskanal
    # π_t = β·E[π_{t+1}] + κ_P·mc_t + κ_M·(rer_t + π*_t) + ε_P
    G0[0, PI]      =  1.0
    G0[0, MC]      = -kP
    G0[0, RER]     = -kappa_M
    G0[0, PI_STAR] = -kappa_M
    Pi[0, PI]      =  beta
    Psi[0, E_P]    =  1.0
 
    # A2. Lønnsinflasjon (Rotemberg, tilsvarer Calvo i log-linearisert form)
    # π_W = β·E[π_W_{t+1}] + κ_W·(φ_L·l + c/(1-h_c) - w)
    G0[4, PIW]  =  1.0
    G0[4, W]    = -kW
    G0[4, L]    =  kW * phi_L
    G0[4, C]    =  kW / (1.0 - h_c)
    Pi[4, PIW]  =  beta
 
    # A3. Reallønns-dynamikk: w = w_{t-1} + π_W - π
    G0[5, W]    =  1.0
    G0[5, PIW]  = -1.0
    G0[5, PI]   =  1.0
    G1[5, W_L]  =  1.0   # direkte kobling (ikke via lagg-mellomled)
 
    # ════════════════════════════════════════════════════════════════════════
    # BLOKK B: HUSHOLDNINGER
    # Sparere (W) og låntakere (NW), aggregat
    # ════════════════════════════════════════════════════════════════════════
 
    # B1. Euler-likning, sparere (W)
    # c_W = a1_W·c_W_{t-1} + a2_W·E[c_W_{t+1}] - a3_W·(i_D - E[π_{t+1}]) + ε_C
    G0[1, C_W]   =  1.0
    G0[1, I_D]   =  a3_W        # innskuddsrente (ikke styringsrente direkte)
    G1[1, C_W]   =  a1_W
    Pi[1, C_W]   =  a2_W
    Pi[1, PI]    = -a3_W
    Psi[1, E_C]  =  a2_W
 
    # B2. Euler-likning, låntakere (NW) — bindende LTV-betingelse
    # c_NW ≈ (1/β_NW)·(m_H·E[q_H_{t+1}] - b_NW) + lønnsinntekt
    # Forenklet: c_NW = (1-m_H)·(w+l) + netto LTV-kanal
    # Full implementering: kolateralkanal via q_H og b_NW
    G0[2, C_NW]  =  1.0
    G0[2, B_NW]  =  (1.0 - m_H) / beta   # netto LTV-kanal
    G0[2, W]     = -(1.0 - m_H)           # reallønnskanal
    G0[2, L]     = -(1.0 - m_H)           # sysselsettingskanal
    Pi[2, Q_H]   =  m_H / beta            # E[q_H_{t+1}]: kollateralverdi
    G0[2, EPS_C] = -a2_W    # A11.1: koble AR(1)-state EPS_C (delt preferansesjokk)
 
    # B3. Aggregert konsum: c = (1-ω)·c_W + ω·c_NW
    G0[3, C]    =  1.0
    G0[3, C_W]  = -(1.0 - omega)
    G0[3, C_NW] = -omega
 
    # B4. Boligetterspørsel, sparere
    # q_H = E[q_H_{t+1}]·(1-δ_H)/((i_D - E[π_{t+1}])) + bolignytte
    G0[6, Q_H]  =  1.0
    G0[6, I_D]  =  1.0
    G0[6, PI]   = -1.0
    G1[6, H_W_L]=  1.0     # lagg via H_W_lag (ligning for kapitalakkumulering)
    Pi[6, Q_H]  =  (1.0 - delta_H)
    Psi[6, E_H] =  1.0     # boligpreferansesjokk
 
    # B5. Boligakkumulering, sparere: h_W = (1-δ_H)·h_W_{t-1} + inv_H_W
    # Forenklet (ingen separate boliginvesteringer): h_W = h_W_{t-1}·(1-δ_H) + inv_H
    G0[7, H_W]   =  1.0
    G1[7, H_W_L] =  (1.0 - delta_H)
    G0[7, Q_H]   = -delta_H   # boliginvestering proporsjonal med q_H
 
    # B6. Boligbeholdning, låntakere (LTV-bindende)
    # b_NW = m_H · (1+i_L_NW) · q_H · h_NW  — LTV-betingelse
    G0[8, H_NW]    =  1.0
    G1[8, H_NW_L]  =  (1.0 - delta_H)
    G0[8, Q_H]     = -delta_H
 
    # ════════════════════════════════════════════════════════════════════════
    # BLOKK C: PRODUKSJON OG KAPITAL
    # ════════════════════════════════════════════════════════════════════════
 
    # C1. BNP (varemarkedsklarering) — Spor A5 rettelse fullført (2026-05-18)
    # MY justert til 0.28 (fastlands-import, uten oljesektor).
    # IHY inkludert i INV-koeffisienten siden modellen ikke har separat INV_H-variabel.
    # Sjekk: CY+(IY+IHY)+GY+XY-MY = 0.50+0.30+0.25+0.23-0.28 = 1.00 ✓
    G0[9, Y]    =  1.0
    G0[9, C]    = -CY
    G0[9, INV]  = -(IY + IHY)   # total investering = kapital + bolig
    G0[9, G]    = -GY
    G0[9, X]    = -XY
    G0[9, M]    =  MY
 
    # C2. Sysselsetting (fra produksjonsfunksjon)
    G0[10, L]   =  1.0
    G0[10, Y]   = -1.0 / (1.0 - alpha_K)
    G0[10, K_L] =  alpha_K / (1.0 - alpha_K)  # kapital fra forrige periode
    G0[10, A]   =  1.0 / (1.0 - alpha_K)
 
    # C3. Kapitalakkumulering MED justeringskostnader
    # k = (1-δ)·k_{t-1} + [1 - S(inv/inv_{t-1})]·inv
    # Log-linearisert: k = (1-δ)·k_{t-1} + δ·inv  (S''=0 gir ren akkumulering)
    # Fase II: S(inv/inv_{t-1}) introduserer inv_lag:
    # k = (1-δ)·k_{t-1} + δ·(1 + φ_I1·(inv - inv_{t-1}))·inv
    # Forenklet første-ordens:
    G0[11, K]     =  1.0
    G0[11, INV]   = -delta
    G1[11, K_L]   =  (1.0 - delta)
 
    # C4. Investeringslikning (Tobin's Q med justeringskostnader)
    # q_K = E[r_K_{t+1}] + (1-δ)·E[q_K_{t+1}] - (i_D - E[π_{t+1}])
    # + φ_I1·(inv - inv_{t-1}) - φ_I2·E[(inv_{t+1} - inv)]
    G0[12, INV]   =  1.0
    G0[12, Q_K]   = -1.0 / (p.phi_I1 + p.phi_I2)  # Q-inverter justeringskost.
    G0[12, INV_L] =  p.phi_I1 / (p.phi_I1 + p.phi_I2)
    Pi[12, INV]   =  p.phi_I2 / (p.phi_I1 + p.phi_I2)  # fremoverskuende justeringskost.
    Psi[12, E_I]  =  1.0
 
    # C5. Marginal kostnad fra MRS=MPN (konsistent med Fase I)
    G0[13, MC]    =  1.0
    G0[13, Y]     = -sigma_tilde
    G0[13, A]     =  (1.0 + phi_L / (1.0 - alpha_K))
 
    # C6. Kapital Tobin's Q
    G0[14, Q_K]   =  1.0
    G0[14, I_D]   =  1.0
    G0[14, PI]    = -1.0
    Pi[14, Q_K]   =  (1.0 - delta)
 
    # ════════════════════════════════════════════════════════════════════════
    # BLOKK D: VALUTA OG HANDEL
    # ════════════════════════════════════════════════════════════════════════
 
    # D1. UIP med AR(1)-glatting, gjeldselastisk premie og olje-valuta-kanal
    # Fase 1B (PE-godkjent 2026-05-26): delvis-justeringsform (Justiniano & Preston 2010)
    #   rer_t = rho_s·rer_{t-1} + (1-rho_s)·[E_t[rer_{t+1}] - (i_D-π) + (i*-π*) + ε_rp + ...]
    # rho_s=0 → ren UIP (bakoverkompatibel); rho_s>0 demper umiddelbar RER-respons.
    # Mekanisme: høy rho_s reduserer BNP-overreaksjon ved pengepolitikk-sjokk.
    phi_O = p.phi_O
    rho_s = getattr(p, 'rho_s', 0.0)
    _w    = 1.0 - rho_s                 # vekt på UIP-forventningsledd
    G0[15, RER]       =  1.0
    G0[15, I_D]       =  _w
    G0[15, PI]        = -_w
    G0[15, I_STAR]    = -_w
    G0[15, PI_STAR]   =  _w
    G0[15, EPS_PREM]  = -_w
    G0[15, EPS_RP]    = -_w            # Funn A: kobler persistent risikopremie-AR(1) inn i UIP
    G0[15, B_NW]      =  _w * phi_B
    G0[15, PO]        =  _w * phi_O
    G1[15, RER]       =  rho_s          # lagget RER-ledd
    Pi[15, RER]       =  _w
    # Funn A: Psi[15, E_rp] fjernet — sjokket går via EPS_RP-tilstanden (rad 42)
    # Funn B: Psi[15, E_prem] fjernet — sjokket går via EPS_PREM-tilstanden (rad 46)
 
    # D2. Eksportetterspørsel (Armington, korrigert µ)
    G0[16, X]   =  1.0
    G0[16, RER] = -mu_X
    G0[16, YS]  = -1.0
 
    # D3. Import (korrigert µ)
    G0[17, M]   =  1.0
    G0[17, PM]  =  mu_M
    G0[17, PI]  = -mu_M
    G0[17, C]   = -CY
    G0[17, G]   = -GY
    G0[17, INV] = -(IY + IHY)
 
    # D4. Importpris
    G0[18, PM]      =  1.0
    G0[18, RER]     = -1.0
    G0[18, PI_STAR] = -1.0
 
    # D5. Nominell valutakurs (residual)
    G0[19, S]    =  1.0
    G0[19, RER]  = -1.0
    G0[19, PI]   = -1.0
    G0[19, PI_STAR] =  1.0
    G1[19, RER]  =  1.0
 
    # ════════════════════════════════════════════════════════════════════════
    # BLOKK E: FINANSIELL SEKTOR
    # Gerali et al. (2010) forenklet
    # ════════════════════════════════════════════════════════════════════════
 
    # E1. Mimicking rule (erstatter Taylor-regel fra Fase I)
    # i_R = ψ_R·i_R_{t-1} + (1-ψ_R)·[ψ_P1·E[π_{t+4}] + ψ_Y·y + ψ_S·rer + ψ_W·π_W] + ε_i
    # Fase II-implementering: bruker π_{t-1} (lagg) for ψ_P1-leddet i første iterasjon
    # Fremoverskuende π: E[π_{t+4}] ≈ ψ_P1·π_t (forenkling for BK-løsning)
    # NB: Denne v1-versjonen er bevart for å holde BK-stabilitet ved default psi_R=0.666.
    # v3 (build_matrices_v3) overstyrer med den korrigerte mimicking rule fra Spor A4b.
    G0[20, I_R]    =  1.0
    G0[20, Y]      = -(1.0 - psi_R) * psi_Y
    G0[20, RER]    = -(1.0 - psi_R) * psi_S
    G1[20, I_R_L]  =  psi_R
    G1[20, PI_L]   =  (1.0 - psi_R) * psi_P1   # lagg av inflasjon
    G1[20, PIW]    =  (1.0 - psi_R) * psi_W * 0.0   # lønnsvekt (0 i forenkling)
    Psi[20, E_i]   =  1.0
 
    # E2. Innskuddsrente (bank, under ufullkommen konkurranse)
    # i_D = i_R - spread_D + ε_prem
    # Spread avhenger av kapitaldekning: spread_D = φ_D·(nb - γ_b·aktiva)
    G0[21, I_D]      =  1.0
    G0[21, I_R]      = -1.0
    G0[21, NB]       =  p.phi_c   # kapitaldekning-kanal
    G0[21, EPS_PREM] = -1.0
 
    # E3. Utlånsrente, sparere (W)
    # i_L_W = i_R + spread_L_W - ε_prem
    G0[22, I_L_W]   =  1.0
    G0[22, I_R]     = -1.0
    G0[22, NB]      = -p.phi_c
    G0[22, EPS_PHI_H] = +1.0   # A4c-konsistens 2026-05-18 (PE): strammere LTV → høyere spread
 
    # E4. Utlånsrente, låntakere (NW) — høyere spread
    G0[23, I_L_NW]  =  1.0
    G0[23, I_R]     = -1.0
    G0[23, NB]      = -1.5 * p.phi_c   # høyere spread for låntakere
    G0[23, EPS_PHI_H] = +1.0   # A4c-konsistens 2026-05-18 (PE): strammere LTV → høyere spread
 
    # E5. Gjeld, sparere (ikke-bindende)
    # b_W: finansiell formueakkumulering sparere
    G0[24, B_W]     =  1.0
    G0[24, I_L_W]   = -(1.0 - omega)
    G0[24, Y]       = -(1.0 - omega)
 
    # E6. Gjeld, låntakere (LTV-bindende)
    # b_NW = m_H · q_H · h_NW / (1 + i_L_NW)
    # A4c-rettelse (2026-05-18): positivt LTV-sjokk = strammere LTV → mindre gjeld.
    # Konsistent med renteoppgang i lign. 22-23 ved samme sjokk.
    G0[25, B_NW]    =  1.0
    G0[25, Q_H]     = -m_H
    G0[25, H_NW]    = -m_H
    G0[25, I_L_NW]  =  m_H
    Psi[25, E_phi_h] = -1.0   # strammere LTV → mindre gjeld (PE-godkjent 2026-05-18)
 
    # E7. Bankkapital-akkumulering (Gerali et al. 2010)
    # A4a-rettelse (2026-05-18, PE-godkjent): bytte til G1-akkumulering.
    # nb_t = (1-δ_b)·nb_{t-1} + φ_o·(i_R_{t-1} + b_NW_{t-1})
    # Tidligere bug: G0[26,NB] += phi_c ga G0[26,NB] = 11.0; ingen lagg-ledd.
    # phi_c-leddet er fjernet her — det inngår i spread-likningene (21-23).
    G0[26, NB]     =  1.0
    G1[26, NB]     =  (1.0 - p.delta_b)   # akkumulering, δ_b = 0.0161
    G1[26, I_R]    = -p.phi_o             # spread-inntekt fra forrige periode
    G1[26, B_NW]   = -p.phi_o
 
    # ════════════════════════════════════════════════════════════════════════
    # BLOKK F: OFFENTLIG SEKTOR
    # Forenklet fiskalregel (GPFG utvides i neste iterasjon)
    # ════════════════════════════════════════════════════════════════════════
 
    # F1. Offentlig konsum (fiskalregel med lagg + AR(1)-sjokk)
    # A11.1 (PE-godkjent 2026-05-21): koble AR(1)-state EPS_G (var dead state)
    G0[27, G]     =  1.0
    G1[27, PO]    =  gamma_G
    G0[27, EPS_G] = -1.0   # A11.1: AR(1)-persistens via EPS_G-state
 
    # F2. Oljepris AR(1)
    G0[28, PO]    =  1.0
    G1[28, PO]    =  p.rho_O
    Psi[28, E_O]  =  1.0
 
    # ════════════════════════════════════════════════════════════════════════
    # LAGG-IDENTITETER (direkte koblinger — ikke via mellomled)
    # ════════════════════════════════════════════════════════════════════════
 
    G0[29, K_L]=1.0;   G1[29, K]=1.0       # k_{t} = k_{t-1}
    G0[30, INV_L]=1.0; G1[30, INV]=1.0     # inv_{t} = inv_{t-1}
    G0[31, H_W_L]=1.0; G1[31, H_W]=1.0    # h_W_{t} = h_W_{t-1}
    G0[32, H_NW_L]=1.0;G1[32, H_NW]=1.0  # h_NW_{t} = h_NW_{t-1}
    G0[33, I_R_L]=1.0;  G1[33, I_R]=1.0    # i_{t} = i_{t-1}
    G0[49, I_R_LL]=1.0; G1[49, I_R_L]=1.0  # i_{t-1} = i_{t-2}  (AR(2)-lagg)
    G0[34, RER_L]=1.0; G1[34, RER]=1.0    # rer_{t} = rer_{t-1}
    G0[35, W_L]=1.0;   G1[35, W]=1.0      # w_{t} = w_{t-1}
    G0[36, PI_L]=1.0;  G1[36, PI]=1.0     # pi_{t} = pi_{t-1}
 
    # ════════════════════════════════════════════════════════════════════════
    # AR(1)-PROSESSER
    # ════════════════════════════════════════════════════════════════════════
 
    G0[37,A]=1.0;     G1[37,A]=p.rho_A;     Psi[37,E_A]=1.0
    G0[38,EPS_C]=1.0; G1[38,EPS_C]=p.rho_C; Psi[38,E_C]=1.0
    G0[39,EPS_H]=1.0; G1[39,EPS_H]=p.rho_H; Psi[39,E_H]=1.0
    G0[40,EPS_G]=1.0; G1[40,EPS_G]=p.rho_G; Psi[40,E_G]=1.0
    # PO allerede håndtert i F2 (indeks 28)
    G0[41,YS]=1.0;    G1[41,YS]=p.rho_Ys;   Psi[41,E_Ys]=1.0
    G0[42,EPS_RP]=1.0;G1[42,EPS_RP]=p.rho_rp;Psi[42,E_rp]=1.0
    G0[43,PI_STAR]=1.0;G1[43,PI_STAR]=p.rho_piS;Psi[43,E_piS]=1.0
    G0[44,I_STAR]=1.0; G1[44,I_STAR]=p.rho_iS;   # Funn C: utenlandsk rente bruker rho_iS, ikke rho_piS
    G0[45,EPS_PHI_H]=1.0;G1[45,EPS_PHI_H]=p.rho_phi_h;Psi[45,E_phi_h]=1.0
    G0[46,EPS_PREM]=1.0; G1[46,EPS_PREM]=p.rho_prem; Psi[46,E_prem]=1.0
    G0[47,EPS_I_ADJ]=1.0;G1[47,EPS_I_ADJ]=p.rho_I;  Psi[47,E_I]=1.0

    # Alt. A bakoverkompatibilitet: U_K eksisterer som tilstand men er triviell
    # i v1/v2 (settes til 0 via identitetsligning)
    G0[U_K, U_K] = 1.0

    return G0, G1, Psi, Pi
 
 
if __name__ == "__main__":
    G0, G1, Psi, Pi = build_matrices()
    print(f"G0 dimensjon: {G0.shape}")
    print(f"Rang G0: {np.linalg.matrix_rank(G0)} av {NZ}")
    print(f"Kondisjon G0: {np.linalg.cond(G0):.1f}")
    print(f"Psi dimensjon: {Psi.shape}")
    print(f"Ikke-null i G0: {np.count_nonzero(np.abs(G0) > 1e-12)}")
    print(f"Ikke-null i G1: {np.count_nonzero(np.abs(G1) > 1e-12)}")
 
 
def build_matrices_v2(p=None):
    """
    Fase II v2 — med korrekte koblinger for kapital, Q_K og mc.
    Alle tre fikser fra debugging er innarbeidet:
      Fix 1: MC = sigma_tilde*y - (1+phi_L/(1-alphaK))*a - alphaK/(1-alphaK)*k_lag
      Fix 2: Q_K inkluderer r_K avkastningsledd (alphaK * mc)
      Fix 3: INV = (1/phi_I1)*q_K med fremoverskuende justeringskostnader
    Bestått: 15/15 kvalitative IRF-krav (TFP validert t=9..20).
    """
    if p is None:
        p = Parameters
 
    G0, G1, Psi, Pi = build_matrices(p)
 
    alpha_K = p.alpha_K
    delta   = p.delta
    sigma_t = p.sigma + p.phi_L / (1.0 - alpha_K)
 
    # Fix 1: MC med kapitalkanal
    G0[MC,:]=0; G1[MC,:]=0
    G0[MC, MC] =  1.0
    G0[MC, Y]  = -sigma_t
    G0[MC, A]  =  (1.0 + p.phi_L / (1.0 - alpha_K))
    G1[MC, K_L] = -alpha_K / (1.0 - alpha_K)
 
    # Fix 2: Q_K med r_K
    G0[Q_K,:]=0; G1[Q_K,:]=0; Pi[Q_K,:]=0
    G0[Q_K, Q_K] =  1.0
    G0[Q_K, I_R] =  1.0
    G0[Q_K, PI]  = -1.0
    G0[Q_K, MC]  = -alpha_K
    G0[Q_K, Y]   = -alpha_K
    G1[Q_K, K_L] = -alpha_K
    Pi[Q_K, Q_K] =  (1.0 - delta)
    Pi[Q_K, PI]  = -1.0
 
    # Fix 3: INV
    G0[INV,:]=0; G1[INV,:]=0; Psi[INV,:]=0; Pi[INV,:]=0
    G0[INV, INV] =  1.0
    G0[INV, Q_K] = -1.0 / p.phi_I1
    G1[INV, INV_L] =  p.phi_I1 / (p.phi_I1 + p.phi_I2)
    Pi[INV, INV] =  p.phi_I2 / (p.phi_I1 + p.phi_I2)
    Psi[INV, E_I] = 1.0
 
    return G0, G1, Psi, Pi
 
 
def build_matrices_v3(
    p=None,
    theta_H: float = 0.05,
    psi_UIP: float = 0.0,
    fwd_housing_weight: float | None = None,
):
    """
    NEMO Fase II v3 — Fullt estimeringsklart likningssystem.

    Bygger på build_matrices_v2 og legger til:
      1. Boligpreferanse-kalibrering via theta_H (skalering av E_H-sjokket)
      2. Stabil boligprislikning med mean-reversion (Gelain et al. 2018)
      3. Bakseende forventningsdannelse for boligpriser (b_sa, lambda_sa)
      4. Korrekt h_c-oppdatering fra estimerte parametere
      5. Mimicking rule med estimert psi_R, psi_P1, psi_Y

    Parametere
    ----------
    p                  : Parameters-klasse (eller underklasse med oppdaterte estimater)
    theta_H            : Skaleringsfaktor for boligpreferansesjokket (default 0.05)
    psi_UIP            : Valutarisikopremie i UIP-likning (default 0.0 = ren UIP).
                         PE-godkjent verdi: 0.02 (A9b, 2026-05-22).
                         Setter G0[15, RER] = 1.0 + psi_UIP (bryter enhetsroten λ=1→1+ψ).
    fwd_housing_weight : Fremoverskuende vekt for boligprisforventning Pi[6, Q_H].
                         None (default) = bruk K&M-kalibrering (w_fwd ≈ 0.393).
                         0.0 = fullt bakseende boligprisforventninger (BK-kandidat).
                         Verdier i [0, 1] interpolerer mellom de to ytterpunktene.

    Returnerer
    ----------
    G0, G1, Psi, Pi : (NZ×NZ), (NZ×NZ), (NZ×NE), (NZ×NZ)
    """
    if p is None:
        p = Parameters
 
    # Start fra v2 (inkluderer alle v1-fikser)
    G0, G1, Psi, Pi = build_matrices_v2(p)
 
    # ── 1. Oppdater h_c fra estimerte parametere ──────────────────────────────
    # build_matrices_v2 arver h_c fra p, men noen koblinger beregnet
    # med avledede konstanter må oppdateres eksplisitt.
    h_c    = p.h_c
    beta   = p.beta
    delta_H = p.delta_H
    sigma  = p.sigma
    omega  = p.omega_NW
    m_H    = p.m_H
 
    a1_W = h_c / (1.0 + h_c)
    a2_W = 1.0 / (1.0 + h_c)
    a3_W = (1.0 - h_c) / (sigma * (1.0 + h_c))
 
    # Oppdater Euler-likning sparere (ligning 1) med korrekt h_c
    G0[1, :] = 0.0; G1[1, :] = 0.0; Pi[1, :] = 0.0; Psi[1, :] = 0.0
    G0[1, C_W]  =  1.0
    G0[1, I_D]  =  a3_W
    G1[1, C_W]  =  a1_W
    Pi[1, C_W]  =  a2_W
    Pi[1, PI]   = -a3_W
    G0[1, EPS_C] = -a2_W    # A11.1 (PE-godkjent 2026-05-21): koble AR(1)-state EPS_C

    # Oppdater låntaker-likning (ligning 2) med korrekt h_c
    G0[2, :] = 0.0; G1[2, :] = 0.0; Pi[2, :] = 0.0; Psi[2, :] = 0.0
    G0[2, C_NW]  =  1.0
    G0[2, B_NW]  =  (1.0 - m_H) / beta
    G0[2, W]     = -(1.0 - m_H)
    G0[2, L]     = -(1.0 - m_H)
    Pi[2, Q_H]   =  m_H / beta
    Psi[2, E_C]  =  a2_W
 
    # ── 2. Boligprislikning med mean-reversion (Gelain et al. 2018) ───────────
    # Erstatter ligning 6 fra v1/v2 med stabil versjon:
    # q_H = b_sa·[lambda_sa·q_H_{t-1} + (1-lambda_sa)·E[q_H_{t+1}]]
    #       + (1-b_sa)·E[q_H_{t+1}]
    #       - (i_D - E[π_{t+1}]) + θ_H·ε_H
    # = [b_sa·lambda_sa]·q_H_{t-1}
    #   + [(1-b_sa) + b_sa·(1-lambda_sa)]·E[q_H_{t+1}]
    #   - (i_D - E[π_{t+1}]) + θ_H·ε_H
    #
    # Med b_sa = 0.6393, lambda_sa = 0.9495 (Tabell 8):
    #   Bakseende vekt : b_sa × lambda_sa ≈ 0.607
    #   Fremoverskuende: 1 - b_sa × lambda_sa ≈ 0.393
    b_sa      = getattr(p, 'b_sa',      0.6393)
    lambda_sa = getattr(p, 'lambda_sa', 0.9495)

    w_back     = b_sa * lambda_sa                   # bakseende vekt (K&M ≈ 0.607)
    w_fwd_kalm = 1.0 - w_back                       # K&M fremoverskuende vekt (≈ 0.393)
    # Alt D: fwd_housing_weight kontrollerer Pi[6,Q_H] og Pi[6,PI].
    # fwd_housing_weight=0.0: begge Pi[6,*]=0 (alle fremoverskuende boligledd fjernes).
    # G1[6,Q_H] beholdes alltid = w_back (K&M-kalibrert, endres ikke).
    w_fwd_eff = w_fwd_kalm if fwd_housing_weight is None else float(fwd_housing_weight)
    # Skaleringsfaktor for Pi[6,PI]: følger samme innstramming som Q_H-vekten
    pi_scale  = (w_fwd_eff / w_fwd_kalm) if w_fwd_kalm > 0 else 0.0

    G0[6, :] = 0.0; G1[6, :] = 0.0; Pi[6, :] = 0.0; Psi[6, :] = 0.0
    G0[6, Q_H]    =  1.0
    G0[6, I_D]    =  1.0
    G0[6, PI]     = -1.0
    G1[6, Q_H]    =  w_back       # bakseende vekt (K&M, beholdes alltid)
    Pi[6, Q_H]    =  w_fwd_eff    # fremoverskuende Q_H (0.0 = fjernet)
    Pi[6, PI]     = -pi_scale     # E[π_{t+1}]: skaleres proporsjonalt med fwd_vekt
    G0[6, EPS_H]  = -theta_H      # A11.1: koble AR(1)-state EPS_H (skalert boligsjokk)
 
    # ── 3. Boligakkumulering v3 (stabilisert) ────────────────────────────────
    # Ligning 7: h_W = (1-δ_H)·h_W_{t-1} + δ_H·q_H
    # Beholder v2-versjon men sikrer symmetri mellom sparere og låntakere
    G0[7, :] = 0.0; G1[7, :] = 0.0
    G0[7, H_W]   =  1.0
    G1[7, H_W_L] =  (1.0 - delta_H)
    G0[7, Q_H]   = -delta_H
 
    # Ligning 8: h_NW (låntakere) — identisk struktur
    G0[8, :] = 0.0; G1[8, :] = 0.0
    G0[8, H_NW]    =  1.0
    G1[8, H_NW_L]  =  (1.0 - delta_H)
    G0[8, Q_H]     = -delta_H
 
    # ── 4. Oppdater mimicking rule med estimerte parametere ───────────────────
    # AR(2) Taylor-regel (Alt. A2, PE-godkjent 2026-06-02):
    #   i_t = psi_R·i_{t-1} + psi_R2·i_{t-2}
    #         + (1 - psi_R - psi_R2)·[psi_P1·π_t + psi_Y·y + psi_S·rer + psi_W·πW] + ε_i
    # psi_R2 < 0 gir mean-reversion; psi_R2=0.0 → eksakt AR(1) (exitstrategi).
    psi_R  = p.psi_R
    psi_R2 = p.psi_R2
    psi_P1 = p.psi_P1
    psi_Y  = p.psi_Y
    psi_S  = p.psi_S
    psi_W  = p.psi_W
    _scale = 1.0 - psi_R - psi_R2   # langsiktig nøytralitetsbetingelse

    G0[20, :] = 0.0; G1[20, :] = 0.0; Psi[20, :] = 0.0
    G0[20, I_R]    =  1.0
    G0[20, Y]      = -_scale * psi_Y
    G0[20, RER]    = -_scale * psi_S
    G0[20, PI]     = -_scale * psi_P1   # samtid inflasjon
    G0[20, PIW]    = -_scale * psi_W    # A7 (PE-godkjent 2026-05-21)
    G0[20, I_R_L]  = -psi_R             # 1-periodes lagg
    G0[20, I_R_LL] = -psi_R2            # 2-periodes lagg (AR(2)); 0 → AR(1)
    Psi[20, E_i]   =  1.0

    # ── 5. Rettelse systemic lag-state bug (Spor A4a/A4c, 2026-05-15) ────────
    # G1 på lagg-tilstander (K_L, INV_L, H_W_L, H_NW_L, W_L) gir 2-periodes
    # lagg: G1[r, X_L] * X_L_{t-1} = X_L_{t-1} = X_{t-2} (feil).
    # Rettelse: G0[r, X_L] = −koeff  →  X_L_t = X_{t-1} (korrekt 1-periodes lagg).
    # Se docs/oppgaver/A_funn_rapport.md for full analyse.
    _delta   = p.delta
    _delta_H = p.delta_H  # re-bruker allerede satt delta_H
    _phi_I1  = p.phi_I1
    _phi_I2  = p.phi_I2
    _alpha_K = p.alpha_K
    _sigma_t = p.sigma + p.phi_L / (1.0 - _alpha_K)

    # Ligning 5: reallønn  w_t = w_{t-1} + π_W_t − π_t
    G0[5, :] = 0.0; G1[5, :] = 0.0
    G0[5, W]    =  1.0
    G0[5, PIW]  = -1.0
    G0[5, PI]   =  1.0
    G0[5, W_L]  = -1.0                          # 1-periodes lagg: W_L_t = W_{t-1}

    # Ligning 7: h_W_t = (1−δ_H)·h_W_{t-1} + δ_H·q_H_t
    G0[7, :] = 0.0; G1[7, :] = 0.0
    G0[7, H_W]   =  1.0
    G0[7, H_W_L] = -(1.0 - _delta_H)            # 1-periodes lagg
    G0[7, Q_H]   = -_delta_H

    # Ligning 8: h_NW_t = (1−δ_H)·h_NW_{t-1} + δ_H·q_H_t
    G0[8, :] = 0.0; G1[8, :] = 0.0
    G0[8, H_NW]    =  1.0
    G0[8, H_NW_L]  = -(1.0 - _delta_H)          # 1-periodes lagg
    G0[8, Q_H]     = -_delta_H

    # Ligning 11: k_t = (1−δ)·k_{t-1} + δ·inv_t
    G0[11, :] = 0.0; G1[11, :] = 0.0
    G0[11, K]     =  1.0
    G0[11, INV]   = -_delta
    G0[11, K_L]   = -(1.0 - _delta)             # 1-periodes lagg

    # Ligning 12: investering (Tobin's Q med justeringskostnader, CEE 2005)
    # CEE-FOC: q_K_t = φ_I1·(1+β)·inv_t − φ_I1·inv_{t-1} − β·φ_I1·E[inv_{t+1}]
    # → inv_t = (1/(φ_I1·(1+β)))·q_K + (1/(1+β))·inv_{t-1} + (β/(1+β))·E[inv_{t+1}]
    # Rettelse 2026-05-18 (PE-godkjent): manglende (1+β)-faktor på Q_K-koeff.
    _beta = p.beta
    G0[12, :] = 0.0; G1[12, :] = 0.0; Psi[12, :] = 0.0; Pi[12, :] = 0.0
    G0[12, INV]   =  1.0
    G0[12, Q_K]   = -1.0 / (_phi_I1 * (1.0 + _beta))   # CEE-korrekt
    G0[12, INV_L] = -(1.0 / (1.0 + _beta))              # CEE: 1/(1+β) bakover → røtter {1, 1/β}
    Pi[12, INV]   =  _beta / (1.0 + _beta)              # CEE: β/(1+β) fremover
    G0[12, EPS_I_ADJ] = -1.0   # A11.1: koble AR(1)-state EPS_I_ADJ

    # Ligning 13: marginal kostnad  mc_t = σ̃·y_t − (1+φ_L/(1-α))·a_t − α/(1-α)·k_{t-1}
    # (v2-fix brukte G1[MC, K_L] = −α/(1-α) → K_{t-2}; rettelse: G0[MC, K_L] = +α/(1-α))
    G0[MC, :] = 0.0; G1[MC, :] = 0.0
    G0[MC, MC]   =  1.0
    G0[MC, Y]    = -_sigma_t
    G0[MC, A]    =  (1.0 + p.phi_L / (1.0 - _alpha_K))
    G0[MC, K_L]  =  _alpha_K / (1.0 - _alpha_K)  # 1-periodes lagg (K_L_t = K_{t-1})

    # Ligning 14: Tobin's Q (A4d-rettelse, PE-godkjent 2026-05-21)
    # r̂_K = mc + y − k̂  (leiepris log-avvik fra SS, koeff=1.0 på y-k̂-ledd)
    # Hybrid: MC beholder α_K, mens (y−k̂) bruker 1.0 — ref. A_funn_rapport.md §A4d.
    # Effekt: TFP-sjokk gir positiv BNP (test_09 bestått), KPI q4 ≈ 0.98× NB.
    G0[Q_K, :] = 0.0; G1[Q_K, :] = 0.0; Pi[Q_K, :] = 0.0
    G0[Q_K, Q_K] =  1.0
    G0[Q_K, I_R] =  1.0
    G0[Q_K, PI]  = -1.0
    G0[Q_K, MC]  = -_alpha_K                      # kostnadskomponent: α_K·mc
    G0[Q_K, Y]   = -1.0                           # A4d: output-koeff = 1.0 (ikke α_K)
    G0[Q_K, K_L] = +1.0                           # A4d: kapital-koeff = 1.0
    G0[Q_K, U_K] = +1.0                           # A4d: utnyttelse-koeff = 1.0
    Pi[Q_K, Q_K] =  (1.0 - _delta)
    Pi[Q_K, PI]  = -1.0

    # ── 6. Alt. A (2026-05-15): variabel kapitalutnyttelse ────────────────────
    # Gjenoppretting av K&M (2019) §2.7-spesifikasjon. φ_u=0.2192 (Tabell 8).
    # k̂_t = k_{t-1} + u_t (log-deviasjoner: effektiv kapital)
    # FOC for u_t:  r_K_t = φ_u · u_t
    # hvor r_K_t = α·MC_t + α·Y_t − α·K_L_t − α·U_K_t
    # → (α + φ_u)·U_K = α·MC + α·Y − α·K_L
    _phi_u = p.phi_u
    G0[U_K, :] = 0.0; G1[U_K, :] = 0.0; Pi[U_K, :] = 0.0
    G0[U_K, U_K] =  (_alpha_K + _phi_u)
    G0[U_K, MC]  = -_alpha_K
    G0[U_K, Y]   = -_alpha_K
    G0[U_K, K_L] = +_alpha_K

    # Modifisere L-ligning (10) og MC-ligning (13) til å bruke k̂ = K_L + U_K
    G0[10, U_K] = _alpha_K / (1.0 - _alpha_K)  # produksjonsfunksjon: l avh. av k̂
    G0[MC, U_K] = _alpha_K / (1.0 - _alpha_K)  # mc avh. av k̂

    # ── 7. Alt D: psi_UIP — valutarisikopremie i UIP-likning ─────────────────
    # G0[15, RER] = 1.0 + psi_UIP bryter enhetsroten λ=1.0 → 1.0+ψ > 1.
    # PE-godkjent verdi: 0.02 (A9b, 2026-05-22). Default 0.0 = ren UIP (v3 standard).
    if psi_UIP != 0.0:
        G0[15, RER] = 1.0 + psi_UIP

    # ── 8. Hybrid NK Phillips-kurve: γ_p (Calvo-prisindeksasjon) ─────────────
    # PE-godkjent 2026-05-24. Basis: K&M Tabell 8 γ_p ≈ 0.35.
    # Hybrid form: π_t = [γ_p/(1+β·γ_p)]·π_{t-1} + [β/(1+β·γ_p)]·E[π_{t+1}]
    #                   + [κ_P/(1+β·γ_p)]·mc_t + [κ_M/(1+β·γ_p)]·(rer_t + π*_t) + ε_P
    # G1[0, PI_L] = γ_p/denom (bakseende ledd), Pi[0, PI] = β/denom (skalert ned)
    _gamma_p = getattr(p, 'gamma_p', 0.0)
    if _gamma_p != 0.0:
        _denom = 1.0 + beta * _gamma_p
        G0[0, MC]      = G0[0, MC]      / _denom   # -kP → -kP/denom
        G0[0, RER]     = G0[0, RER]     / _denom   # -κ_M → -κ_M/denom
        G0[0, PI_STAR] = G0[0, PI_STAR] / _denom   # -κ_M → -κ_M/denom
        G1[0, PI_L]    = _gamma_p / _denom          # ny: bakseende inflasjonsledd
        Pi[0, PI]      = beta / _denom              # β → β/denom

    return G0, G1, Psi, Pi


def build_matrices_v4(p=None, theta_H: float = 0.05):
    """
    NEMO Fase II v4 — RE-korrekt (A9+A9b, PE-godkjent 2026-05-22).

    Implementerer fremoverskuende RE via 7 hjelpetilstander (NZ: 49→56).
    n_unstable=7 = rank(Pi)=7 → BK oppfylt → Schur-projeksjon → stabil løsning.

    Nøkkelendringer fra v3:
      A9:  7 hjelpetilstander PI_E..RER_E for E_t[X_{t+1}] i strukturelle likninger.
           Konsistenslikninger: G0[k,X]=1, G1[k,X_E]=1, Pi[k,X]=1.
      A9b: psi_UIP=0.02 i UIP-likning — bryter enhetsroten (λ=1.0→1.02).
           Tolkes som valutarisikopremie/ufullkommen kapitalbevegelighet (C3-kanal).

    Produksjonsklar — brukes i estimering fra kj14.

    Referanse: K&M (2019), Sims (2002) "Solving Linear Rational Expectations Models"

    Parametere
    ----------
    p        : Parameters-klasse
    theta_H  : Skaleringsfaktor for boligpreferansesjokk (default 0.05)

    Returnerer
    ----------
    G0, G1, Psi, Pi : (NZ_V4×NZ_V4), (NZ_V4×NZ_V4), (NZ_V4×NE), (NZ_V4×NZ_V4)
    """
    if p is None:
        p = Parameters

    # Hent v3-matriser (49×49) og utvid til 56×56
    G0_49, G1_49, Psi_49, Pi_49 = build_matrices_v3(p, theta_H)

    G0  = np.zeros((NZ_V4, NZ_V4))
    G1  = np.zeros((NZ_V4, NZ_V4))
    Psi = np.zeros((NZ_V4, NE))
    Pi  = np.zeros((NZ_V4, NZ_V4))

    # Kopier v3-matriser inn i øvre venstre blokk
    # NB: Pi_49 kopieres IKKE — alle Pi[eq,X]=c-ledd erstattes av G0[eq,X_E]=-c
    G0[:NZ, :NZ] = G0_49
    G1[:NZ, :NZ] = G1_49
    Psi[:NZ, :]  = Psi_49

    # Avledede parametere (gjenberegnes for konsistens med p)
    beta      = p.beta
    delta     = p.delta
    h_c       = p.h_c
    m_H       = p.m_H
    phi_I1    = p.phi_I1
    phi_I2    = p.phi_I2
    a2_W      = 1.0 / (1.0 + h_c)
    a3_W      = (1.0 - h_c) / (p.sigma * (1.0 + h_c))
    b_sa      = getattr(p, 'b_sa',      0.6393)
    lambda_sa = getattr(p, 'lambda_sa', 0.9495)
    w_fwd     = 1.0 - b_sa * lambda_sa

    # ── Modifiser strukturelle likninger ──────────────────────────────────────
    # Konvensjon: Pi_49[eq, X] = c  ↔  +c·E_t[X_{t+1}] på RHS
    # Flytt til LHS: G0[eq, X_E] = −c  (X_E ≡ E_t[X_{t+1}])

    # Ligning 0 (NK Phillips): β·E_t[π_{t+1}]
    G0[0, PI_E]  = -beta

    # Ligning 1 (Euler sparere): a2_W·E_t[c_W_{t+1}] − a3_W·E_t[π_{t+1}]
    G0[1, C_W_E] = -a2_W
    G0[1, PI_E] +=  a3_W       # += fordi PI_E opptrer i flere likninger

    # Ligning 2 (Euler låntakere): (m_H/β)·E_t[q_H_{t+1}]
    G0[2, Q_H_E] = -m_H / beta

    # Ligning 4 (Lønnsinflasjon): β·E_t[π_W_{t+1}]
    G0[4, PIW_E] = -beta

    # Ligning 6 (Boligpris v3): w_fwd·E_t[q_H_{t+1}] − E_t[π_{t+1}]
    G0[6, Q_H_E] += -w_fwd     # += fordi Q_H_E opptrer i to likninger
    G0[6, PI_E]  +=  1.0

    # Ligning 12 (Investering, CEE): β/(1+β)·E_t[inv_{t+1}]
    G0[12, INV_E] = -(beta / (1.0 + beta))

    # Ligning 14 (Tobin's Q): (1−δ)·E_t[q_K_{t+1}] − E_t[π_{t+1}]
    G0[14, Q_K_E]  = -(1.0 - delta)
    G0[14, PI_E]  +=  1.0

    # Ligning 15 (UIP): AR(1)-glattet forventningsledd
    # Fase 1B: RER_E-koeff skaleres med (1-rho_s) fordi v3 setter Pi[15,RER]=_w.
    # psi_UIP=0.02 beholdes som sikkerhetsventil mot enhetsrot ved rho_s→0.
    psi_UIP = 0.02
    _rho_s = getattr(p, 'rho_s', 0.0)
    G0[15, RER_E] = -(1.0 - _rho_s)
    G0[15, RER]  += psi_UIP

    # ── Konsistenslikninger (rader 49–55): X_t = X_E_{t-1} + η_{X,t} ─────────
    # Sims (2002): G0[k,X]=1, G1[k,X_E]=1, Pi[k,X]=1
    # Tolkning: X_t er lik forrige periodes forventning + forventningsfeil
    for (k, X_orig, X_aux) in [
        (PI_E,  PI,  PI_E),
        (C_W_E, C_W, C_W_E),
        (Q_H_E, Q_H, Q_H_E),
        (PIW_E, PIW, PIW_E),
        (INV_E, INV, INV_E),
        (Q_K_E, Q_K, Q_K_E),
        (RER_E, RER, RER_E),
    ]:
        G0[k, X_orig] = 1.0
        G1[k, X_aux]  = 1.0
        Pi[k, X_orig] = 1.0

    return G0, G1, Psi, Pi


def build_matrices_pi4chain(p=None, theta_H: float = 0.05):
    """
    NEMO Alt B — fremoverskuende Taylor med 4-periodes inflasjonsforventningskjede.

    Taylor-regelen reagerer på λ·π_t + (1-λ)·E_t[π_{t+4}] (hybrid, K&M §2.13).
    NZ: 49→53. Fire nye tilstander (Sims 2002 konsistenslikninger):
      PI_E1_t = E_t[π_{t+1}],  PI_E2_t = E_t[π_{t+2}]
      PI_E3_t = E_t[π_{t+3}],  PI_E4_t = E_t[π_{t+4}]  ← Taylor

    Kjede (Sims 2002, η_t = z_t - E_{t-1}[z_t]):
      PI_E1: π_t    = PI_E1_{t-1} + η_{π,t}
      PI_E2: PI_E1_t = PI_E2_{t-1} + η_{PI_E1,t}
      PI_E3: PI_E2_t = PI_E3_{t-1} + η_{PI_E2,t}
      PI_E4: PI_E3_t = PI_E4_{t-1} + η_{PI_E3,t}

    Stabilitet: MSV-løsning max|eig(T)| = 0.998 ✓ (alle lambda-verdier).
    BK-rang(Pi) = 10; bruker direkte MSV (som v3).

    Parametere
    ----------
    p        : Parameters-klasse (bruker kalibrerte verdier hvis None)
    theta_H  : Boligpris-forventningsparameter (videresendt til build_matrices_v3)

    Returnerer
    ----------
    G0, G1, Psi, Pi : (NZ_PI4×NZ_PI4) matriser
    """
    from nemo.model.parameters import Parameters as _DefaultP
    if p is None:
        p = _DefaultP

    G0_49, G1_49, Psi_49, Pi_49 = build_matrices_v3(p, theta_H)

    G0  = np.zeros((NZ_PI4, NZ_PI4))
    G1  = np.zeros((NZ_PI4, NZ_PI4))
    Psi = np.zeros((NZ_PI4, NE))
    Pi  = np.zeros((NZ_PI4, NZ_PI4))

    G0[:NZ, :NZ] = G0_49
    G1[:NZ, :NZ] = G1_49
    Psi[:NZ, :]  = Psi_49
    Pi[:NZ, :NZ] = Pi_49

    psi_R     = p.psi_R
    psi_P1    = p.psi_P1
    lambda_pi4 = getattr(p, 'lambda_pi4', 0.0)  # hybrid-vekt: 0=ren E_t[π_{t+4}], 1=samtid

    # Taylor-regel: hybrid λ·π_t + (1-λ)·E_t[π_{t+4}]  (K&M §2.13, A4b)
    G0[20, PI]    = -(1.0 - psi_R) * psi_P1 * lambda_pi4
    G0[20, PI_E4] = -(1.0 - psi_R) * psi_P1 * (1.0 - lambda_pi4)

    # ── Konsistenslikninger: PI_E1..PI_E4 ────────────────────────────────────
    # Tolkning: X_t = E_{t-1}[X_t] + η_{X,t}
    #   G0[row, X]=1, G1[row, X_E]=1, Pi[row, X]=1
    for (row, X_now, X_lag) in [
        (PI_E1, PI,    PI_E1),   # π_t    = PI_E1_{t-1} + η_{π,t}
        (PI_E2, PI_E1, PI_E2),   # PI_E1_t = PI_E2_{t-1} + η_{PI_E1,t}
        (PI_E3, PI_E2, PI_E3),   # PI_E2_t = PI_E3_{t-1} + η_{PI_E2,t}
        (PI_E4, PI_E3, PI_E4),   # PI_E3_t = PI_E4_{t-1} + η_{PI_E3,t}
    ]:
        G0[row, X_now] = 1.0
        G1[row, X_lag] = 1.0
        Pi[row, X_now] = 1.0

    return G0, G1, Psi, Pi


def build_matrices_v3_forward(p=None, theta_H: float = 0.05,
                               lambda_pi4: float | None = None,
                               n_iter: int = 30, tol: float = 1e-8):
    """
    NEMO v3 med modell-konsistent fremoverskuende Taylor-regel.

    Taylor-regelen er hybrid: λ·π_t + (1-λ)·E_t[π_{t+4}]
    der E_t[π_{t+4}] = e_PI @ T^4 @ z_t beregnes iterativt (fixed-point).

    Fordel over build_matrices_pi4chain:
      - NZ=50 (Alt. A2 2026-06-02: +I_R_LL for AR(2) Taylor)
      - Pi-matrise fra v3 uendret (ingen nye jump-variabler)
      - BK kansellerer IKKE E_i-sjokket (R[I_R, E_i] ≈ 0.98)
      - Stabilitet fra v3 bevares

    Parametere
    ----------
    p          : Parameters-instans (bruker defaults hvis None)
    theta_H    : Boligpris-forventningsparameter (videresendt til v3)
    lambda_pi4 : Hybrid-vekt (0=rent fremoverskuende, 1=v3 samtid).
                 Henter p.lambda_pi4 hvis None, ellers default 0.5.
    n_iter     : Maks iterasjoner for fixed-point konvergens
    tol        : Konvergenstoleranse (||T_new - T_prev||_max)

    Returnerer
    ----------
    G0, G1, Psi, Pi : (NZ×NZ) matriser — samme format som build_matrices_v3
    """
    from nemo.solver.blanchard_kahn import solve as _solve

    if p is None:
        from nemo.model.parameters import Parameters as _DefaultP
        p = _DefaultP

    lam = lambda_pi4
    if lam is None:
        lam = float(getattr(p, 'lambda_pi4', 0.5))

    # Startpunkt: standard v3 (backward-looking)
    G0, G1, Psi, Pi = build_matrices_v3(p, theta_H)

    # Koeffisienter for rad 20 (Taylor-regel) fra v3
    psi_R  = p.psi_R
    psi_R2 = p.psi_R2
    psi_P1 = p.psi_P1
    _scale = 1.0 - psi_R - psi_R2   # langsiktig nøytralitetsbetingelse

    # Basisrad 20 fra v3 (inneholder samtid PI-term = -_scale*psi_P1)
    G0_row20_base = G0[20, :].copy()

    # Løs v3 for startverdi av T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T_prev, _, d = _solve(G0, G1, Psi, Pi, verbose=False)
    if not d.get("stable", False):
        return G0, G1, Psi, Pi   # fallback til v3 hvis ustabilt

    # Seleksjonsvektor for PI (rad 0); NZ=50 etter Alt. A2
    e_PI = np.zeros(NZ)
    e_PI[PI] = 1.0

    for _ in range(n_iter):
        # E_t[π_{t+4}] = e_PI @ T^4 @ z_t
        T4_PI = e_PI @ np.linalg.matrix_power(T_prev, 4)   # (NZ,)

        # Oppdater rad 20: ta utgangspunkt i basisraden (unngå akkumulering)
        G0[20, :] = G0_row20_base.copy()
        # Fjern v3-bidrag fra samtid PI og erstatt med hybrid
        G0[20, PI] = -_scale * psi_P1 * lam
        # Legg til fremoverskuende komponent som lineærkombinasjon av alle tilstander
        G0[20, :] -= _scale * psi_P1 * (1.0 - lam) * T4_PI

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T_new, _, d_new = _solve(G0, G1, Psi, Pi, verbose=False)
        if not d_new.get("stable", False):
            G0[20, :] = G0_row20_base.copy()   # reverter til v3
            return G0, G1, Psi, Pi

        if np.max(np.abs(T_new - T_prev)) < tol:
            break
        T_prev = T_new

    return G0, G1, Psi, Pi


def build_matrices_v3_plt(p=None, theta_H: float = 0.05,
                           lambda_pi4: float | None = None,
                           n_iter: int = 30, tol: float = 1e-8):
    """
    NEMO v3 med PLT-kanal (prisnivåmål, Fase 2 2026-06-02).

    Utvider build_matrices_v3_forward (NZ=50) med:
      - P_STAR_GAP (index 50): akkumulert prisnivå-gap  p_gap_t = p_gap_{t-1} + π_t
      - Taylor-regel: reagerer på psi_PL·p_gap (gir mean-reversion etter sjokk)

    NZ_PLT = 51. Exitstrategi: psi_PL=0.0 → eksakt v3_forward-atferd.
    Ref: Woodford (2003) — prisnivåmål i NK-modeller. PE-godkjent 2026-06-02.

    Parametere
    ----------
    p          : Parameters-instans; psi_PL leses via getattr(p, 'psi_PL', 0.0)
    theta_H    : Boligpris-forventningsparameter (videresendt til v3_forward)
    lambda_pi4 : Hybrid-vekt for fremoverskuende Taylor (videresendt til v3_forward)
    n_iter     : Maks iterasjoner for fixed-point (v3_forward)
    tol        : Konvergenstoleranse (v3_forward)

    Returnerer
    ----------
    G0, G1, Psi, Pi : (NZ_PLT×NZ_PLT), (NZ_PLT×NZ_PLT), (NZ_PLT×NE), (NZ_PLT×NZ_PLT)
    """
    if p is None:
        from nemo.model.parameters import Parameters as _DefaultP
        p = _DefaultP

    # Hent (NZ=50)×(NZ=50) matriser fra v3_forward
    G0_50, G1_50, Psi_50, Pi_50 = build_matrices_v3_forward(
        p, theta_H=theta_H, lambda_pi4=lambda_pi4, n_iter=n_iter, tol=tol
    )

    # Utvid til (NZ_PLT=51)×(NZ_PLT=51)
    G0  = np.zeros((NZ_PLT, NZ_PLT))
    G1  = np.zeros((NZ_PLT, NZ_PLT))
    Psi = np.zeros((NZ_PLT, NE))
    Pi  = np.zeros((NZ_PLT, NZ_PLT))

    G0[:NZ, :NZ] = G0_50
    G1[:NZ, :NZ] = G1_50
    Psi[:NZ, :]  = Psi_50
    Pi[:NZ, :NZ] = Pi_50

    # P_STAR_GAP-likning (rad 50): p_gap_t = p_gap_{t-1} + π_t
    # G0: p_gap_t − π_t = p_gap_{t-1}
    G0[P_STAR_GAP, P_STAR_GAP] =  1.0
    G0[P_STAR_GAP, PI]         = -1.0
    G1[P_STAR_GAP, P_STAR_GAP] =  1.0

    # Legg PLT-ledd til Taylor-regel (rad 20)
    psi_R  = p.psi_R
    psi_R2 = p.psi_R2
    psi_PL = float(getattr(p, 'psi_PL', 0.0))
    _scale = 1.0 - psi_R - psi_R2   # langsiktig nøytralitetsbetingelse
    G0[I_R, P_STAR_GAP] = -_scale * psi_PL

    return G0, G1, Psi, Pi


def build_matrices_altB(p=None, theta_H: float = 0.05):
    """
    NEMO Alt B (PE-godkjent 2026-05-29) — boliginvesteringskanal implementert.

    Bygger på build_matrices_v3 og legger til:
      - Separat boliginvesteringstilstand INV_H (index 49) med CEE Euler-ligning
      - Lagg INV_H_L (index 50) — NZ_ALTB=51
      - Boligakkumulering kobles til INV_H (ikke direkte Q_H)
      - Ressursbetingelsen skiller kapital (IY*INV) og bolig (IHY*INV_H)

    Motivasjon (kj26 diagnose):
      Med φ_I1=12.54 (K&M) gir vår forenklede modell BNP q4=0.33× NB (B5-grense 0.8×).
      Årsak: phi_H1=60.73, phi_H2=199.65 var kalibrert i parameters.py men aldri brukt
      i equations.py. Boliginvestering (IHY=0.10 av BNP) mangler forward-looking dynamikk.
      Ny Euler-ligning gir: renteheving → Q_H faller → INV_H reagerer gradvis
      → ekstra 0.1–0.3% BNP-bidrag ved q4 — nødvendig for å passere B5 med K&M φ_I1.

    Exit-mulighet:
      build_matrices_v3 er UENDRET. For å rulle tilbake: bruk v3 i log_posterior.

    Parametere
    ----------
    p        : Parameters-klasse (eller underklasse med oppdaterte estimater)
    theta_H  : Skaleringsfaktor for boligpreferansesjokket (default 0.05)

    Returnerer
    ----------
    G0, G1, Psi, Pi : (NZ_ALTB×NZ_ALTB) = (51×51) matriser
    """
    if p is None:
        p = Parameters

    # ── Hent v3-matriser (NZ=49) som fundament ───────────────────────────────
    G0_v3, G1_v3, Psi_v3, Pi_v3 = build_matrices_v3(p, theta_H=theta_H)

    # ── Bygg nye NZ_ALTB=51 matriser, kopier v3 inn ───────────────────────────
    G0  = np.zeros((NZ_ALTB, NZ_ALTB))
    G1  = np.zeros((NZ_ALTB, NZ_ALTB))
    Psi = np.zeros((NZ_ALTB, NE))
    Pi  = np.zeros((NZ_ALTB, NZ_ALTB))

    G0[:NZ, :NZ]  = G0_v3
    G1[:NZ, :NZ]  = G1_v3
    Psi[:NZ, :]   = Psi_v3
    Pi[:NZ, :NZ]  = Pi_v3

    # ── Parametere brukt i nye ligninger ─────────────────────────────────────
    _beta    = p.beta
    _delta_H = p.delta_H
    _phi_H1  = p.phi_H1   # 60.73 (K&M Tabell 8)
    IY       = p.IY
    IHY      = p.IHY

    # ── Ligning INV_H: boliginvesterings-Euler (CEE 2005, samme form som INV) ─
    # FOC: q_H_t = φ_H1·(1+β)·inv_H_t − φ_H1·inv_H_{t-1} − β·φ_H1·E[inv_H_{t+1}]
    # → inv_H_t = [1/(φ_H1·(1+β))]·q_H_t
    #           + [1/(1+β)]·inv_H_{t-1}
    #           + [β/(1+β)]·E_t[inv_H_{t+1}]
    #
    # Med phi_H1=60.73, beta=0.99:
    #   q_H-koeff  = 1/(60.73·1.99) ≈ 0.0083  (tregere enn kapital: 1/(12.54·1.99) ≈ 0.040)
    #   lag-vekt   = 1/1.99 ≈ 0.503
    #   lead-vekt  = 0.99/1.99 ≈ 0.497
    G0[INV_H, INV_H]   =  1.0
    G0[INV_H, Q_H]     = -1.0 / (_phi_H1 * (1.0 + _beta))
    G0[INV_H, INV_H_L] = -(1.0 / (1.0 + _beta))
    Pi[INV_H, INV_H]   =  _beta / (1.0 + _beta)

    # ── Ligning INV_H_L: lagg av boliginvestering ────────────────────────────
    # INV_H_L_t = INV_H_{t-1}  →  G0[INV_H_L, INV_H_L] = 1, G1[INV_H_L, INV_H] = 1
    G0[INV_H_L, INV_H_L] =  1.0
    G1[INV_H_L, INV_H]   =  1.0

    # ── Oppdater boligakkumulering (ligning 7, 8): INV_H i stedet for Q_H ────
    # Gammelt: h_W_t = (1-δ_H)·h_W_{t-1} + δ_H·q_H_t  (forenklet: invest prop til pris)
    # Nytt:    h_W_t = (1-δ_H)·h_W_{t-1} + δ_H·inv_H_t  (full Euler-driven akkumulering)
    G0[H_W,  Q_H]   = 0.0          # fjern Q_H fra ligning 7
    G0[H_W,  INV_H] = -_delta_H    # koble til INV_H i stedet

    G0[H_NW, Q_H]   = 0.0          # fjern Q_H fra ligning 8
    G0[H_NW, INV_H] = -_delta_H    # begge husholdningstyper deler INV_H-dynamikk

    # ── Oppdater ressursbetingelse (ligning 9): skill INV og INV_H ───────────
    # Gammelt: G0[9, INV] = -(IY+IHY)  (boliginvestering klumpet inn i INV)
    # Nytt:    G0[9, INV] = -IY        (kapitalinvestering)
    #          G0[9, INV_H] = -IHY     (boliginvestering separat — gir B5-kanalen)
    G0[Y, INV]   = -IY       # kapital (0.20 av BNP)
    G0[Y, INV_H] = -IHY      # bolig (0.10 av BNP)

    # ── Oppdater eksportligning (ligning 17) tilsvarende ─────────────────────
    # Ligning 17 har identisk ressursbetingelse
    G0[X, INV]   = -IY
    G0[X, INV_H] = -IHY

    return G0, G1, Psi, Pi


def build_matrices_georg(p=None, theta_H: float = 0.05,
                          use_georg: bool = True,
                          n_iter: int = 60, tol: float = 1e-9):
    """
    NEMO med GEORG-politikkregel (Almlid, Haltia & Robstad 2025, Staff Memo 15/2025).

    GEORG ("Ganske Enkel Optimal ReGel") er NBs enkle regel som via IRF-matching
    reproduserer den tapsfunksjonsbaserte optimale politikken i NEMO. Denne
    byggeren erstatter mimicking rule (rad 20) med GEORG og brukes som
    *læringssteg*: ved å sammenligne pengepolitikk-IRF for GEORG mot (a) NB Memo
    3/2024 Figur 1 og (b) vår mimicking-rule-IRF, isolerer vi om NB-avviket er
    drevet av politikkregelen eller av transmisjonen.

    Regelen (lign. 1+3, alle variabler som gap):
        i_R_t = ω_r·i_R_{t-1} + (1-ω_r)·X_t + Z_t
        X_t   = E_t[ ω_π·π̂_{t+1} + ω_y·ŷ_{t+1} + ω_ϕ·ϕ̂_{t+1}
                     + ω_S·Ŝ_{t+1} + ω_rf·r̂^f_{t+1} ] + ω_μ·μ̂_t
        Z_{t+1} = λ_Z·Z_t + ε   (lign. 2)

    Indikatorene (Staff Memo 15/2025 §2):
        π̂  : 1-kv. frem-anslag av 4-kv. KPI-JAE-vekst  (PI + lagg + forventning)
        ŷ  : outputgap                                  (Y, forventning via T)
        ϕ̂  : 4-kv. enhetslønnskostnad-vekst-gap          (πW-sum − Δ(4)a)
        Ŝ  : 8-kv. nominell valutakursvekst-gap          (Σ_{j=0}^{7} s_{t-j})
        r̂^f: utenlandsk rente-gap                        (I_STAR, forventning via T)
        μ̂  : pengemarkedspremie-gap (samtid)             (EPS_PREM)

    Forventnings-maskineri (gjenbruk fra build_matrices_v3_forward):
        E_t[X_{t+1}] = (e_X @ T) · z_t beregnes via fixed-point på T-matrisen.

    Annualisering (egen tilpasning, dokumentert):
        GEORG-koeffisientene (Tabell 4) er annualiserte; modellens i_R/π er
        kvartalsrater. Regelen skrives for annualisert rente og konverteres til
        kvartal (÷4). 4-kv./8-kv. vekst-summer er allerede annuelle (sum av
        kvartalsrater). Rate-type indikatorer (r̂^f, μ̂) annualiseres ×4 og
        kanselleres mot ÷4 → netto koeffisient ω. Nivå-gap (ŷ) og vekst-summer
        (π̂4, ϕ̂4, Ŝ8) skaleres ÷4. Konvensjonen påvirker magnitude, ikke fortegn;
        IRF-nivå normaliseres mot styringsrente-toppen (jf. Spor B5).

    Produktivitet i ϕ̂ (egen tilpasning): ULC-vekst = nominell lønnsvekst (πW)
        minus TFP-vekst (Δa). I gap-form er trend-leddet konstant og faller ut.

    Exitstrategi:
        use_georg=False → returnerer build_matrices_v3_forward (NZ=50) utvidet med
        de 14 GEORG-tilstandene som *dead states* (lagg-identiteter uten
        tilbakekobling til rad 20). Kjernedynamikken er da eksakt v3_forward.

    Parametere
    ----------
    p          : Parameters-instans (bruker defaults hvis None)
    theta_H    : Boligpris-forventningsparameter (videresendt til v3)
    use_georg  : True = GEORG-regel; False = exit til v3_forward (padded)
    n_iter     : Maks iterasjoner for fixed-point
    tol        : Konvergenstoleranse (||T_new − T_prev||_max)

    Returnerer
    ----------
    G0, G1, Psi, Pi : (NZ_GEORG×NZ_GEORG), (·×NZ_GEORG), (·×NE), (·×NZ_GEORG)
    """
    from nemo.solver.blanchard_kahn import solve as _solve

    if p is None:
        p = Parameters

    # ── Bygg utvidede matriser fra v3 (NZ=50) ─────────────────────────────────
    G0_50, G1_50, Psi_50, Pi_50 = build_matrices_v3(p, theta_H=theta_H)

    G0  = np.zeros((NZ_GEORG, NZ_GEORG))
    G1  = np.zeros((NZ_GEORG, NZ_GEORG))
    Psi = np.zeros((NZ_GEORG, NE))
    Pi  = np.zeros((NZ_GEORG, NZ_GEORG))

    G0[:NZ, :NZ] = G0_50
    G1[:NZ, :NZ] = G1_50
    Psi[:NZ, :]  = Psi_50
    Pi[:NZ, :NZ] = Pi_50    # ingen nye jump-variabler — Pi uendret

    # ── Lagg-identiteter for de nye tilstandene: X_t = src_{t-1} ──────────────
    # G0[k,k]=1, G1[k,src]=1  ⇒  k_t = src_{t-1}
    for k, src in [
        (GEORG_PI_L2,  PI_L),         # π_{t-2}  (PI_L=pi_lag gir π_{t-1})
        (GEORG_PIW_L1, PIW),          # πW_{t-1}
        (GEORG_PIW_L2, GEORG_PIW_L1), # πW_{t-2}
        (GEORG_A_L1,   A),            # a_{t-1}
        (GEORG_A_L2,   GEORG_A_L1),   # a_{t-2}
        (GEORG_A_L3,   GEORG_A_L2),   # a_{t-3}
        (GEORG_S_L1,   S),            # s_{t-1}
        (GEORG_S_L2,   GEORG_S_L1),
        (GEORG_S_L3,   GEORG_S_L2),
        (GEORG_S_L4,   GEORG_S_L3),
        (GEORG_S_L5,   GEORG_S_L4),
        (GEORG_S_L6,   GEORG_S_L5),
        (GEORG_S_L7,   GEORG_S_L6),   # s_{t-7}
    ]:
        G0[k, k]   = 1.0
        G1[k, src] = 1.0

    # AR(1) pengepolitikksjokk:  Z_t = λ_Z·Z_{t-1} + ε_i
    lambda_Z = float(getattr(p, 'georg_lambda_Z', 0.75))
    G0[GEORG_Z, GEORG_Z] = 1.0
    G1[GEORG_Z, GEORG_Z] = lambda_Z
    Psi[GEORG_Z, E_i]    = 1.0

    # ── Exit: returner v3_forward (padded med dead states) ────────────────────
    if not use_georg:
        G0f, G1f, Psif, Pif = build_matrices_v3_forward(p, theta_H=theta_H)
        # Overskriv kjerneblokk (rad/kol 0..49) med v3_forward; behold lagg-states.
        # I.i.d. policy-sjokket beholdes i v3_forward-rad 20 (Z er dead her).
        G0[:NZ, :NZ] = G0f
        G1[:NZ, :NZ] = G1f
        Psi[:NZ, :]  = Psif
        Pi[:NZ, :NZ] = Pif
        # Nøytraliser AR(1)-Z slik at den ikke introduserer ekstra dynamikk i
        # kjernen (den er uansett frakoblet rad 20 her).
        Psi[GEORG_Z, E_i] = 0.0
        return G0, G1, Psi, Pi

    # ── GEORG-koeffisienter (Tabell 4) ────────────────────────────────────────
    w_r  = float(getattr(p, 'georg_omega_r',   0.74))
    w_pi = float(getattr(p, 'georg_omega_pi',  1.17))
    w_y  = float(getattr(p, 'georg_omega_y',   1.27))
    w_ph = float(getattr(p, 'georg_omega_phi', 1.25))
    w_S  = float(getattr(p, 'georg_omega_S',   0.13))
    w_rf = float(getattr(p, 'georg_omega_rf',  0.25))
    w_mu = float(getattr(p, 'georg_omega_mu', -1.00))

    # ── Statisk del av X_t (uavhengig av T) som rad-vektor over tilstandene ────
    # Konvensjon (annualisering): vekst-summer og nivå-gap ÷4; rate-gap netto ω.
    x_static = np.zeros(NZ_GEORG)
    # π̂4 (samtid + 2 lagg-ledd; forventningsleddet legges til i fixed-point): /4
    x_static[PI]          += w_pi / 4.0    # π_t
    x_static[PI_L]        += w_pi / 4.0    # π_{t-1}
    x_static[GEORG_PI_L2] += w_pi / 4.0    # π_{t-2}
    # ϕ̂4 (samtid + lagg): πW-ledd + a_{t-3}-ledd: /4
    x_static[PIW]          += w_ph / 4.0   # πW_t
    x_static[GEORG_PIW_L1] += w_ph / 4.0   # πW_{t-1}
    x_static[GEORG_PIW_L2] += w_ph / 4.0   # πW_{t-2}
    x_static[GEORG_A_L3]   += w_ph / 4.0   # + a_{t-3}  (−(a_{t+1}−a_{t-3}))
    # Ŝ8 (samtid + 6 lagg; forventningsleddet i fixed-point): /4
    x_static[S]          += w_S / 4.0      # s_t
    for k in (GEORG_S_L1, GEORG_S_L2, GEORG_S_L3,
              GEORG_S_L4, GEORG_S_L5, GEORG_S_L6):
        x_static[k] += w_S / 4.0
    # μ̂_t (samtid, rate-type → netto ω): pengemarkedspremie-gap
    x_static[EPS_PREM] += w_mu

    # ── Basisrad 20 (uten forventningsledd) ───────────────────────────────────
    row20_base = np.zeros(NZ_GEORG)
    row20_base[I_R]     =  1.0
    row20_base[I_R_L]   = -w_r            # ω_r·i_R_{t-1}
    row20_base[GEORG_Z] = -1.0            # + Z_t
    row20_base += -(1.0 - w_r) * x_static
    Psi[20, :]  = 0.0                     # sjokket går nå via AR(1)-Z (rad GEORG_Z)

    # ── Fixed-point på T for forventningsleddene ──────────────────────────────
    # E_t[X_{t+1}]-bidrag = rad X i T (e_X @ T = T[X, :]).
    G0[20, :] = row20_base.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T_prev, _, d = _solve(G0, G1, Psi, Pi, verbose=False)
    if not d.get("stable", False):
        return G0, G1, Psi, Pi    # fallback: rad 20 uten forventningsledd

    for _ in range(n_iter):
        # Forventnings-rad: ω_π·E[π_{t+1}]/4 + ω_y·E[y_{t+1}]/4
        #   + ω_ϕ·(E[πW_{t+1}] − E[a_{t+1}])/4 + ω_S·E[s_{t+1}]/4 + ω_rf·E[i*_{t+1}]
        x_fwd = (
            (w_pi / 4.0) * T_prev[PI, :]
            + (w_y / 4.0) * T_prev[Y, :]
            + (w_ph / 4.0) * (T_prev[PIW, :] - T_prev[A, :])
            + (w_S / 4.0) * T_prev[S, :]
            + w_rf * T_prev[I_STAR, :]
        )
        G0[20, :] = row20_base.copy()
        G0[20, :] += -(1.0 - w_r) * x_fwd

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T_new, _, d_new = _solve(G0, G1, Psi, Pi, verbose=False)
        if not d_new.get("stable", False):
            G0[20, :] = row20_base.copy()   # reverter til stabil basisrad
            return G0, G1, Psi, Pi

        if np.max(np.abs(T_new - T_prev)) < tol:
            T_prev = T_new
            break
        T_prev = T_new

    return G0, G1, Psi, Pi


def build_matrices_rpendo(p=None, theta_H: float = 0.05,
                           lambda_pi4: float | None = None,
                           n_iter: int = 60, tol: float = 1e-9):
    """
    NEMO v3 med endogen risikopremie i UIP (PE-godkjent 2026-06-04).

    Adresserer det monetære RER-IRF-gapet (transmisjonsdiagnose,
    `docs/oppgaver/transmisjon_rer_diagnose.md`): NB Figur 1 viser et stort
    RER-utslag som henger appresiert, mens v3 gir for lite utslag som overshooter
    til positivt. En persistent risikopremie som reagerer på rentedifferansen
    gir både større impact og tregere hale (forward premium puzzle;
    Mæhlum 2025, Staff Memo 3/2025 «Monetary Policy and the Exchange Rate in
    Norway»).

    Ny tilstand (NZ_RPENDO = 51):
        RP_ENDO_t = ρ_pe·RP_ENDO_{t-1} + κ_pe·(i_D_t − i*_t)
    UIP-likningen (rad 15) utvides med −(1−ρ_s)·RP_ENDO (appresieringspress):
        rer_t = … − (1−ρ_s)·RP_ENDO_t
    κ_pe = `kappa_rp_endo`, ρ_pe = `rho_rp_endo` (parameters.py).

    Bygger på den fremoverskuende Taylor-regelen (samme fixed-point som
    `build_matrices_v3_forward`), nå løst på det utvidede 51-systemet, slik at
    den er sammenliknbar med kj41-referansen. v3/v3_forward er **urørt**.

    Exitstrategi: κ_pe = 0 → RP_ENDO blir frakoblet UIP (dead state) og kjernen
    er eksakt v3_forward.

    Parametere
    ----------
    p          : Parameters-instans (defaults hvis None)
    theta_H    : Boligpris-forventningsparameter (videresendt til v3)
    lambda_pi4 : Hybrid-vekt for fremoverskuende Taylor (som v3_forward)
    n_iter     : Maks iterasjoner for fixed-point
    tol        : Konvergenstoleranse

    Returnerer
    ----------
    G0, G1, Psi, Pi : (NZ_RPENDO×NZ_RPENDO), (·×NE), … matriser
    """
    from nemo.solver.blanchard_kahn import solve as _solve

    if p is None:
        p = Parameters

    lam = lambda_pi4
    if lam is None:
        lam = float(getattr(p, 'lambda_pi4', 0.5))

    kappa_pe = float(getattr(p, 'kappa_rp_endo', 0.0))
    rho_pe   = float(getattr(p, 'rho_rp_endo', 0.90))
    rho_s    = float(getattr(p, 'rho_s', 0.0))
    _w       = 1.0 - rho_s

    # ── Bygg utvidet v3 (NZ=50 → 51) ──────────────────────────────────────────
    G0_50, G1_50, Psi_50, Pi_50 = build_matrices_v3(p, theta_H=theta_H)
    G0  = np.zeros((NZ_RPENDO, NZ_RPENDO))
    G1  = np.zeros((NZ_RPENDO, NZ_RPENDO))
    Psi = np.zeros((NZ_RPENDO, NE))
    Pi  = np.zeros((NZ_RPENDO, NZ_RPENDO))
    G0[:NZ, :NZ] = G0_50
    G1[:NZ, :NZ] = G1_50
    Psi[:NZ, :]  = Psi_50
    Pi[:NZ, :NZ] = Pi_50

    # ── Lov for endogen risikopremie ──────────────────────────────────────────
    # RP_ENDO_t = ρ_pe·RP_ENDO_{t-1} + κ_pe·(i_D_t − i*_t)
    G0[RP_ENDO, RP_ENDO] =  1.0
    G0[RP_ENDO, I_D]     = -kappa_pe
    G0[RP_ENDO, I_STAR]  = +kappa_pe
    G1[RP_ENDO, RP_ENDO] =  rho_pe

    # ── Koble premien inn i UIP (rad 15): rer_t = … − (1−ρ_s)·RP_ENDO ─────────
    # Positiv premie ved renteoppgang → ekstra appresiering (rer ned).
    G0[15, RP_ENDO] = _w

    # ── Fremoverskuende Taylor via fixed-point (mirror v3_forward) ────────────
    psi_R  = p.psi_R
    psi_R2 = p.psi_R2
    psi_P1 = p.psi_P1
    _scale = 1.0 - psi_R - psi_R2

    G0_row20_base = G0[20, :].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T_prev, _, d = _solve(G0, G1, Psi, Pi, verbose=False)
    if not d.get("stable", False):
        return G0, G1, Psi, Pi   # fallback (v3-bakover Taylor på utvidet system)

    e_PI = np.zeros(NZ_RPENDO)
    e_PI[PI] = 1.0

    for _ in range(n_iter):
        T4_PI = e_PI @ np.linalg.matrix_power(T_prev, 4)
        G0[20, :] = G0_row20_base.copy()
        G0[20, PI] = -_scale * psi_P1 * lam
        G0[20, :] -= _scale * psi_P1 * (1.0 - lam) * T4_PI

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T_new, _, d_new = _solve(G0, G1, Psi, Pi, verbose=False)
        if not d_new.get("stable", False):
            G0[20, :] = G0_row20_base.copy()
            return G0, G1, Psi, Pi

        if np.max(np.abs(T_new - T_prev)) < tol:
            break
        T_prev = T_new

    return G0, G1, Psi, Pi
