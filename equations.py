"""
================================================================================
NEMO FASE II — LIKNINGSSYSTEM
Γ₀ z_t = Γ₁ z_{t-1} + Ψ ε_t + Π η_t

Tilstandsvektor (NZ = 48):
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

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from parameters import Parameters

# ── Dimensjoner ───────────────────────────────────────────────────────────────
NZ = 48
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
    'yS','eps_rp','pi_star','i_star','eps_phi_h','eps_prem','eps_I_adj'
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
    kappa_M = 0.03             # importpriskanal (beholdes fra Fase I)

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
    Psi[2, E_C]  =  a2_W                  # delt preferansesjokk

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

    # C1. BNP (varemarkedsklarering)
    G0[9, Y]    =  1.0
    G0[9, C]    = -CY
    G0[9, INV]  = -IY
    G0[9, G]    = -GY
    G0[9, X]    = -XY
    G0[9, M]    =  MY
    # Boliginvestering inngår delvis i INV (forenklet)

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

    # D1. UIP med pengemarkedspremie (utvidet fra Fase I)
    # E[rer_{t+1}] = rer + (i_D - π) - (i* - π*) + ε_rp + ε_prem
    G0[15, RER]       =  1.0
    G0[15, I_D]       =  1.0
    G0[15, PI]        = -1.0
    G0[15, I_STAR]    = -1.0
    G0[15, PI_STAR]   =  1.0
    G0[15, EPS_PREM]  = -1.0   # pengemarkedspremie som UIP-skift
    Pi[15, RER]       =  1.0
    Psi[15, E_rp]     =  1.0
    Psi[15, E_prem]   =  1.0

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
    G0[22, EPS_PHI_H] = -1.0   # LTV-sjokk påvirker spread

    # E4. Utlånsrente, låntakere (NW) — høyere spread
    G0[23, I_L_NW]  =  1.0
    G0[23, I_R]     = -1.0
    G0[23, NB]      = -1.5 * p.phi_c   # høyere spread for låntakere
    G0[23, EPS_PHI_H] = -1.0

    # E5. Gjeld, sparere (ikke-bindende)
    # b_W: finansiell formueakkumulering sparere
    G0[24, B_W]     =  1.0
    G0[24, I_L_W]   = -(1.0 - omega)
    G0[24, Y]       = -(1.0 - omega)

    # E6. Gjeld, låntakere (LTV-bindende)
    # b_NW = m_H · q_H · h_NW / (1 + i_L_NW)
    G0[25, B_NW]    =  1.0
    G0[25, Q_H]     = -m_H
    G0[25, H_NW]    = -m_H
    G0[25, I_L_NW]  =  m_H
    Psi[25, E_phi_h] = 1.0   # LTV-sjokk

    # E7. Bankkapital-akkumulering (netto kapital)
    # nb = (1-δ_b)·nb_{t-1} + spread·lån - kapitalkrav
    G0[26, NB]     =  1.0
    G0[26, I_R]    = -p.phi_o   # spread-inntekt
    G0[26, B_NW]   = -p.phi_o
    G0[26, NB]     +=  p.phi_c  # kapitaldekning-kostnad (allerede +1.0 over)
    # Forenklet: nb ≈ spread·(b_W + b_NW) - φ_c·(nb - γ_b·aktiva)

    # ════════════════════════════════════════════════════════════════════════
    # BLOKK F: OFFENTLIG SEKTOR
    # Forenklet fiskalregel (GPFG utvides i neste iterasjon)
    # ════════════════════════════════════════════════════════════════════════

    # F1. Offentlig konsum (fiskalregel med lagg + AR(1)-sjokk)
    G0[27, G]    =  1.0
    G1[27, PO]   =  gamma_G
    Psi[27, E_G] =  1.0   # offentlig forbrukssjokk

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
    G0[33, I_R_L]=1.0; G1[33, I_R]=1.0    # i_{t} = i_{t-1}
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
    G0[44,I_STAR]=1.0; G1[44,I_STAR]=p.rho_piS;  # utenlandsk rente
    G0[45,EPS_PHI_H]=1.0;G1[45,EPS_PHI_H]=p.rho_phi_h;Psi[45,E_phi_h]=1.0
    G0[46,EPS_PREM]=1.0; G1[46,EPS_PREM]=p.rho_prem; Psi[46,E_prem]=1.0
    G0[47,EPS_I_ADJ]=1.0;G1[47,EPS_I_ADJ]=p.rho_I;  Psi[47,E_I]=1.0

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
