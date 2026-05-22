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
 
import numpy as np

from nemo.model.parameters import Parameters
 
# ── Dimensjoner ───────────────────────────────────────────────────────────────
# Alt. A (2026-05-15): NZ 48→49 — variabel kapitalutnyttelse u_t lagt til
# (K&M 2019 Tabell 8 φ_u=0.2192 var kalibrert men ikke implementert)
NZ = 49
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

# A9 (PE-godkjent 2026-05-22): 7 hjelpetilstander for RE-forventninger (NZ 49→56)
PI_E=49; C_W_E=50; Q_H_E=51; PIW_E=52; INV_E=53; Q_K_E=54; RER_E=55
NZ_V4 = 56

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
 
    # D1. UIP med pengemarkedspremie, gjeldselastisk premie og olje-valuta-kanal
    # E[rer_{t+1}] = rer + (i_D - π) - (i* - π*) + ε_rp + ε_prem + φ_B·b_NW - φ_O·po
    # φ_B·b_NW: gjeldselastisk ankerfeste (K&M §3.4, Tabell 8, PE-godkjent 2026-05-19)
    # φ_O·po: direkte olje→valutakurs-kanal (PE-godkjent 2026-05-20):
    #   høy oljepris → NOK-appresiering (RER ned) — kalibrert fra hist. NOK/olje-korr ~0.7
    #   Hypotese: ε_rp absorberer denne kanalen når den mangler → sigma_rp=0.017 vs K&M 0.006
    phi_O = p.phi_O
    G0[15, RER]       =  1.0
    G0[15, I_D]       =  1.0
    G0[15, PI]        = -1.0
    G0[15, I_STAR]    = -1.0
    G0[15, PI_STAR]   =  1.0
    G0[15, EPS_PREM]  = -1.0   # pengemarkedspremie som UIP-skift
    G0[15, B_NW]      =  phi_B # gjeldselastisk premie — φ_B=0.0016 (K&M Tabell 8)
    G0[15, PO]        =  phi_O # olje-valuta-kanal — høy oljepris styrker NOK (RER ned)
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
 
 
def build_matrices_v3(p=None, theta_H: float = 0.05):
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
    p        : Parameters-klasse (eller underklasse med oppdaterte estimater)
    theta_H  : Skaleringsfaktor for boligpreferansesjokket (default 0.05)
               Brukes til å kontrollere boligsektorens relative volatilitet
               under estimering. Posterior mean ≈ 0.05 i K&M (2019).
 
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
 
    w_back = b_sa * lambda_sa                   # bakseende vekt
    w_fwd  = 1.0 - w_back                       # fremoverskuende vekt
 
    G0[6, :] = 0.0; G1[6, :] = 0.0; Pi[6, :] = 0.0; Psi[6, :] = 0.0
    G0[6, Q_H]    =  1.0
    G0[6, I_D]    =  1.0
    G0[6, PI]     = -1.0
    G1[6, Q_H]    =  w_back       # bakseende: q_H_{t-1} (via lagg-identitet)
    Pi[6, Q_H]    =  w_fwd        # fremoverskuende: E[q_H_{t+1}]
    Pi[6, PI]     = -1.0          # E[π_{t+1}]
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
    # Ligning 20: i_R = ψ_R·i_{t-1} + (1-ψ_R)·[ψ_P1·π_t + ψ_Y·y + ψ_S·rer] + ε_i
    # Spor A4b (2026-05-15):
    #   1) samtid π (G0[20, PI]) i stedet for π_{t-1}
    #   2) G0[20, I_R_L] = -psi_R i stedet for G1[20, I_R_L] = psi_R
    #      (G1-bruken ga 2-periodes oscillasjon)
    psi_R  = p.psi_R
    psi_P1 = p.psi_P1
    psi_Y  = p.psi_Y
    psi_S  = p.psi_S
    psi_W  = p.psi_W

    G0[20, :] = 0.0; G1[20, :] = 0.0; Psi[20, :] = 0.0
    G0[20, I_R]   =  1.0
    G0[20, Y]     = -(1.0 - psi_R) * psi_Y
    G0[20, RER]   = -(1.0 - psi_R) * psi_S
    G0[20, PI]    = -(1.0 - psi_R) * psi_P1   # samtid inflasjon
    G0[20, PIW]   = -(1.0 - psi_R) * psi_W    # A7 (PE-godkjent 2026-05-21): lønnsinflasjon, K&M §2.13
    G0[20, I_R_L] = -psi_R                     # 1-periodes lagg via lagg-tilstand
    Psi[20, E_i]  =  1.0

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

    # Ligning 15 (UIP): E_t[rer_{t+1}]
    # A9b (PE-godkjent 2026-05-22): psi_UIP=0.02 bryter enhetsroten fra ren UIP.
    # Ren UIP (koeff=1.0) gir companion eigenverdi λ=1.0 (enhetsrot) → BK ikke oppfylt.
    # Med psi_UIP: (1+ψ)·rer_t = E_t[rer_{t+1}] + ... → eigenverdi 1+ψ≈1.02 > 1.001.
    # Tolkning: 2% valutarisikopremie/ufullkommen kapitalbevegelighet (C3-kanal).
    psi_UIP = 0.02
    G0[15, RER_E] = -1.0
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
