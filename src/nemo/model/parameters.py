"""
================================================================================
NEMO FASE II — PARAMETERE
Kalibrert direkte fra Kravik og Mimir (2019), Tabell 8, 9 og 10
================================================================================
Endringer fra Fase I:
  phi_L    : 1.50 → 3.00 → 1.50  (A_phi_L-rettelse, PE-godkjent 2026-05-21)
  delta    : 0.025 → 0.0108 (δ_K, Tabell 8 — 131% avvik korrigert)
  delta_H  : NY → 0.0228   (δ_H, boligkapitaldepresiering)
  eta_M    : 1.50 → 0.50   (µ, Tabell 8 — faktor 3)
  eta_X    : 1.50 → 0.50   (µ*, Tabell 8)
  h        : 0.74 → 0.9384 (b_c, Tabell 8 — habit konsum)
  h_L      : NY → 0.5862   (b_l, habit fritid)
  h_H      : NY → 0.9867   (b_h, habit bolig)
  rho_rp   : 0.50 → 0.737  (λ_B, Tabell 9)
  alpha_K  : 0.33 → 0.256  (α, Tabell 8)
  rho_A    : 0.85 → 0.804  (λ_zL, Tabell 9)
  rho_C    : 0.75 → 0.7248 (λ_u, Tabell 9)
  rho_O    : 0.85 → 0.8736 (λ_PO*, Tabell 9)
  rho_Ys   : 0.85 → 0.7825 (λ_U*, Tabell 9)

  NYE SEKTORER (Fase II):
  - Kapitalakkumulering med investeringsjusteringskostnader
  - Boligsektor (Gelain et al. 2018)
  - Banksektor (Gerali et al. 2010): sparere / låntakere / bank
  - Full GPFG-fiskalregel (NEMO seksjon 2.9.4)
  - Fremoverskuende mimicking rule (NEMO seksjon 2.13)

Referanse: Kravik og Mimir (2019), Tabell 8–10.
================================================================================
"""


class Parameters:
    """
    Alle parametere for NEMO Fase II.
    Kalibrert fra Kravik og Mimir (2019), Tabell 8, 9 og 10.
    Estimerte parametere merket med (EST) — posteriormoment brukt som startverdi.
    Kalibrerte parametere merket med (CAL).
    """

    # ═══════════════════════════════════════════════════════════════════════
    # BLOKK A: HUSHOLDNINGER
    # NEMO seksjon 2.2 | Kravik & Mimir Tabell 8
    # ═══════════════════════════════════════════════════════════════════════

    # Diskonteringsfaktor
    beta     = 0.99       # (CAL) β — Tabell 8: 0.99

    # Habit persistence — NEMO har tre separate (konsum, bolig, fritid)
    h_c      = 0.9384     # (EST) b_c — habit konsum, Tabell 8
    h_H      = 0.9867     # (EST) b_h — habit boligservices, Tabell 8
    h_l      = 0.5862     # (EST) b_l — habit fritid, Tabell 8
    h_d      = 0.4813     # (EST) b_d — habit innskudd, Tabell 8

    # Invers Frisch-elastisitet
    phi_L    = 1.50       # (CAL) ζ — Tabell 8: 1.5 (A_phi_L-rettelse, PE-godkjent 2026-05-21)

    # Invers IES (log-nytte: σ=1)
    sigma    = 1.00       # (CAL) σ

    # Innskuddsnytte (deposit preferences)
    z_d      = 0.1889     # (EST) z_d — Tabell 8

    # Husholdningsdeling: andel låntakere (NW) og sparere
    # NEMO: NW = "Non-optimizers" (låntakere med belåningsgrense)
    omega_NW = 0.35       # (CAL) Andel låntakere — typisk 30-40% i norsk data

    # Loan-to-Value (LTV) for husholdninger
    m_H      = 0.80       # (CAL) m_H — maks LTV boliglån (norsk regulering)
    m_e      = 0.35       # (CAL) m_e — maks LTV for entreprenørers næringseiendom

    # Bakseende forventningsdannelse (boligpriser, Gelain et al. 2018)
    b_sa     = 0.6393     # (EST) b_sa — andel bakseende agenter, Tabell 8
    lambda_sa = 0.9495    # (EST) λ_sa — grad av bakseende, Tabell 8

    # ═══════════════════════════════════════════════════════════════════════
    # BLOKK B: PRODUKSJON OG PRISSETTING
    # NEMO seksjon 2.3–2.4 | Tabell 8
    # ═══════════════════════════════════════════════════════════════════════

    # Kapitalandel i mellomvareproduksjon
    alpha_K  = 0.256      # (CAL) α — Tabell 8: 0.256 (Fase I: 0.33)

    # Kapitaldepresiering
    delta    = 0.0108     # (CAL) δ_K — Tabell 8: 0.0108 (≈4.3% p.a.)
    delta_H  = 0.0228     # (CAL) δ_H — Tabell 8: 0.0228 (boligkapital)

    # Elastisitet kapital-arbeid
    xi       = 0.929      # (CAL) ξ — Tabell 8: 0.929 (kapital-arbeid subst.)

    # Rotemberg-prissettingskostnader (tilsvarer Calvo i lineærisert form)
    phi_PQ   = 669.0      # (CAL) φ_PQ — Tabell 8: 669 (NEMO: Rotemberg, ikke Calvo)
    phi_W    = 666.92     # (CAL) φ_W  — Tabell 8: 666.92

    # Avledede Calvo-ekvivalenter (brukt i log-linearisert form)
    # I Rotemberg: κ_P = (ε_P - 1) / φ_PQ ≈ 0.05 for ε_P ≈ 6
    # Disse er konsistente med Fase I-helningene
    @classmethod
    def kappa_P(cls):
        """NK Phillips-kurve-helning (tilsvarer Fase I ~0.05)."""
        eps_P = 6.0  # elastisitet mellom differensierte varer
        return (eps_P - 1.0) / cls.phi_PQ

    @classmethod
    def kappa_W(cls):
        """Lønnsinflasjon-helning."""
        eps_W = 6.0
        return (eps_W - 1.0) / cls.phi_W

    # Investeringsjusteringskostnader (ny i Fase II)
    # Oppr. est. under bugget INV-ligning (G1-lag, 2-periodes). Etter A4a-fix
    # (G0-lag, korrekt 1-periode) er disse verdiene inkonsistente.
    # Rekalibrert 2026-05-15: phi_I1=4, phi_I2=8 → INV std≈3.8 % (data: 4.1 %).
    phi_I1   = 4.0        # (CAL) φ_I1 — SS-justeringskost., rekalibrert etter A4a
    phi_I2   = 8.0        # (CAL) φ_I2 — periode-justeringskost., rekalibrert etter A4a
    phi_H1   = 60.7278    # (EST) φ_H1 — Tabell 8: boligi nvestering (SS-avvik)
    phi_H2   = 199.6549   # (EST) φ_H2 — Tabell 8: boliginvestering (periode-avvik)

    # Kapitalutnyttelseskostnad
    phi_u    = 0.2192     # (EST) φ_u — Tabell 8

    # ═══════════════════════════════════════════════════════════════════════
    # BLOKK C: HANDEL OG VALUTA
    # NEMO seksjon 2.10–2.12 | Tabell 8
    # ═══════════════════════════════════════════════════════════════════════

    # Substitusjonselastisitet (final goods, Armington)
    mu_M     = 0.50       # (CAL) µ  — Tabell 8: 0.5 (Fase I: 1.5 — korrigert)
    mu_X     = 0.50       # (CAL) µ* — Tabell 8: 0.5

    # Eksport- og importprisjusteringskostnader
    phi_PM   = 830.10     # (EST) φ_PM  — Tabell 8 (import)
    phi_PMX  = 285.60     # (EST) φ_PM* — Tabell 8 (eksport i utenlandsk valuta)

    # Elastisitet mellom importerte varer
    theta_F  = 6.0        # (CAL) θ_F — Tabell 8: 6
    theta_FX = 6.0        # (CAL) θ_F* — Tabell 8: 6

    # Gjeldsavhengig risikopremie
    phi_B    = 0.0016     # (CAL) φ_B — Tabell 8 (lavere enn Fase I χ=0.001)

    # Importpriskanal i NK Phillips-kurve (A14.9 — PE-godkjent 2026-05-22)
    kappa_M  = 0.03       # (CAL) κ_M — importprisinflasjon-koeff. i Phillips
    gamma_p  = 0.0        # (CAL) γ_p — Calvo-prisindeksasjon; 0=rent fremoverskuende; estimeres i kj12

    # Olje-valuta-kanal i UIP (PE-godkjent 2026-05-20)
    # Kalibrert fra historisk NOK/olje-korrelasjon ~0.7:
    # +10% oljepris → ~1.5% NOK-appresiering → phi_O ≈ 0.15
    phi_O    = 0.15       # (CAL) direkte olje→RER-kanal i UIP

    # Nasjonalregnskapsandeler (Norske data 2001–2019)
    # CY+IY+IHY+GY+XY-MY = 1.00 (Spor A5 rettelse, 2026-05-15)
    # Opprinnelig MY=0.34 inkluderte olje-sektoren; fastland ~28 %.
    # K&M (2019) § 3 (kalibrering): ressursbetingelse skal balansere.
    CY       = 0.50
    IY       = 0.20       # Inkluderer kapitalakkumulering
    IHY      = 0.10       # Boliginvestering / BNP
    GY       = 0.25
    XY       = 0.23
    MY       = 0.28       # Fastlands-import / BNP (redusert fra 0.34)

    # ═══════════════════════════════════════════════════════════════════════
    # BLOKK D: BANKSEKTOR
    # NEMO seksjon 2.8 | Gerali et al. (2010) | Tabell 8
    # ═══════════════════════════════════════════════════════════════════════

    # Basel III-krav (kapitaldekning)
    gamma_b  = 0.136      # (CAL) γ_b^ss — Tabell 10: mål kapital/risikovektet aktiva
    CCB_b    = 0.02       # (CAL) CCB — Tabell 10: kontrasyklisk kapitalbuffer

    # Justeringskostnader — bankrenter
    phi_D    = 0.0732     # (EST) φ_D — Tabell 8: innskuddsrente-justering
    phi_e    = 18.5013    # (EST) φ_e — Tabell 8: næringslånsrente-justering
    phi_F    = 18.3597    # (EST) φ_F — Tabell 8: husholdningslånsrente-justering

    # Risikovekter (kapitaldekning)
    rho_h    = 0.40       # (CAL) ρ_h — Tabell 8: risikovekt boliglån
    rho_e    = 0.80       # (CAL) ρ_e — Tabell 8: risikovekt næringslån

    # Bankens driftskostnader
    phi_o    = 0.0046     # (CAL) φ_o — Tabell 8: fast driftskostnad
    delta_b  = 0.0161     # (CAL) δ_b — Tabell 8: utbytteandel bankkapital

    # Elastisitet innskudd
    theta_D  = 7.007      # (EST) θ_D — Tabell 8

    # Avvik fra kapitalmål — kostnad
    phi_c    = 10.0       # (CAL) φ_c — Tabell 8

    # Andel oljeinntekt til banksektor (via GPFG)
    sigma_GF = 0.0501     # (CAL) σ_GF — Tabell 8

    # ═══════════════════════════════════════════════════════════════════════
    # BLOKK E: OLJESEKTOR (utvidet)
    # NEMO seksjon 2.9 | Tabell 8 og 10
    # ═══════════════════════════════════════════════════════════════════════

    # Produksjonsfaktorer i oljesektoren
    alpha_l  = 0.28       # (CAL) α_l — Tabell 8: arbeid i olje-supply
    alpha_oX = 0.15       # (CAL) α_o* — Tabell 8: utenlandsk andel i supply
    alpha_o  = 0.55       # (CAL) α_o  — Tabell 8: rigger i oljeproduksjon
    alpha_q  = 0.69       # (CAL) α_q  — Tabell 8: fastlandsvarer i supply

    # Oljekapitaldepresiering og justeringskostnader
    delta_O  = 0.021      # (CAL) δ_O — Tabell 8: rigger
    phi_RI   = 8.2151     # (EST) φ_RI — Tabell 8: oljeinvesteringsjust.
    phi_PR   = 1245.6     # (EST) φ_PR — Tabell 8: priser innenlands
    phi_PRX  = 1723.1     # (EST) φ_PR* — Tabell 8: priser utenlands

    # Elastisiteter oljemarked
    theta_R  = 400        # (CAL) θ_R  — Tabell 8: subst. supply-varer innenlands
    theta_RX = 5.0        # (CAL) θ_R* — Tabell 8: subst. supply-varer eksport

    # Produktiviteter (steady state = 1)
    Z_O      = 1.0        # (CAL) Z_O — oljeutvinning
    Z_R      = 1.0        # (CAL) Z_R — olje-supply

    # GPFG-mekanisme (full, erstatter γ_G·pO_{t-1})
    # Fremtidig implementering: se build_oil_fiscal_block()
    # For nå: forenklet overgangsregel
    gamma_G  = 0.40       # Beholdes som fallback under utvikling

    # ═══════════════════════════════════════════════════════════════════════
    # BLOKK F: PENGEPOLITIKK — MIMICKING RULE
    # NEMO seksjon 2.13 | Tabell 8
    # ═══════════════════════════════════════════════════════════════════════

    # Tapsfunksjonsvekter
    lambda_dr = 0.40      # (CAL) λ_dr — vekt rente-endring
    lambda_lr = 0.02      # (CAL) λ_lr — vekt rente-gap
    lambda_y  = 0.30      # (CAL) λ_y  — vekt outputgap

    # Mimicking rule-koeffisienter (NEMO Tabell 8)
    psi_R    = 0.6663     # (EST) ψ_R  — renteglatting mimicking rule
    psi_P1   = 0.2921     # (EST) ψ_P1 — fremtidig inflasjon (4-kv. fremoverskuende)
    psi_Y    = 0.2417     # (EST) ψ_Y  — output mimicking rule
    psi_S    = 0.0159     # (EST) ψ_S  — reell valutakurs
    psi_W    = 0.8705     # (EST) ψ_W  — lønnsinflasjon
    psi_RF   = 0.0        # (CAL) ψ_RF — utenlandsk rente

    # Diskontfaktor sentralbank
    beta_p   = 0.99       # (CAL) β_p

    # ═══════════════════════════════════════════════════════════════════════
    # BLOKK G: UTENLANDSSEKTOR
    # NEMO seksjon 2.10–2.12 | Tabell 8
    # ═══════════════════════════════════════════════════════════════════════

    # Utenlandsk Taylor-regel
    psi_R_star  = 0.8414  # (EST) ψ_R* — renteglatting handelspartnere
    psi_P_star  = 1.4606  # (EST) ψ_P* — inflasjonsrespons handelspartnere
    psi_Y_star  = 0.04    # (CAL) ψ_Y* — outputrespons handelspartnere

    # Utenlandsk diskontfaktor
    beta_star   = 0.999   # (CAL) β* — Tabell 8

    # Oljepris-dynamikk
    beta_O      = 0.2026  # (EST) β_O — forventet oljepris påvirker nåpris
    kappa_O     = 4.0027  # (EST) κ_O — global output → oljepris

    # Handelspartner-AR(1) parametere
    rho_P_star  = 0.8862  # (EST) ρ_P* — persistens inflasjon, Tabell 8
    rho_Y_star  = 0.6146  # (EST) ρ_Y* — persistens output, Tabell 8

    # ═══════════════════════════════════════════════════════════════════════
    # BLOKK H: SJOKK-PARAMETERE (Tabell 9)
    # ═══════════════════════════════════════════════════════════════════════

    # Persistens-parametere (AR(1)-koeffisienter)
    rho_A    = 0.804      # (EST) λ_zL  — TFP
    rho_C    = 0.7248     # (EST) λ_u   — konsumpreferanse
    rho_O    = 0.8736     # (EST) λ_PO* — oljepris
    rho_Ys   = 0.7825     # (EST) λ_U*  — etterspørsel (trading partners)
    rho_rp   = 0.737      # (EST) λ_B   — risikopremie (korrigert fra Fase I: 0.50)
    rho_G    = 0.9145     # (EST) λ_G   — offentlig forbruk
    rho_H    = 0.6938     # (EST) λ_h   — boligpreferanse (ny Fase II)
    rho_I    = 0.6457     # (EST) λ_I   — investeringsjust.kost. (ny)
    rho_IH   = 0.8608     # (EST) λ_IH  — boliginvesteringsjust. (ny)
    rho_IOIL = 0.834      # (EST) λ_IOIL — oljeinvesteringsteknologi (ny)
    rho_phi_e = 0.9102    # (EST) λ_θe  — LTV-sjokk entreprenører (ny)
    rho_phi_h = 0.783     # (EST) λ_θH  — LTV-sjokk husholdninger (ny)
    rho_prem = 0.8168     # (EST) λ_prem — pengemarkedspremie (ny)
    rho_pw   = 0.2797     # (EST) λ_φ   — lønnspåslagssjokk
    rho_piS  = 0.70       # (CAL) persistens utenlandsk inflasjon
    rho_iS   = 0.70       # (CAL) persistens utenlandsk rente

    # Standardavvik (Tabell 9, multiplisert med 100 → divider med 100 for modellen)
    sigma_A    = 0.00598  # σ_zL/100
    sigma_C    = 0.030209 # σ_u/100
    sigma_P    = 0.201448 # σ_θH/100  (prismarkup)
    sigma_O    = 0.079181 # σ_PO*/100
    sigma_Ys   = 0.011147 # σ_U*/100
    sigma_rp   = 0.006178 # σ_B/100
    sigma_i    = 0.000302 # σ_RN3M/100
    sigma_G    = 0.003806 # σ_G/100
    sigma_H    = 0.286767 # σ_h/100
    sigma_I    = 0.230179 # σ_I/100
    sigma_IH   = 0.02575  # σ_IH/100
    sigma_IOIL = 0.026119 # σ_IOIL/100
    sigma_phi_e = 0.025902 # σ_θe/100
    sigma_phi_h = 0.254232 # σ_θH/100
    sigma_prem  = 0.000372 # σ_prem/100
    sigma_pw    = 0.633097 # σ_φ/100

    # ═══════════════════════════════════════════════════════════════════════
    # BLOKK I: STEADY STATE-VERDIER (Tabell 10)
    # ═══════════════════════════════════════════════════════════════════════

    pi_ss       = 1.0062  # (CAL) Π_ss — kvartalsvis bruttoinflasjon
    pi_star_ss  = 1.005   # (CAL) Π*_ss — utenlandsk inflasjon
    pi_H_ss     = 1.0113  # (CAL) Π_H_ss — relativ boligprisvekst
    gz_ss       = 1.0025  # (CAL) Π_z_ss — trendproduktivitetsvekst

    # ═══════════════════════════════════════════════════════════════════════
    # KOMPATIBILITET MED FASE I (avledede verdier)
    # ═══════════════════════════════════════════════════════════════════════

    # Disse beregnes for bakoverkompatibilitet med Fase I-kode
    rho_piS_compat = 0.70
    rho_iS_compat  = 0.70

    @classmethod
    def sigma_tilde(cls):
        """
        σ̃ = σ + φ_L/(1−α_K): mc-elastisitet brukt i Fase I-mc-likning.
        Fase II: erstattes av full faktoretterspørsel.
        """
        return cls.sigma + cls.phi_L / (1.0 - cls.alpha_K)

    @classmethod
    def summary(cls):
        """Skriv ut de viktigste parametrene og Fase I-avvikene."""
        print("=" * 65)
        print("  NEMO FASE II — PARAMETEROVERSIKT")
        print("  Kravik og Mimir (2019), Tabell 8–10")
        print("=" * 65)
        rows = [
            ("β (diskontering)",       cls.beta,    0.9975, "Tabell 8"),
            ("h_c (habit konsum)",      cls.h_c,     0.74,   "Tabell 8"),
            ("φ_L (inv. Frisch)",       cls.phi_L,   1.50,   "Tabell 8"),
            ("α_K (kapitalandel)",      cls.alpha_K, 0.33,   "Tabell 8"),
            ("δ (kapitaldepr.)",        cls.delta,   0.025,  "Tabell 8"),
            ("µ_M (handelselas.)",      cls.mu_M,    1.50,   "Tabell 8"),
            ("ρ_rp (risikopremie)",     cls.rho_rp,  0.50,   "Tabell 9"),
            ("ρ_A (TFP persistens)",    cls.rho_A,   0.85,   "Tabell 9"),
            ("ρ_C (konsum persistens)", cls.rho_C,   0.75,   "Tabell 9"),
            ("ρ_O (oljepris)",          cls.rho_O,   0.85,   "Tabell 9"),
            ("ψ_R (mimicking rule)",    cls.psi_R,   0.85,   "Tabell 8"),
            ("ψ_Y (output response)",   cls.psi_Y,   0.10,   "Tabell 8"),
        ]
        print(f"\n  {'Parameter':<28} {'Fase II':>9} {'Fase I':>9}  {'Kilde'}")
        print(f"  {'─'*60}")
        for name, f2, f1, src in rows:
            diff = abs(f2 - f1) / abs(f1) * 100 if f1 != 0 else 0
            flag = " ←" if diff > 15 else ""
            print(f"  {name:<28} {f2:>9.4f} {f1:>9.4f}  {src}{flag}")
        print(f"\n  ← = endret >15% fra Fase I")
        print("=" * 65)


if __name__ == "__main__":
    Parameters.summary()
    print(f"\n  σ̃ (mc-elastisitet): {Parameters.sigma_tilde():.4f}")
    print(f"  κ_P (Phillips-helning): {Parameters.kappa_P():.5f}")
    print(f"  κ_W (lønnsinfl.-helning): {Parameters.kappa_W():.5f}")
