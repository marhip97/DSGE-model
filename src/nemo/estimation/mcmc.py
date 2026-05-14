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
    build_matrices_v3, NZ, NE,
    Y, C, INV, X, M, PI, W, I_R, RER, PO, YS,
    Q_H, B_NW, C_NW, I_D, I_L_NW, L, MC,
    E_A, E_C, E_P, E_O, E_Ys, E_rp, E_i, E_H, E_phi_h
)
from nemo.solver.blanchard_kahn import solve as bk_solve
from nemo.model.parameters import Parameters

# sigma_A kalibreres fast — svakt identifisert
SIGMA_A_FIXED = 0.006

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
    H[5,PI]=4.0; H[6,W]=1.0; H[7,I_R]=4.0; H[8,I_R]=4.0; H[9,RER]=1.0
    H[10,PO]=1.0; H[11,YS]=1.0; H[12,Q_H]=1.0; H[13,B_NW]=1.0
    return H

def build_Sv():
    sme = {'dy_obs':0.005,'dc_obs':0.008,'dinv_obs':0.015,'dx_obs':0.010,
           'dm_obs':0.012,'pi_obs':0.008,'dw_obs':0.004,'i_R_obs':0.0005,
           'i_3m_obs':0.0005,'ds_obs':0.010,'dpO_obs':0.050,'dyS_obs':0.006,
           'dh_obs':0.004,'db_obs':0.002}
    return np.diag([sme[n]**2 for n in OBS_NAMES])


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERE OG PRIOR
# sigma_A er fjernet fra estimering — kalibreres fast
# ══════════════════════════════════════════════════════════════════════════════

PARAM_PRIORS = {
    'rho_A':   ('beta',     2.0,  0.5,  0.01, 0.9995),
    'rho_C':   ('beta',     2.0,  0.5,  0.01, 0.9995),
    'rho_O':   ('beta',     2.0,  0.5,  0.01, 0.9995),
    'rho_Ys':  ('beta',     2.0,  0.5,  0.01, 0.9995),
    'rho_rp':  ('beta',     2.0,  0.5,  0.01, 0.9995),
    'rho_H':   ('beta',     2.0,  0.5,  0.01, 0.9995),
    # sigma_A er fjernet
    'sigma_C':  ('inv_gamma', 2.0, 0.0182, 1e-5, 0.5),
    'sigma_O':  ('inv_gamma', 2.0, 0.0475, 1e-5, 1.0),
    'sigma_Ys': ('inv_gamma', 2.0, 0.0067, 1e-5, 0.5),
    'sigma_rp': ('inv_gamma', 2.0, 0.0037, 1e-5, 0.5),
    'sigma_i':  ('inv_gamma', 2.0, 0.0002, 1e-5, 0.1),
    'sigma_P':  ('inv_gamma', 2.0, 0.0027, 1e-5, 0.5),
    'sigma_H':  ('inv_gamma', 2.0, 0.0500, 1e-5, 1.0),
    'psi_R':   ('beta',   4.0, 2.0,  0.30, 0.990),
    'psi_P1':  ('normal', 0.29, 0.10, 0.05, 1.50),
    'psi_Y':   ('normal', 0.24, 0.05, 0.01, 0.80),
    'h_c':     ('beta',   4.0, 1.5,  0.30, 0.9995),
}
PARAM_NAMES = list(PARAM_PRIORS.keys())
N_PARAMS    = len(PARAM_NAMES)

PARAM_NAMES_V3_FULL = ['rho_A','rho_C','rho_O','rho_Ys','rho_rp','rho_H',
                        'sigma_A','sigma_C','sigma_O','sigma_Ys','sigma_rp',
                        'sigma_i','sigma_P','sigma_H','psi_R','psi_P1','psi_Y','h_c']

KM = {'rho_A':0.804,'rho_C':0.725,'rho_O':0.874,'rho_Ys':0.783,
      'rho_rp':0.737,'rho_H':0.694,'sigma_C':0.030,
      'sigma_O':0.079,'sigma_Ys':0.011,'sigma_rp':0.006,'sigma_i':0.0003,
      'sigma_P':0.003,'sigma_H':0.050,'psi_R':0.666,'psi_P1':0.292,
      'psi_Y':0.242,'h_c':0.938}

def log_prior(theta):
    lp = 0.0
    for i, name in enumerate(PARAM_NAMES):
        x = theta[i]; spec = PARAM_PRIORS[name]; lb,ub = spec[-2],spec[-1]
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
    return lp


# ══════════════════════════════════════════════════════════════════════════════
# KALMAN-FILTER MED COVID-HULL
# ══════════════════════════════════════════════════════════════════════════════

def build_Q(theta):
    # sigma_A er fast
    smap = {E_C:'sigma_C',E_P:'sigma_P',E_O:'sigma_O',
            E_Ys:'sigma_Ys',E_rp:'sigma_rp',E_i:'sigma_i',E_H:'sigma_H'}
    Q = np.zeros((NE,NE))
    Q[E_A,E_A] = SIGMA_A_FIXED**2   # fast
    for idx,pn in smap.items():
        s = theta[PARAM_NAMES.index(pn)] if pn in PARAM_NAMES else getattr(Parameters,pn,0.01)
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

def log_posterior(theta, H, Sv, Y_pre, Y_post):
    lp=log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    try:
        class Pt(Parameters): pass
        for i,n in enumerate(PARAM_NAMES): setattr(Pt,n,float(theta[i]))
        setattr(Pt,'sigma_A',SIGMA_A_FIXED)   # fast
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
        save_prefix="chain_v3_v2"):

    rng=np.random.default_rng(seed); N=N_PARAMS
    scale=scale_init; post_std=post_std_init.copy()
    C_prop=np.diag((scale*2.38/np.sqrt(N)*post_std)**2+1e-12)
    theta=theta_init.copy()
    lp_cur=log_posterior(theta,H,Sv,Y_pre,Y_post)
    if not np.isfinite(lp_cur):
        raise ValueError(f"Startverdi gir ikke-endelig log-posterior: {lp_cur}")
    recalib_log=[]; t_total=time.time()

    def _run(n_steps, phase, adapt=False, monitor=False):
        nonlocal theta,lp_cur,scale,C_prop
        ch=np.zeros((n_steps,N)); lp_v=np.zeros(n_steps)
        acc=0; acc_win=0; t0=time.time()
        for i in range(n_steps):
            tp=theta+rng.multivariate_normal(np.zeros(N),C_prop)
            lpp=log_posterior(tp,H,Sv,Y_pre,Y_post)
            if np.log(rng.uniform())<lpp-lp_cur:
                theta=tp; lp_cur=lpp; acc+=1; acc_win+=1
            ch[i]=theta; lp_v[i]=lp_cur
            if adapt and (i+1)%adapt_every==0:
                rate=acc_win/adapt_every; acc_win=0
                if   rate<0.10: scale*=0.60
                elif rate<0.15: scale*=0.75
                elif rate<0.20: scale*=0.88
                elif rate>0.40: scale*=1.40
                elif rate>0.32: scale*=1.20
                elif rate>0.28: scale*=1.10
                scale=float(np.clip(scale,0.005,10.0))
                C_prop=np.diag((scale*2.38/np.sqrt(N)*post_std)**2+1e-12)
            if monitor and (i+1)%check_every==0:
                ok,pmax,emin,_=check_convergence(ch[:i+1],psrf_thr,ess_pct_thr)
                rem=(time.time()-t0)/(i+1)*(n_steps-i-1)
                status="OK" if ok else f"PSRF={pmax:.2f} ESS={emin:.0f}"
                if verbose:
                    print(f"  [{i+1:7d}/{n_steps}] acc={acc/(i+1):.3f}  "
                          f"lp={lp_cur:.1f}  scale={scale:.4f}  "
                          f"konv={status}  gjenstår≈{rem/60:.1f}min")
                # Løpende lagring
                np.save(f"{save_prefix}_partial.npy", ch[:i+1])
            elif verbose and (i+1)%5000==0:
                rem=(time.time()-t0)/(i+1)*(n_steps-i-1)
                print(f"  [{i+1:7d}/{n_steps}] acc={acc/(i+1):.3f}  "
                      f"lp={lp_cur:.1f}  scale={scale:.4f}  "
                      f"gjenstår≈{rem/60:.1f}min  [{phase}]")
        return ch, lp_v, acc/n_steps

    if verbose:
        print(f"\n{'='*65}")
        print(f"  NEMO v3 — ADAPTIV MCMC v2 (sigma_A fast={SIGMA_A_FIXED})")
        print(f"  T_pre={len(Y_pre)} kv  T_post={len(Y_post)} kv  N={N} param")
        print(f"  Produksjon={n_production:,}  Burn-in={burnin:,}")
        print(f"  Startscale={scale_init:.4f}  PSRF-krav<{psrf_thr}")
        print(f"{'='*65}")
        print(f"\n  Startverdi log-posterior: {lp_cur:.2f}")
        print(f"\n--- FASE 1: Adaptiv burn-in ({burnin:,} trekk) ---")

    ch_bi,_,acc_bi=_run(burnin,"BURN-IN",adapt=True)
    if verbose: print(f"  Ferdig. acc={acc_bi:.3f}  scale={scale:.4f}")

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
        C_prop=np.diag((scale*2.38/np.sqrt(N)*post_std)**2+1e-12)
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
    obs_df=pd.read_csv("data/processed/nemo_data_faktisk_v2.csv",index_col=0,parse_dates=True)
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
                theta_start[i] = KM.get(name,0.5); post_std[i]=0.05
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
