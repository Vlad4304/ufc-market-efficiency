#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UFC forward-test tracker v2: Elo + calibration + edge cap + sharp(Pinnacle) benchmark + CLV.
Measurement tool, not betting advice."""
from __future__ import annotations
import os, re, json, urllib.request, datetime as dt
from collections import defaultdict
import numpy as np, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH   = os.path.join(SCRIPT_DIR, "picks_log.csv")
KEY_PATH   = os.path.join(SCRIPT_DIR, "odds_api_key.txt")
MASTER_URL = "https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/main/ufc-master.csv"
ODDS_URL   = ("https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/"
             "?regions=us,eu&markets=h2h,totals&oddsFormat=american&apiKey={key}")
EV_THRESHOLD = 0.05
EV_CAP       = 0.30          # edges above this are treated as model error -> skipped
ODDS_MIN, ODDS_MAX = -350, 600
TOTALS_POINT = 2.5
SHARP_BOOKS = ['pinnacle','betfair_ex_eu','matchbook']   # priority order
CALIB_FRAC  = 0.15
EXCLUDE = {'R_fighter','B_fighter','date','location','country','Winner','finish',
    'finish_details','finish_round','finish_round_time','total_fight_time_secs',
    'R_odds','B_odds','R_ev','B_ev','r_dec_odds','b_dec_odds','r_ko_odds','b_ko_odds',
    'r_sub_odds','b_sub_odds','weight_class','gender','better_rank'}
COLUMNS=['logged_at','fight_id','date','fighter_R','fighter_B','market','selection','book',
    'american_odds','decimal_odds','model_prob','implied_prob','edge_pct','sharp_prob',
    'model_vs_sharp','closing_decimal','clv_pct','result','pnl_units']

def log(m): print(f"[tracker] {m}", flush=True)
def norm(s): return re.sub(r'\s+',' ', re.sub(r'[^a-z ]','', str(s).lower().strip()))
def fid(a,b,d): return "|".join(sorted([norm(a),norm(b)]))+"@"+str(d)[:10]
def a2d(a):
    a=float(a); return 1+a/100 if a>0 else 1+100/abs(a)
def devig2(pa,pb):
    s=pa+pb; return (pa/s, pb/s) if s>0 else (pa,pb)
def method_of(f):
    if f in ('U-DEC','S-DEC','M-DEC'): return 'DEC'
    if f=='KO/TKO': return 'KO'
    if f=='SUB': return 'SUB'
    return None
def fetch_master():
    p="/tmp/_master.csv"; urllib.request.urlretrieve(MASTER_URL,p); return pd.read_csv(p)
def fetch_live_odds():
    key=open(KEY_PATH).read().strip()
    with urllib.request.urlopen(ODDS_URL.format(key=key), timeout=40) as r:
        rem=r.headers.get('x-requests-remaining'); data=json.load(r)
    log(f"live odds: {len(data)} MMA events (API remaining: {rem})"); return data

def add_elo(master):
    m=master.copy(); m['date']=pd.to_datetime(m['date'],errors='coerce')
    m=m.sort_values('date').reset_index(drop=True)
    elo=defaultdict(lambda:1500.0); Rp=[]; Bp=[]; K=32
    for _,r in m.iterrows():
        a=norm(r['R_fighter']); b=norm(r['B_fighter']); ra=elo[a]; rb=elo[b]
        Rp.append(ra); Bp.append(rb)
        ea=1/(1+10**((rb-ra)/400))
        w=r['Winner']; sa=1.0 if w=='Red' else (0.0 if w=='Blue' else 0.5)
        elo[a]=ra+K*(sa-ea); elo[b]=rb+K*((1-sa)-(1-ea))
    m['R_elo']=Rp; m['B_elo']=Bp; m['elo_diff']=np.array(Rp)-np.array(Bp)
    return m, dict(elo)

def _calibrated(Xtr,ytr,Xcal,ycal):
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    try: from sklearn.frozen import FrozenEstimator
    except Exception: FrozenEstimator=None
    base=XGBClassifier(n_estimators=350,max_depth=4,learning_rate=0.05,subsample=0.9,
        colsample_bytree=0.8,eval_metric='logloss',random_state=42)
    base.fit(Xtr,ytr)
    if FrozenEstimator is not None:
        cal=CalibratedClassifierCV(FrozenEstimator(base),method='isotonic'); cal.fit(Xcal,ycal)
        return cal
    return base

def build_models(master):
    d=master[master['Winner'].isin(['Red','Blue'])].copy()
    d['m']=d['finish'].map(method_of); d=d[d['m'].notna()].copy()
    feats=[c for c in d.columns if c not in EXCLUDE and not c.endswith('_dif')]
    X=d[feats].apply(pd.to_numeric,errors='coerce'); X=X.loc[:,X.notna().any()]; cols=list(X.columns)
    n=len(d); cut=int(n*(1-CALIB_FRAC))
    ywin=(d['Winner']=='Red').astype(int).values
    win=_calibrated(X.iloc[:cut],ywin[:cut],X.iloc[cut:],ywin[cut:])
    fr=pd.to_numeric(d['finish_round'],errors='coerce').values
    msk=(~np.isnan(fr)) & (fr>=1) & (fr<=5)
    from xgboost import XGBClassifier
    yr=(fr[msk].astype(int)-1)  # classes 0..4 -> rounds 1..5
    rnd=XGBClassifier(n_estimators=350,max_depth=4,learning_rate=0.05,subsample=0.9,
        colsample_bytree=0.8,objective='multi:softprob',num_class=5,
        eval_metric='mlogloss',random_state=42)
    rnd.fit(X[msk],yr)
    log(f"models trained on {n} fights, {len(cols)} features (incl. Elo); rounds=distribution"); return win,rnd,cols

def over_prob(rnd_proba, point):
    # P(fight lasts beyond point rounds) = P(finish_round >= floor(point)+1)
    import math
    k=int(math.floor(point))  # over k.5 -> need round >= k+1
    # rnd_proba index i -> round i+1
    return float(sum(rnd_proba[i] for i in range(len(rnd_proba)) if (i+1)>point))

def build_profiles(master, final_elo):
    R=[c for c in master.columns if c.startswith('R_') and c not in ('R_odds','R_ev','R_fighter','R_elo')]
    neutral=[c[2:] for c in R]
    m=master.copy(); m['date']=pd.to_datetime(m['date'],errors='coerce'); m=m.sort_values('date')
    prof={}
    for _,row in m.iterrows():
        for corner,fcol in [('R','R_fighter'),('B','B_fighter')]:
            name=norm(row[fcol])
            if not name: continue
            prof[name]={n: row[f'{corner}_{n}'] for n in neutral}
    log(f"built profiles for {len(prof)} fighters"); return prof,neutral

def make_feature_row(cols,neutral,prof,final_elo,rname,bname,no_rounds=3):
    a,b=norm(rname),norm(bname)
    if a not in prof or b not in prof: return None
    row={}
    for n in neutral: row[f'R_{n}']=prof[a][n]; row[f'B_{n}']=prof[b][n]
    ea=final_elo.get(a,1500.0); eb=final_elo.get(b,1500.0)
    row['R_elo']=ea; row['B_elo']=eb; row['elo_diff']=ea-eb
    row['title_bout']=0; row['no_of_rounds']=no_rounds; row['empty_arena']=0
    return pd.DataFrame([row]).reindex(columns=cols).apply(pd.to_numeric,errors='coerce')

def load_log():
    if os.path.exists(LOG_PATH):
        try:
            df=pd.read_csv(LOG_PATH); df=df[df['fight_id'].notna()].reset_index(drop=True)
            for c in COLUMNS:
                if c not in df.columns: df[c]=''
            return df
        except Exception: pass
    return pd.DataFrame(columns=COLUMNS)

def sharp_prob_for(ev):
    """return dict name->de-vigged sharp prob from best available sharp book."""
    for sb in SHARP_BOOKS:
        for bk in ev.get('bookmakers',[]):
            if bk['key']==sb:
                for mk in bk.get('markets',[]):
                    if mk['key']=='h2h' and len(mk['outcomes'])==2:
                        o=mk['outcomes']
                        try:
                            pa=1/a2d(o[0]['price']); pb=1/a2d(o[1]['price'])
                            pa,pb=devig2(pa,pb)
                            return {o[0]['name']:pa, o[1]['name']:pb}
                        except Exception: pass
    return {}

def main():
    master=fetch_master()
    master,final_elo=add_elo(master)
    win,rnd,cols=build_models(master)
    prof,neutral=build_profiles(master,final_elo)
    try: events=fetch_live_odds()
    except Exception as e: log(f"ERROR fetching live odds: {e}"); events=[]

    logdf=load_log()
    seen=set((logdf['fight_id'].astype(str)+'::'+logdf['market'].astype(str)+'::'+
              logdf['selection'].astype(str)+'::'+logdf['book'].astype(str))) if len(logdf) else set()
    new=[]; today=dt.date.today().isoformat(); skipped=0; capped=0
    now=dt.datetime.now(dt.timezone.utc); horizon=now+dt.timedelta(days=21)
    # current consensus decimal per (fid,market,selection) for CLV closing capture
    consensus=defaultdict(list)

    for ev in events:
        try:
            ct=dt.datetime.fromisoformat(ev['commence_time'].replace('Z','+00:00'))
            if ct>horizon or ct<now-dt.timedelta(days=2): continue
        except Exception: ct=None
        date=str(ev.get('commence_time',''))[:10]
        # favorite by consensus across ALL books -> Red corner
        imp=defaultdict(list)
        for bk in ev.get('bookmakers',[]):
            for mk in bk.get('markets',[]):
                if mk['key']=='h2h':
                    for o in mk['outcomes']:
                        try: imp[o['name']].append(1/a2d(o['price']))
                        except Exception: pass
        if len(imp)<2: continue
        avg={k:np.mean(v) for k,v in imp.items()}
        fav=max(avg,key=avg.get); dog=[k for k in avg if k!=fav][0]
        R_name,B_name=fav,dog
        Xrow=make_feature_row(cols,neutral,prof,final_elo,R_name,B_name,3)
        if Xrow is None: skipped+=1; continue
        p_R=float(win.predict_proba(Xrow)[0,1]); rnd_proba=rnd.predict_proba(Xrow)[0]
        sp=sharp_prob_for(ev)  # name->prob
        f=fid(R_name,B_name,date)
        for bk in ev.get('bookmakers',[]):
            book=bk['key']
            for mk in bk.get('markets',[]):
                if mk['key']=='h2h':
                    for o in mk['outcomes']:
                        sel='R' if norm(o['name'])==norm(R_name) else 'B'
                        prob=p_R if sel=='R' else 1-p_R
                        spv=sp.get(o['name'],'')
                        try: consensus[(f,'moneyline',sel)].append(a2d(o['price']))
                        except Exception: pass
                        _try_log(new,seen,today,f,date,R_name,B_name,'moneyline',sel,book,o['price'],prob,spv)
                elif mk['key']=='totals':
                    for o in mk['outcomes']:
                        pt=float(o.get('point',0))
                        if abs(pt-round(pt))<1e-6: continue   # skip integer lines (push risk)
                        po=over_prob(rnd_proba,pt)
                        prob=po if o['name']=='Over' else 1-po
                        mname=f'rounds_o/u{pt}'
                        try: consensus[(f,mname,o['name'])].append(a2d(o['price']))
                        except Exception: pass
                        _try_log(new,seen,today,f,date,R_name,B_name,mname,o['name'],book,o['price'],prob,'')

    capped=_CAP_COUNT[0]
    if new:
        logdf=pd.concat([logdf,pd.DataFrame(new)],ignore_index=True); log(f"logged {len(new)} new +EV picks")
    else: log("no new +EV picks this run")
    if skipped: log(f"skipped {skipped} fights (debut/no history)")
    if capped: log(f"capped {capped} absurd-edge picks (>{int(EV_CAP*100)}%)")

    # ---- CLV closing capture: overwrite closing for pending picks of imminent/active events ----
    cons_med={k:float(np.median(v)) for k,v in consensus.items()}
    upd=0
    for i,row in logdf.iterrows():
        if str(row.get('result','')) not in ('','nan'): continue
        key=(row['fight_id'],row['market'],row['selection'])
        if key in cons_med:
            cd=cons_med[key]; logdf.at[i,'closing_decimal']=round(cd,3)
            try: logdf.at[i,'clv_pct']=round((float(row['decimal_odds'])/cd-1)*100,2)
            except Exception: pass
            upd+=1
    if upd: log(f"updated closing/CLV for {upd} pending picks")

    # ---- reconcile settled fights from history ----
    res={}
    md=master[master['Winner'].isin(['Red','Blue'])].copy(); md['m']=md['finish'].map(method_of)
    md['fr']=pd.to_numeric(md['finish_round'],errors='coerce')
    for _,r in md.iterrows(): res[fid(r['R_fighter'],r['B_fighter'],r['date'])]=(r['Winner'][0],r['fr'])
    settled=0
    for i,row in logdf.iterrows():
        if str(row.get('result','')) not in ('','nan'): continue
        r=res.get(row['fight_id'])
        if not r: continue
        wc,fr=r
        if row['market']=='moneyline': hit=(row['selection']==wc)
        elif row['market'].startswith('rounds'):
            try: pt=float(str(row['market']).split('o/u')[1])
            except Exception: continue
            if pd.isna(fr): continue
            over=fr>pt
            hit=(row['selection']=='Over' and over) or (row['selection']=='Under' and not over)
        else: continue
        logdf.at[i,'result']='WIN' if hit else 'LOSS'
        logdf.at[i,'pnl_units']=round((float(row['decimal_odds'])-1) if hit else -1.0,3); settled+=1
    if settled: log(f"reconciled {settled} settled picks")
    logdf=logdf[COLUMNS]
    logdf.to_csv(LOG_PATH,index=False)

    # ---- scorecard ----
    s=logdf[logdf['result'].isin(['WIN','LOSS'])].copy()
    print("\n================ FORWARD-TEST SCORECARD ================")
    print(f"Total picks: {len(logdf)}  (settled: {len(s)}, pending: {len(logdf)-len(s)})")
    clv=pd.to_numeric(logdf['clv_pct'],errors='coerce').dropna()
    if len(clv): print(f"CLV captured: {len(clv)} picks | avg CLV={clv.mean():+.2f}%  (>0 = beating closing = good)")
    if len(s):
        s['pnl_units']=pd.to_numeric(s['pnl_units'],errors='coerce')
        for grp in ['ALL']+sorted(s['market'].unique()):
            sub=s if grp=='ALL' else s[s['market']==grp]; n=len(sub); w=(sub['result']=='WIN').sum()
            roi=sub['pnl_units'].sum()/n*100 if n else 0
            print(f"  {grp:16s}: n={n:4d} hit%={w/n*100:5.1f} ROI={roi:+6.2f}% P/L={sub['pnl_units'].sum():+.2f}u")
        print("\n  Trust ROI/CLV only after ~150-200+ settled bets. Small n = noise.")
    else: print("  No settled bets yet.")
    print(f"\nLog: {LOG_PATH}\n=======================================================")

_CAP_COUNT=[0]
def _try_log(new,seen,today,f,date,home,away,market,sel,book,price,prob,sharp_prob=''):
    try: am=float(price)
    except: return
    if not (ODDS_MIN<=am<=ODDS_MAX): return
    dec=a2d(am); edge=prob*dec-1
    if edge<EV_THRESHOLD: return
    if edge>EV_CAP: _CAP_COUNT[0]+=1; return
    k=f"{f}::{market}::{sel}::{book}"
    if k in seen: return
    seen.add(k)
    mvs = (round(prob-float(sharp_prob),4) if sharp_prob not in ('',None) else '')
    new.append(dict(logged_at=today,fight_id=f,date=date,fighter_R=home,fighter_B=away,
        market=market,selection=sel,book=book,american_odds=am,decimal_odds=round(dec,3),
        model_prob=round(prob,4),implied_prob=round(1/dec,4),edge_pct=round(edge*100,1),
        sharp_prob=(round(float(sharp_prob),4) if sharp_prob not in ('',None) else ''),
        model_vs_sharp=mvs,closing_decimal='',clv_pct='',result='',pnl_units=''))

if __name__=="__main__": main()
