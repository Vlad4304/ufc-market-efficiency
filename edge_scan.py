import json,urllib.request,itertools
from collections import defaultdict
import os
K=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'odds_api_key.txt')).read().strip()
def a2d(a):
    a=float(a); return 1+a/100 if a>0 else 1+100/abs(a)
def fetch(markets):
    url=f"https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/?regions=us,eu&markets={markets}&oddsFormat=decimal&apiKey={K}"
    with urllib.request.urlopen(url,timeout=40) as r:
        print("api remaining:",r.headers.get('x-requests-remaining'))
        return json.load(r)
data=fetch("h2h")
import datetime as dt
now=dt.datetime.now(dt.timezone.utc); hor=now+dt.timedelta(days=21)

arbs=[]; values=[]
for ev in data:
    try:
        ct=dt.datetime.fromisoformat(ev['commence_time'].replace('Z','+00:00'))
        if ct>hor or ct<now-dt.timedelta(hours=6): continue
    except: continue
    home,away=ev['home_team'],ev['away_team']
    best={}   # name -> (best_decimal, book)
    pin={}    # pinnacle decimals
    for bk in ev.get('bookmakers',[]):
        for mk in bk.get('markets',[]):
            if mk['key']!='h2h': continue
            for o in mk['outcomes']:
                n=o['name']; d=float(o['price'])
                if n not in best or d>best[n][0]: best[n]=(d,bk['key'])
                if bk['key']=='pinnacle': pin[n]=d
    if len(best)<2: continue
    names=list(best.keys())[:2]
    # ---- ARBITRAGE: best price each side ----
    inv=sum(1/best[n][0] for n in names)
    if inv<1:
        arbs.append((home,away,inv,{n:best[n] for n in names}))
    # ---- VALUE vs PINNACLE (de-vig) ----
    if len(pin)>=2:
        s=sum(1/pin[n] for n in pin)
        for n in names:
            if n not in pin: continue
            true_p=(1/pin[n])/s            # sharp de-vigged prob
            bd,bk=best[n]
            ev_pct=true_p*bd-1
            if ev_pct>0.02:
                values.append((ev_pct,home,away,n,bd,bk,true_p))

print(f"\n===== ARBITRAGE (guaranteed profit, any sizing) — {len(arbs)} found =====")
for h,a,inv,b in sorted(arbs,key=lambda x:x[2]):
    print(f"  {h} vs {a}: profit={ (1/inv-1)*100:.2f}%  -> {b}")
if not arbs: print("  none")

print(f"\n===== +EV vs PINNACLE (best book beats sharp line) — {len(values)} found =====")
for ev_pct,h,a,n,bd,bk,tp in sorted(values,reverse=True)[:25]:
    print(f"  +{ev_pct*100:4.1f}%  {n:22s} @ {bd:.2f} ({bk:12s}) | sharp_p={tp:.3f} | {h} vs {a}")
if not values: print("  none")
