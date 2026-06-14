#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI news-risk flag for UFC fighters — alternative-data demo.

Idea: a pure-stats model can't beat the market, because the market prices the
same numbers. The only place ML can plausibly add signal is NEW information the
line hasn't absorbed yet (injuries, weight-cut trouble, camp changes). This
script fetches recent news per fighter and produces a structured risk flag.

Two modes:
  * Heuristic (default, no API key): keyword scoring over recent headlines.
  * LLM (optional): if ANTHROPIC_API_KEY is set, an LLM reads the headlines and
    returns a calibrated risk read (more nuanced than keywords).

Portfolio/educational tool. NOT betting advice.
"""
from __future__ import annotations
import os, sys, re, urllib.request, urllib.parse, html, datetime as dt

RISK_TERMS = {
    'injury':      ['injur','hurt','tear','torn','acl','knee','hand','rib','surgery','strain','sprain'],
    'withdrawal':  ['withdraw','pull out','pulled out','out of','off the card','replacement','cancel'],
    'weight':      ['weight cut','missed weight','miss weight','overweight','rehydrat','drained'],
    'illness':     ['ill','sick','flu','covid','hospital','staph'],
    'personal':    ['arrest','legal','suspend','usada','doping','personal reasons'],
    'form_pos':    ['looks great','best shape','career-best','sharp','dialed in','hulk'],
}
WEIGHTS = {'injury':3,'withdrawal':4,'weight':2,'illness':3,'personal':2,'form_pos':-1}

def fetch_news(name, max_items=8):
    q = urllib.parse.quote(f'{name} UFC')
    url = f'https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en'
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            xml = r.read().decode('utf-8', 'ignore')
    except Exception as e:
        return [], f'fetch error: {e}'
    items = re.findall(r'<item>(.*?)</item>', xml, re.S)[:max_items]
    out = []
    for it in items:
        t = re.search(r'<title>(.*?)</title>', it, re.S)
        d = re.search(r'<pubDate>(.*?)</pubDate>', it, re.S)
        if t:
            out.append((html.unescape(re.sub('<.*?>','',t.group(1))).strip(),
                        (d.group(1)[:16] if d else '')))
    return out, None

def heuristic_risk(headlines):
    text = ' '.join(h.lower() for h,_ in headlines)
    score = 0; hits = {}
    for cat, terms in RISK_TERMS.items():
        found = [w for w in terms if w in text]
        if found:
            hits[cat] = found; score += WEIGHTS[cat]
    level = 'HIGH' if score>=4 else 'MED' if score>=2 else ('POS' if score<0 else 'LOW')
    return level, score, hits

def llm_risk(name, headlines):
    key = os.environ.get('ANTHROPIC_API_KEY')
    if not key: return None
    try:
        import json as _j
        prompt = ("You are an MMA analyst. From these recent headlines about "
            f"{name}, output a JSON: {{risk_level: LOW/MED/HIGH, betting_relevant: bool, "
            "note: one short sentence}}. Only flag things that affect fight outcome "
            "(injury, weight, withdrawal, illness, camp issues).\n\nHeadlines:\n"
            + "\n".join(f"- {h} ({d})" for h,d in headlines))
        body = _j.dumps({"model":"claude-haiku-4-5-20251001","max_tokens":200,
            "messages":[{"role":"user","content":prompt}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
            headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            import json as _jj; resp=_jj.load(r)
        return resp["content"][0]["text"]
    except Exception as e:
        return f"(llm error: {e})"

def report(name):
    heads, err = fetch_news(name)
    print(f"\n=== {name} ===")
    if err: print("  ", err); return
    if not heads: print("   no recent news found"); return
    for h,d in heads[:6]: print(f"   • {h}  [{d}]")
    lvl, sc, hits = heuristic_risk(heads)
    print(f"   --> heuristic risk: {lvl} (score {sc}) {hits if hits else ''}")
    llm = llm_risk(name, heads)
    if llm: print(f"   --> LLM read: {llm}")

if __name__ == "__main__":
    names = sys.argv[1:] or ["Ilia Topuria","Justin Gaethje","Sean O'Malley","Aiemann Zahabi"]
    print("AI news-risk flags (alternative-data demo) —", dt.date.today().isoformat())
    for n in names: report(n)
    print("\nNote: heuristic mode by default; set ANTHROPIC_API_KEY for LLM mode. Not betting advice.")
