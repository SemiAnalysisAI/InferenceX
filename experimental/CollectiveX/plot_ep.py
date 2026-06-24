#!/usr/bin/env python3
"""CollectiveX — render EP dispatch/combine sweeps to a self-contained HTML.

Reads the family=moe result JSONs (tests/run_ep.py output) and emits ONE
dependency-free HTML file (inline SVG, no CDN — opens offline) with:

  * an interactive explorer: operation (dispatch | combine | round-trip) x
    phase (decode | prefill) x x-axis (tokens/rank | global tokens) x y-axis
    (latency | tokens/s | alg bandwidth), one colored line per SKU/backend/EP;
  * a static small-multiples grid (phase x operation) of latency vs tokens/rank.

Only source-tokens-per-rank varies along a line; everything else (backend, EP
degree, phase, precision, top-k/experts/hidden, routing) is fixed and identifies
the line — per the CollectiveX EP framework.

    python3 plot_ep.py --results-dir results --out results/plots/collectivex_ep.html
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# SKU -> color (matches the matplotlib convention used for the NCCL plots).
COLORS = {"b200": "#1f77b4", "gb200": "#2ca02c", "mi355x": "#d62728",
          "b300": "#9467bd", "gb300": "#8c564b", "h100": "#ff7f0e", "h200": "#e377c2"}


def load_series(results_dir: str) -> list[dict]:
    series = []
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("family") != "moe" or not d.get("rows"):
            continue
        sku = (d.get("runner") or "?").split("_")[0].split("-")[0]
        rows = []
        for r in d["rows"]:
            op = {k: r.get(f"{k}_us_p50") for k in ("dispatch", "combine")}
            op["serial"] = r.get("serial_us_p50") or r.get("roundtrip_us_p50")  # serial=D+C (old: roundtrip)
            if not all(op.values()):
                continue
            rows.append({
                "t": r["tokens_per_rank"], "gt": r.get("global_tokens"),
                "dispatch": op["dispatch"], "combine": op["combine"], "serial": op["serial"],
                "fanout": r.get("fanout_mean"),
                # comm-only-v1 schema: routed_bytes_total (Σ recv across ranks, one-way) +
                # recv_tokens_max; fall back to the old single-point fields.
                "bytes": r.get("routed_bytes_total") or r.get("dispatch_bytes") or 0,
                "recv": r.get("recv_tokens_max") or r.get("recv_tokens") or 0,
                "correct": bool(r.get("correct")),
            })
        if not rows:
            continue
        sh = d.get("shape", {})
        mode = d.get("mode", "normal")
        ml = "" if mode == "normal" else f" · {mode.upper()}"
        series.append({
            "sku": sku, "backend": d.get("backend"), "ep": d.get("ep_size"),
            "phase": d.get("phase", "decode"), "mode": mode,
            "label": f'{sku.upper()} · {d.get("backend")} · EP{d.get("ep_size")}{ml}',
            "color": COLORS.get(sku, "#555"),
            "topo": d.get("topology_class"), "transport": d.get("transport"),
            "contract": d.get("measurement_contract", "?"),
            "prov": d.get("backend_provenance", {}),
            "shape": sh, "rows": rows,
        })
    return series


HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CollectiveX — EP dispatch / combine</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--ink:#e6e9ef;--mut:#9aa4b2;--line:#2a2f3a;--accent:#5b8def}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:24px 18px 64px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;color:var(--mut);font-weight:600;margin:28px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--mut);font-size:12.5px;margin:0 0 18px}
.controls{display:flex;flex-wrap:wrap;gap:14px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.grp{display:flex;flex-direction:column;gap:5px}
.grp .lab{font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--mut)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{background:transparent;color:var(--mut);border:0;padding:6px 11px;font-size:12.5px;cursor:pointer}
.seg button:hover{color:var(--ink)}
.seg button.on{background:var(--accent);color:#fff}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:6px 2px 0;color:var(--mut);font-size:12.5px}
.legend .it{display:flex;align-items:center;gap:7px}
.legend .sw{width:22px;height:3px;border-radius:2px;display:inline-block}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.gtit{font-size:12.5px;color:var(--ink);margin:0 0 2px;font-weight:600}
.note{color:var(--mut);font-size:12px;margin-top:10px}
svg{display:block;width:100%;height:auto}
.ax{stroke:var(--line);stroke-width:1}.gl{stroke:var(--line);stroke-width:1;opacity:.45}
.tk{fill:var(--mut);font-size:11px}.axl{fill:var(--mut);font-size:11.5px}
.ttl{fill:var(--ink);font-size:13px;font-weight:600}
circle.pt{stroke:#0f1115;stroke-width:1}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>CollectiveX — EP dispatch / combine</h1>
<p class="sub" id="prov"></p>
"""

TAIL = "</div></body></html>"

JS = r"""
const SKUS = [...new Set(DATA.map(s=>s.sku))];
const OPS = {dispatch:"Dispatch", combine:"Combine", serial:"Serial D+C"};
const YK  = {lat:"Latency (µs)", tps:"Tokens / s", bw:"Alg bandwidth (GB/s)"};
const XK  = {t:"Source tokens / rank", gt:"Global source tokens"};
const ST  = {op:"dispatch", phase:"decode", x:"t", y:"lat", ylog:true};

function xval(r,xk){ return xk==="t"? r.t : r.gt; }
function metric(r,op,yk){
  const us=r[op];
  if(yk==="lat") return us;
  if(yk==="tps") return r.gt/(us*1e-6);
  return us>0 ? r.bytes/(us*1e3) : 0;   // GB/s, dispatch payload as the volume proxy
}
function fmt(v){
  if(v>=1e9) return (v/1e9).toFixed(v<1e10?2:0)+"G";
  if(v>=1e6) return (v/1e6).toFixed(v<1e7?2:0)+"M";
  if(v>=1e3) return (v/1e3).toFixed(v<1e4?1:0)+"k";
  if(v>=10)  return v.toFixed(0);
  if(v>=1)   return v.toFixed(v<3?1:0);
  return v.toFixed(2);
}
function logTicks(mn,mx){
  const t=[]; let e=Math.floor(Math.log10(mn));
  for(;Math.pow(10,e)<=mx*1.0001;e++) for(const m of [1,2,5]){const v=m*Math.pow(10,e); if(v>=mn*0.999&&v<=mx*1.001)t.push(v);}
  return t.length?t:[mn,mx];
}
function linTicks(mn,mx){
  const span=mx-mn||1, step=Math.pow(10,Math.floor(Math.log10(span))); const t=[];
  let s=step; if(span/step>6)s=step*2; if(span/step<3)s=step/2;
  for(let v=Math.ceil(mn/s)*s; v<=mx*1.0001; v+=s) t.push(+v.toFixed(6));
  return t.length?t:[mn,mx];
}
const mapLog=(v,a,b,p,q)=>p+(Math.log(v)-Math.log(a))/(Math.log(b)-Math.log(a))*(q-p);
const mapLin=(v,a,b,p,q)=>p+(v-a)/(b-a)*(q-p);

// Build one SVG chart. opts: {op,phase,x,y,ylog,title,legend,w,h}
function chart(o){
  const W=o.w||900, H=o.h||520, m={l:64,r:16,t:34,b:46};
  const sl = DATA.filter(s=>s.phase===o.phase && (o.ep==null || s.ep===o.ep));
  const pts = sl.map(s=>({s, P:s.rows.map(r=>({x:xval(r,o.x), y:metric(r,o.op,o.y), r}))
                                     .filter(p=>p.x>0 && (o.ylog? p.y>0 : p.y>=0))}));
  let xs=[], ys=[]; pts.forEach(g=>g.P.forEach(p=>{xs.push(p.x);ys.push(p.y);}));
  if(!xs.length) return '<svg viewBox="0 0 '+W+' '+H+'"><text x="'+(W/2)+'" y="'+(H/2)+'" class="axl" text-anchor="middle">no data</text></svg>';
  const xmn=Math.min(...xs), xmx=Math.max(...xs);
  let ymn=Math.min(...ys), ymx=Math.max(...ys);
  if(o.ylog){ ymn=Math.min(...ys.filter(v=>v>0)); } else { ymn=Math.min(0,ymn); }
  if(ymx===ymn) ymx=ymn+1;
  const X0=m.l,X1=W-m.r,Y0=H-m.b,Y1=m.t;
  const xv=v=>mapLog(v,xmn,xmx,X0,X1);                      // x always log (geometric sweep)
  const yv=v=>o.ylog?mapLog(Math.max(v,ymn),ymn,ymx,Y0,Y1):mapLin(v,ymn,ymx,Y0,Y1);
  let s='<svg viewBox="0 0 '+W+' '+H+'" role="img">';
  s+='<text x="'+X0+'" y="20" class="ttl">'+o.title+'</text>';
  // y grid + ticks
  const yt=o.ylog?logTicks(ymn,ymx):linTicks(ymn,ymx);
  yt.forEach(v=>{const y=yv(v); s+='<line class="gl" x1="'+X0+'" y1="'+y+'" x2="'+X1+'" y2="'+y+'"/>'+
    '<text class="tk" x="'+(X0-7)+'" y="'+(y+3.5)+'" text-anchor="end">'+fmt(v)+'</text>';});
  // x grid + ticks (label the actual sweep points)
  const xt=[...new Set(xs)].sort((a,b)=>a-b);
  xt.forEach(v=>{const x=xv(v); s+='<line class="gl" x1="'+x+'" y1="'+Y0+'" x2="'+x+'" y2="'+Y1+'"/>'+
    '<text class="tk" x="'+x+'" y="'+(Y0+16)+'" text-anchor="middle">'+fmt(v)+'</text>';});
  // axes
  s+='<line class="ax" x1="'+X0+'" y1="'+Y0+'" x2="'+X1+'" y2="'+Y0+'"/><line class="ax" x1="'+X0+'" y1="'+Y0+'" x2="'+X0+'" y2="'+Y1+'"/>';
  s+='<text class="axl" x="'+((X0+X1)/2)+'" y="'+(H-6)+'" text-anchor="middle">'+XK[o.x]+'  (log)</text>';
  s+='<text class="axl" transform="translate(15,'+((Y0+Y1)/2)+') rotate(-90)" text-anchor="middle">'+YK[o.y]+(o.ylog?'  (log)':'')+'</text>';
  // lines + points
  pts.forEach(g=>{ if(!g.P.length) return;
    const d=g.P.map((p,i)=>(i?'L':'M')+xv(p.x).toFixed(1)+' '+yv(p.y).toFixed(1)).join(' ');
    s+='<path d="'+d+'" fill="none" stroke="'+g.s.color+'" stroke-width="2"/>';
    g.P.forEach(p=>{ s+='<circle class="pt" cx="'+xv(p.x).toFixed(1)+'" cy="'+yv(p.y).toFixed(1)+'" r="3.2" fill="'+g.s.color+'">'+
      '<title>'+g.s.label+'  ·  T/rank='+p.r.t+'  global='+p.r.gt+'\n'+OPS[o.op]+': '+p.r[o.op].toFixed(1)+' µs'+
      '\ntokens/s='+fmt(p.r.gt/(p.r[o.op]*1e-6))+'  ·  fan-out='+(p.r.fanout!=null?p.r.fanout.toFixed(2):'?')+
      '  ·  recv(max)='+p.r.recv+(p.r.correct?'':'  ✗')+'</title></circle>'; });
  });
  s+='</svg>'; return s;
}
function legend(phase, ep){
  return '<div class="legend">'+DATA.filter(s=>s.phase===phase && (ep==null||s.ep===ep)).map(s=>
    '<span class="it"><span class="sw" style="background:'+s.color+'"></span>'+s.label+'</span>').join('')+'</div>';
}
function seg(name,opts,cur){
  return '<div class="seg">'+Object.entries(opts).map(([k,v])=>
    '<button data-grp="'+name+'" data-val="'+k+'" class="'+(k===cur?'on':'')+'">'+v+'</button>').join('')+'</div>';
}
function renderControls(){
  document.getElementById('controls').innerHTML =
    '<div class="grp"><span class="lab">Operation</span>'+seg('op',OPS,ST.op)+'</div>'+
    '<div class="grp"><span class="lab">Phase</span>'+seg('phase',{decode:"Decode",prefill:"Prefill"},ST.phase)+'</div>'+
    '<div class="grp"><span class="lab">X-axis</span>'+seg('x',XK,ST.x)+'</div>'+
    '<div class="grp"><span class="lab">Y-axis</span>'+seg('y',YK,ST.y)+'</div>'+
    '<div class="grp"><span class="lab">Y scale</span>'+seg('ylog',{true:"Log",false:"Linear"},String(ST.ylog))+'</div>';
  document.querySelectorAll('#controls button').forEach(b=>b.onclick=()=>{
    const g=b.dataset.grp, v=b.dataset.val; ST[g]= g==='ylog'? v==='true' : v; renderControls(); renderMain(); });
}
function renderMain(){
  document.getElementById('chart').innerHTML = chart({op:ST.op,phase:ST.phase,x:ST.x,y:ST.y,ylog:ST.ylog,
    title:OPS[ST.op]+' — '+ST.phase+' ('+YK[ST.y].toLowerCase()+' vs '+XK[ST.x].toLowerCase()+')'});
  document.getElementById('mlegend').innerHTML = legend(ST.phase);
}
function renderGrid(){
  // SEPARATE panels per (phase, EP degree): EP4 and EP8 are different communication
  // problems, never overlaid on the tokens/rank axis. (Cross-EP comparison belongs on
  // the global-tokens axis in the explorer above.)
  const phases=[...new Set(DATA.map(s=>s.phase))].sort();
  const eps=[...new Set(DATA.map(s=>s.ep))].sort((a,b)=>a-b);
  let h='';
  phases.forEach(ph=>{ eps.forEach(ep=>{
    if(!DATA.some(s=>s.phase===ph && s.ep===ep)) return;
    h+='<h2>'+ph[0].toUpperCase()+ph.slice(1)+' · EP'+ep+' — latency vs source tokens/rank (µs, log–log)</h2>'+
       legend(ph,ep)+'<div class="grid">';
    ['dispatch','combine','serial'].forEach(op=>{ h+='<div class="card"><div class="gtit">'+OPS[op]+'</div>'+
      chart({op,phase:ph,ep,x:'t',y:'lat',ylog:true,title:'',w:340,h:260})+'</div>'; });
    h+='</div>'; }); });
  document.getElementById('grid').innerHTML=h;
}
(function(){
  const s0=DATA[0]||{shape:{}}; const sh=s0.shape||{};
  const provs=[...new Set(DATA.map(s=>s.backend+' '+(s.prov.deepep_version||s.prov.mori_commit||'?')))];
  const fo=[...new Set(DATA.map(s=>(s.rows[0]&&s.rows[0].fanout!=null)?s.rows[0].fanout.toFixed(1):'?'))].join('/');
  document.getElementById('prov').textContent=
    'Fair-WORKLOAD build ('+(s0.contract||'comm-only-v1')+'): one DETERMINISTIC shared routing trace '+
    '(seed-fixed, '+(sh.routing||'?')+', identical on every SKU; mean fan-out ≈'+fo+' dest-ranks/token) — '+
    'only source tokens/rank varies along a line. Fixed: hidden='+(sh.hidden||'?')+', top-k='+(sh.topk||'?')+
    ', experts='+(sh.experts||'?')+', '+(sh.dispatch_dtype||'?')+' dispatch. Dispatch & combine timed SEPARATELY '+
    'as pure comm (staging untimed); SERIAL = their sum (not an independently-measured chained op). '+
    'Latency = median over iterations of per-iteration cross-rank max. SELECTED STACK '+provs.join(', ')+
    ' at each backend’s DEFAULT resource budget (NOT resource-normalized / not best-available V2/auto-tuned). '+
    'EP degrees in separate panels. Hover for fan-out / recv / tokens-s.';
  renderControls(); renderMain(); renderGrid();
})();
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX EP HTML plotter")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/plots/collectivex_ep.html")
    args = ap.parse_args()

    series = load_series(args.results_dir)
    if not series:
        print(f"no family=moe results with rows under {args.results_dir}")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    html = HEAD + '<div class="controls" id="controls"></div>' \
        + '<div class="card"><div id="chart"></div></div><div id="mlegend"></div>' \
        + '<div id="grid"></div>' \
        + '<p class="note">Self-contained (inline SVG, no external scripts). Generated from ' \
        + f'{len(series)} EP sweeps. Bandwidth = total routed payload across ranks ÷ latency ' \
        + '(payload-only, round-trip ≈ 2×); latency is the primary metric. Resource budgets are ' \
        + 'each backend&#39;s default (not yet normalized) — see provenance.</p>' \
        + "<script>\nconst DATA = " + json.dumps(series) + ";\n" + JS + "\n</script>\n" + TAIL
    with open(args.out, "w") as fh:
        fh.write(html)
    phases = sorted({s["phase"] for s in series})
    print(f"wrote {args.out}  ({len(series)} series across SKUs={sorted({s['sku'] for s in series})}, phases={phases})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
