"""Gráficas D3 (v7, embebida localmente) para la interfaz Streamlit."""
import json
from pathlib import Path

D3_JS = (Path(__file__).parent / "static" / "d3.min.js").read_text(encoding="utf-8")

PALETTE = {"bg": "#FBF9F5", "ink": "#1E2226", "muted": "#7A7F87", "grid": "#E7E2DA",
           "accent": "#0E6E66", "error": "#B4532A", "bar": "#D9D3C9",
           "x": "#0E6E66", "y": "#C08A1E", "z": "#4A5A73"}

_BASE_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Instrument+Sans:wght@400;500;600&display=swap');
  html,body{{margin:0;background:transparent;font-family:'Instrument Sans',system-ui,sans-serif;color:{PALETTE['ink']};}}
  .tick text{{font-family:'IBM Plex Mono',monospace;font-size:10px;fill:{PALETTE['muted']};}}
  .tick line{{stroke:{PALETTE['grid']};}} .domain{{display:none;}}
  .lbl{{font-size:11px;fill:{PALETTE['muted']};letter-spacing:.04em;text-transform:uppercase;}}
  .tip{{position:absolute;pointer-events:none;background:{PALETTE['ink']};color:#fff;padding:6px 9px;border-radius:6px;
        font:11px 'IBM Plex Mono',monospace;opacity:0;transition:opacity .12s;white-space:nowrap;}}
  .legend{{display:flex;gap:16px;font-size:12px;color:{PALETTE['muted']};margin:0 0 6px 58px;}}
  .legend i{{display:inline-block;width:14px;height:2px;margin-right:6px;vertical-align:middle;}}
</style>
<script>{D3_JS}</script>
"""


def signals_html(x, fs=50, width=760):
    """3 paneles (body_acc, body_gyro, total_acc) con eje x compartido y cursor de inspección."""
    data = json.dumps([[round(float(v), 5) for v in row] for row in x])
    panels = json.dumps([{"name": "Aceleración corporal", "unit": "g", "off": 0},
                         {"name": "Velocidad angular", "unit": "rad/s", "off": 3},
                         {"name": "Aceleración total", "unit": "g", "off": 6}])
    return _BASE_CSS + f"""
<div class="legend"><span><i style="background:{PALETTE['x']}"></i>eje x</span>
<span><i style="background:{PALETTE['y']}"></i>eje y</span><span><i style="background:{PALETTE['z']}"></i>eje z</span></div>
<div id="sig" style="position:relative"></div>
<script>
const X={data}, P={panels}, fs={fs}, W=Math.max(420,document.body.clientWidth||{width}), H=122, M={{l:58,r:14,t:20,b:6}}, C=["{PALETTE['x']}","{PALETTE['y']}","{PALETTE['z']}"];
const n=X.length, t=d3.range(n).map(i=>i/fs);
const root=d3.select("#sig"), tip=root.append("div").attr("class","tip");
const svg=root.append("svg").attr("width",W).attr("height",P.length*H+26);
const xs=d3.scaleLinear().domain([0,t[n-1]]).range([M.l,W-M.r]);
const panels=[];
P.forEach((p,k)=>{{
  const g=svg.append("g").attr("transform",`translate(0,${{k*H}})`);
  const vals=X.flatMap(r=>[r[p.off],r[p.off+1],r[p.off+2]]);
  const pad=(d3.max(vals)-d3.min(vals))*0.08||0.01;
  const ys=d3.scaleLinear().domain([d3.min(vals)-pad,d3.max(vals)+pad]).range([H-M.b,M.t]).nice();
  g.append("g").attr("transform",`translate(${{M.l}},0)`).call(d3.axisLeft(ys).ticks(4).tickSize(-(W-M.l-M.r)).tickPadding(8));
  const tx=g.append("text").attr("class","lbl").attr("x",M.l).attr("y",12).text(p.name);
  tx.append("tspan").attr("dx",8).style("text-transform","none").style("letter-spacing","0").text("["+p.unit+"]");
  [0,1,2].forEach(j=>{{
    const line=d3.line().x((d,i)=>xs(t[i])).y(d=>ys(d[p.off+j])).curve(d3.curveMonotoneX);
    const path=g.append("path").datum(X).attr("fill","none").attr("stroke",C[j]).attr("stroke-width",1.4)
      .attr("stroke-linejoin","round").attr("opacity",.92).attr("d",line);
    const L=path.node().getTotalLength();
    path.attr("stroke-dasharray",L).attr("stroke-dashoffset",L).transition().delay(k*120+j*60).duration(700)
      .ease(d3.easeCubicOut).attr("stroke-dashoffset",0);
  }});
  panels.push({{g,ys,p}});
}});
svg.append("g").attr("transform",`translate(0,${{P.length*H}})`).call(d3.axisBottom(xs).ticks(6).tickFormat(d=>d+" s").tickSize(4));
const cursor=svg.append("line").attr("y1",M.t).attr("y2",P.length*H).attr("stroke","{PALETTE['ink']}").attr("stroke-width",.6).attr("opacity",0);
svg.append("rect").attr("x",M.l).attr("y",0).attr("width",W-M.l-M.r).attr("height",P.length*H).attr("fill","transparent")
 .on("mousemove",ev=>{{
   const [mx,my]=d3.pointer(ev), i=Math.max(0,Math.min(n-1,Math.round(xs.invert(mx)*fs)));
   const k=Math.max(0,Math.min(P.length-1,Math.floor(my/H))), p=P[k];
   cursor.attr("x1",xs(t[i])).attr("x2",xs(t[i])).attr("opacity",.5);
   const v=[0,1,2].map(j=>X[i][p.off+j].toFixed(3));
   const sc=1;
   tip.style("opacity",1).style("left",(xs(t[i])*sc+12)+"px").style("top",(k*H*sc+8)+"px")
      .html(`t = ${{t[i].toFixed(2)}} s<br>x ${{v[0]}} · y ${{v[1]}} · z ${{v[2]}} ${{p.unit}}`);
 }}).on("mouseleave",()=>{{cursor.attr("opacity",0);tip.style("opacity",0);}});
</script>"""


def proba_html(labels, proba, pred, true_idx=None, width=420):
    rows = sorted([{"l": l, "p": float(p), "i": i} for i, (l, p) in enumerate(zip(labels, proba))],
                  key=lambda r: -r["p"])
    return _BASE_CSS + f"""
<div id="pb"></div>
<script>
const R={json.dumps(rows)}, pred={pred}, tru={json.dumps(true_idx)}, W=Math.max(320,document.body.clientWidth||{width}), rh=34, M={{l:132,r:62}};
const svg=d3.select("#pb").append("svg").attr("width",W).attr("height",R.length*rh+6);
const xs=d3.scaleLinear().domain([0,1]).range([M.l,W-M.r]);
const g=svg.selectAll("g").data(R).join("g").attr("transform",(d,k)=>`translate(0,${{k*rh+4}})`);
g.append("rect").attr("x",M.l).attr("y",7).attr("height",14).attr("rx",7).attr("width",W-M.l-M.r).attr("fill","{PALETTE['grid']}").attr("opacity",.55);
g.append("rect").attr("x",M.l).attr("y",7).attr("height",14).attr("rx",7).attr("width",0)
  .attr("fill",d=>d.i===pred?(tru===null||tru===pred?"{PALETTE['accent']}":"{PALETTE['error']}"):"{PALETTE['bar']}")
  .transition().delay((d,k)=>k*60).duration(650).ease(d3.easeCubicOut).attr("width",d=>Math.max(0,xs(d.p)-M.l));
g.append("text").attr("x",M.l-12).attr("y",18).attr("text-anchor","end").style("font-size","13px")
  .style("font-weight",d=>d.i===pred?600:400).style("fill",d=>d.i===pred?"{PALETTE['ink']}":"{PALETTE['muted']}")
  .text(d=>d.l+(tru!==null&&d.i===tru?"  ●":""));
g.append("text").attr("x",W-M.r+10).attr("y",18).style("font-family","'IBM Plex Mono',monospace").style("font-size","12px")
  .style("fill",d=>d.i===pred?"{PALETTE['ink']}":"{PALETTE['muted']}").text(d=>(d.p*100).toFixed(1)+"%");
</script>"""
