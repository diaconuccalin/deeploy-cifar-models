"""Render the architecture diagrams (deliverables/<model>/*.png) with headless Chrome.

The layer stacks come from figures_template.html (next to this script); the skip connections are
drawn here: each branches off the arrow entering the block and ends in the + circle.

Usage: python make_figures.py [out_dir]
"""
import os, subprocess, sys, tempfile
from PIL import Image, ImageChops

ROOT = os.path.expanduser("~/tallinn/deliverables")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_template.html")).read()

CSS = """<style>
header, footer, .specs, .notes, .net .controls, .badges { display:none !important; }
body { padding:24px !important; width:560px; background:#fff !important; }
.wrap { max-width:520px; } .nets { grid-template-columns:1fr !important; }
.net { border:0 !important; padding:0 !important; position:relative; }
.net:nth-of-type(%d) { display:none !important; }
.res::after { display:none; }
.res { padding:0 44px; }
.edge.nohead::after { display:none; } .edge.nohead::before { bottom:0; }
.edge.branch { height:46px; } .edge.branch .shape { top:13px; }
.add { position:relative; justify-self:stretch; display:block; height:30px; }
.add b { position:absolute; left:50%%; top:0; width:30px; height:30px; transform:translateX(-50%%); }
.add .txt { position:absolute; right:calc(50%% + 24px); top:5px; white-space:nowrap; }
svg.skips { position:absolute; left:0; top:0; overflow:visible; pointer-events:none; }
</style>"""

JS = """<script>
function edge(cls) { const e = document.createElement('div'); e.className = 'edge ' + cls; return e; }
async function draw() {
  await document.fonts.ready;
  document.querySelectorAll('.nc').forEach(e => e.textContent = '10 / 100');
  const jobs = [];
  document.querySelectorAll('.res').forEach(res => {
    const group = res.closest('.group');
    let branch;
    if (group.querySelector('.op, .res') === res) {       // branches at the block input
      group.style.paddingTop = 0;
      group.previousElementSibling.classList.add('nohead');
      branch = edge('branch'); res.prepend(branch);
    } else { branch = res.previousElementSibling; branch.classList.add('branch'); }
    const add = res.querySelector('.add'), b = add.querySelector('b');
    const t = document.createElement('span'); t.className = 'txt';
    t.textContent = add.textContent.replace('+', '').trim();
    add.replaceChildren(b, t);
    if (group.lastElementChild === res) {                 // block output leaves from the circle
      group.style.paddingBottom = 0;
      const tail = edge('nohead'); tail.style.height = '22px'; group.append(tail);
    }
    const c = res.classList.contains('attn') ? '--k-attn' : res.classList.contains('mlp') ? '--k-lin' : '--k-dw';
    jobs.push([res, branch, b, c]);
  });
  const net = [...document.querySelectorAll('.net')].find(n => n.offsetParent !== null);
  const o = net.getBoundingClientRect(), NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('class', 'skips');
  svg.setAttribute('width', o.width); svg.setAttribute('height', o.height);
  for (const [res, branch, b, c] of jobs) {
    if (!net.contains(res)) continue;
    const R = res.getBoundingClientRect(), E = branch.getBoundingClientRect(), B = b.getBoundingClientRect();
    const x0 = E.left + E.width / 2 - o.left, y0 = E.bottom - o.top - 18;
    const x1 = R.right - o.left - 18, y1 = B.top + B.height / 2 - o.top, xe = B.right - o.left, r = 8;
    const col = getComputedStyle(document.documentElement).getPropertyValue(c).trim();
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('d', `M${x0} ${y0} H${x1 - r} Q${x1} ${y0} ${x1} ${y0 + r} V${y1 - r} Q${x1} ${y1} ${x1 - r} ${y1} H${xe + 7}`);
    path.setAttribute('fill', 'none'); path.setAttribute('stroke', col); path.setAttribute('stroke-width', 2.25);
    const head = document.createElementNS(NS, 'path');
    head.setAttribute('d', `M${xe} ${y1} l9 -5.5 v11 z`); head.setAttribute('fill', col);
    const dot = document.createElementNS(NS, 'circle');
    dot.setAttribute('cx', x0); dot.setAttribute('cy', y0); dot.setAttribute('r', 4); dot.setAttribute('fill', col);
    svg.append(path, head, dot);
  }
  net.append(svg);
}
window.addEventListener('load', draw);
</script>"""

with tempfile.TemporaryDirectory() as tmp:
    for name, hide, sub in [("miniMobileNetV2_6x", 2, "MiniMobileNetV2"), ("CCT_2_32_32_128", 1, "CCT")]:
        page, raw = os.path.join(tmp, name + ".html"), os.path.join(tmp, name + ".png")
        open(page, "w").write('<!doctype html><html data-theme="light"><head><meta charset="utf-8">' + SRC +
                              CSS % hide + JS + "</html>")
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                        "--force-device-scale-factor=2", "--virtual-time-budget=8000", "--window-size=600,2600",
                        "--screenshot=" + raw, "file://" + page], check=True, capture_output=True)
        im = Image.open(raw).convert("RGB")
        l, t, r, b = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
        assert b < im.height, "diagram cut off"
        out = os.path.join(OUT or os.path.join(ROOT, sub), name + ".png")
        im.crop((max(0, l - 40), max(0, t - 40), min(im.width, r + 40), min(im.height, b + 40))).save(out)
        print(out)
