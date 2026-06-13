function openModal(id) {
  const modal = document.getElementById(id);
  const content = modal.querySelector('.modal-content');
  if (!content.querySelector('.modal-header')) {
    const h1 = content.querySelector('h1');
    const closeBtn = content.querySelector('.close');
    const header = document.createElement('div');
    header.className = 'modal-header';
    if (h1) header.appendChild(h1);
    if (closeBtn) header.appendChild(closeBtn);
    content.insertBefore(header, content.firstChild);
    const body = document.createElement('div');
    body.className = 'modal-body';
    while (content.firstChild && content.firstChild !== header) body.appendChild(content.firstChild);
    while (header.nextSibling) body.appendChild(header.nextSibling);
    content.appendChild(body);
  }
  modal.style.display = "block";
  document.body.style.overflow = "hidden";
}

function closeModal(id) {
  document.getElementById(id).style.display = "none";
  document.body.style.overflow = "";
}

window.onclick = function (event) {
  document.querySelectorAll('.modal').forEach(modal => {
    if (event.target === modal) {
      modal.style.display = "none";
      document.body.style.overflow = "";
    }
  });
};

function toggleMenu() {
  document.getElementById('nav-menu').classList.toggle('show');
}

function filterAllProjects(value) {
  value = value.toLowerCase();
  document.querySelectorAll(".project-category").forEach(category => {
    const collapsed = category.dataset.collapsed === 'true';
    const hasImportant = category.querySelectorAll(".card.important").length > 0;
    let hasVisible = false;
    category.querySelectorAll(".card").forEach(card => {
      const title = card.querySelector("h3").textContent.toLowerCase();
      const tags = Array.from(card.querySelectorAll(".tags span")).map(t => t.textContent.toLowerCase());
      const match = title.includes(value) || tags.some(t => t.includes(value));
      const show = match && (!collapsed || !hasImportant || card.classList.contains('important'));
      card.style.display = show ? "block" : "none";
      if (show) hasVisible = true;
    });
    category.style.display = hasVisible ? "block" : "none";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  filterAllProjects("");

  // Inject show more/less buttons next to each category title
  document.querySelectorAll(".project-category").forEach(category => {
    const hasImportant = category.querySelectorAll(".card.important").length > 0;
    const title = category.querySelector('.category-title');
    const wrapper = document.createElement('div');
    wrapper.className = 'category-header';
    title.parentNode.insertBefore(wrapper, title);
    wrapper.appendChild(title);
    if (!hasImportant) return; // no button, all cards visible
    category.dataset.collapsed = 'true';
    const btn = document.createElement('button');
    btn.className = 'btn-toggle';
    btn.textContent = 'Show More';
    btn.addEventListener('click', () => {
      const collapsed = category.dataset.collapsed === 'true';
      category.dataset.collapsed = collapsed ? 'false' : 'true';
      btn.textContent = collapsed ? 'Show Less' : 'Show More';
      filterAllProjects(document.querySelector('.project-search input')?.value || '');
    });
    wrapper.appendChild(btn);
  });

  filterAllProjects("");

  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  // TurtleBot + 2-link arm tracing animation
  const canvas = document.getElementById('arm-canvas');
  if (canvas && typeof opentype !== 'undefined') {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    const SPEED       = 4;    // writing speed
    const SPEED_DRIVE = 0.5;  // entry/exit drive speed
    const WAIT_MS     = 5000; // pause between exit and re-entry (ms)
    const FONT_SIZE = 48;
    const MARGIN_X  = 20;
    const BASELINE  = 80;
    const BOT_CY    = BASELINE + 80;
    const BOT_W = 44, BOT_H = 50;
    const PLATE_H = 6;
    const WHEEL_R = 14, WHEEL_W = 7;
    const L1 = 70, L2 = 60;

    function ik(bx, by, tx, ty) {
      const dx = tx - bx, dy = ty - by;
      const d = Math.min(Math.hypot(dx, dy), L1 + L2 - 0.5);
      const cosA2 = (d*d - L1*L1 - L2*L2) / (2*L1*L2);
      const a2 = -Math.acos(Math.max(-1, Math.min(1, cosA2)));
      const a1 = Math.atan2(dy, dx) - Math.atan2(L2*Math.sin(a2), L1 + L2*Math.cos(a2));
      return {
        elbow: { x: bx + Math.cos(a1)*L1, y: by + Math.sin(a1)*L1 },
        tip:   { x: bx + Math.cos(a1)*L1 + Math.cos(a1+a2)*L2, y: by + Math.sin(a1)*L1 + Math.sin(a1+a2)*L2 }
      };
    }

    function drawBot(bx) {
      const cx = bx, cy = BOT_CY;
      ctx.save();
      [-1, 1].forEach(side => {
        const wx = cx + side * (BOT_W/2 + WHEEL_W*0.3), wy = cy + 10;
        ctx.beginPath(); ctx.ellipse(wx, wy, WHEEL_W, WHEEL_R, 0, 0, Math.PI*2);
        ctx.fillStyle = '#222'; ctx.strokeStyle = '#555'; ctx.lineWidth = 2; ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.arc(wx, wy, 4, 0, Math.PI*2); ctx.fillStyle = '#00c8f0'; ctx.fill();
      });
      [cy - 14, cy, cy + 14].forEach((py, i) => {
        ctx.shadowColor = '#00c8f0'; ctx.shadowBlur = i === 1 ? 10 : 4;
        ctx.fillStyle = i === 1 ? '#1a3a5a' : '#162a3a';
        ctx.strokeStyle = '#00c8f0'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.roundRect(cx - BOT_W/2, py - PLATE_H/2, BOT_W, PLATE_H, 2); ctx.fill(); ctx.stroke();
      });
      ctx.shadowBlur = 0;
      [-12, 12].forEach(ox => { ctx.fillStyle = '#333'; ctx.fillRect(cx + ox - 2, cy - 14, 4, 28); });
      ctx.shadowColor = '#0af'; ctx.shadowBlur = 12;
      ctx.fillStyle = '#0a1a2a'; ctx.strokeStyle = '#0af'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.ellipse(cx, cy - 22, 10, 7, 0, 0, Math.PI*2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#0af'; ctx.beginPath(); ctx.arc(cx + 6, cy - 22, 2.5, 0, Math.PI*2); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = '#555'; ctx.lineWidth = 3; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(cx, cy - BOT_H/2); ctx.lineTo(cx, cy - BOT_H/2 - 8); ctx.stroke();
      ctx.restore();
    }

    function drawArm(bx, tx, ty) {
      const base = { x: bx, y: BOT_CY - BOT_H/2 - 8 };
      const { elbow, tip } = ik(base.x, base.y, tx, ty);
      [[base, elbow, '#0af', 9], [elbow, tip, '#06b', 6]].forEach(([a, b, color, lw]) => {
        const g = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
        g.addColorStop(0, color); g.addColorStop(1, '#049');
        ctx.save();
        ctx.strokeStyle = g; ctx.lineWidth = lw; ctx.lineCap = 'round';
        ctx.shadowColor = '#00c8f0'; ctx.shadowBlur = 8;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = 2; ctx.shadowBlur = 0;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.restore();
      });
      [base, elbow].forEach((j, i) => {
        const r = i === 0 ? 7 : 5;
        ctx.save(); ctx.shadowColor = '#00c8f0'; ctx.shadowBlur = 10;
        ctx.beginPath(); ctx.arc(j.x, j.y, r, 0, Math.PI*2);
        ctx.fillStyle = '#1a1a2e'; ctx.fill(); ctx.strokeStyle = '#00c8f0'; ctx.lineWidth = 2; ctx.stroke();
        ctx.restore();
        ctx.beginPath(); ctx.arc(j.x, j.y, r*0.4, 0, Math.PI*2); ctx.fillStyle = '#00c8f0'; ctx.fill();
      });
      const angle = Math.atan2(tip.y - elbow.y, tip.x - elbow.x);
      ctx.save(); ctx.translate(tip.x, tip.y); ctx.rotate(angle);
      ctx.fillStyle = '#ddd'; ctx.beginPath(); ctx.roundRect(-14, -3, 14, 6, 2); ctx.fill();
      ctx.fillStyle = '#00c8f0'; ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(-5,-2.5); ctx.lineTo(-5,2.5); ctx.closePath(); ctx.fill();
      ctx.restore();
    }

    function samplePath(otPath) {
      const pts = [], N = 12;
      let cx = 0, cy = 0, sx = 0, sy = 0;
      for (const cmd of otPath.commands) {
        if (cmd.type === 'M') { cx = cmd.x; cy = cmd.y; sx = cx; sy = cy; pts.push({ x: cx, y: cy, move: true }); }
        else if (cmd.type === 'L') { pts.push({ x: cmd.x, y: cmd.y }); cx = cmd.x; cy = cmd.y; }
        else if (cmd.type === 'C') {
          for (let i = 1; i <= N; i++) { const t=i/N,mt=1-t; pts.push({ x:mt*mt*mt*cx+3*mt*mt*t*cmd.x1+3*mt*t*t*cmd.x2+t*t*t*cmd.x, y:mt*mt*mt*cy+3*mt*mt*t*cmd.y1+3*mt*t*t*cmd.y2+t*t*t*cmd.y }); }
          cx = cmd.x; cy = cmd.y;
        } else if (cmd.type === 'Q') {
          for (let i = 1; i <= N; i++) { const t=i/N,mt=1-t; pts.push({ x:mt*mt*cx+2*mt*t*cmd.x1+t*t*cmd.x, y:mt*mt*cy+2*mt*t*cmd.y1+t*t*cmd.y }); }
          cx = cmd.x; cy = cmd.y;
        } else if (cmd.type === 'Z') { pts.push({ x: sx, y: sy }); cx = sx; cy = sy; }
      }
      return pts;
    }

    const TEXT = 'Robotics Engineer • Innovator • Problem Solver';

    // Zhang-Suen thinning: reduces binary image to 1px skeleton
    function zhangSuen(grid, w, h) {
      const g = grid.slice();
      let changed = true;
      while (changed) {
        changed = false;
        for (let pass = 0; pass < 2; pass++) {
          const toRemove = [];
          for (let y = 1; y < h-1; y++) for (let x = 1; x < w-1; x++) {
            if (!g[y*w+x]) continue;
            const [p2,p3,p4,p5,p6,p7,p8,p9] = [
              g[(y-1)*w+x],g[(y-1)*w+x+1],g[y*w+x+1],g[(y+1)*w+x+1],
              g[(y+1)*w+x],g[(y+1)*w+x-1],g[y*w+x-1],g[(y-1)*w+x-1]
            ];
            const nb = p2+p3+p4+p5+p6+p7+p8+p9;
            if (nb < 2 || nb > 6) continue;
            const seq = [p2,p3,p4,p5,p6,p7,p8,p9,p2];
            let trans = 0; for (let i=0;i<8;i++) if (!seq[i]&&seq[i+1]) trans++;
            if (trans !== 1) continue;
            if (pass===0 && (p2*p4*p6)!==0) continue;
            if (pass===0 && (p4*p6*p8)!==0) continue;
            if (pass===1 && (p2*p4*p8)!==0) continue;
            if (pass===1 && (p2*p6*p8)!==0) continue;
            toRemove.push(y*w+x);
          }
          for (const i of toRemove) { g[i] = 0; changed = true; }
        }
      }
      return g;
    }

    // Trace skeleton pixels into ordered polyline segments
    function traceSkeleton(grid, w, h) {
      const visited = new Uint8Array(grid.length);
      const dirs = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];
      const segments = [];

      const nbCount = (x, y) => dirs.filter(([dy,dx]) => {
        const nx=x+dx, ny=y+dy;
        return nx>=0&&ny>=0&&nx<w&&ny<h&&grid[ny*w+nx];
      }).length;

      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        if (!grid[y*w+x] || visited[y*w+x]) continue;
        // Start segments from endpoints (1 neighbor) or junctions (3+), else any unvisited
        const nc = nbCount(x, y);
        if (nc === 0) { visited[y*w+x] = 1; continue; }
        // Walk each unvisited neighbor as a separate segment
        const startNeighbors = dirs.map(([dy,dx])=>({nx:x+dx,ny:y+dy}))
          .filter(({nx,ny})=>nx>=0&&ny>=0&&nx<w&&ny<h&&grid[ny*w+nx]&&!visited[ny*w+nx]);
        if (startNeighbors.length === 0) { visited[y*w+x] = 1; continue; }
        visited[y*w+x] = 1;
        for (const {nx: sx, ny: sy} of startNeighbors) {
          if (visited[sy*w+sx]) continue;
          const seg = [{x,y}, {x:sx,y:sy}];
          visited[sy*w+sx] = 1;
          let cx = sx, cy = sy;
          while (true) {
            const next = dirs.map(([dy,dx])=>({nx:cx+dx,ny:cy+dy}))
              .find(({nx,ny})=>nx>=0&&ny>=0&&nx<w&&ny<h&&grid[ny*w+nx]&&!visited[ny*w+nx]);
            if (!next) break;
            visited[next.ny*w+next.nx] = 1;
            seg.push({x:next.nx, y:next.ny});
            cx = next.nx; cy = next.ny;
          }
          if (seg.length >= 1) segments.push(seg);
        }
      }
      return segments;
    }

    // Rasterize a glyph and extract its skeleton as canvas-space points
    function glyphSkeleton(font, glyphIndex, glyphX, scale) {
      const glyph = font.glyphs.get(glyphIndex);
      const renderSize = FONT_SIZE * 4; // render 4x for better skeleton quality
      const pad = 6;
      const gc = document.createElement('canvas');
      gc.width = Math.ceil(glyph.advanceWidth * renderSize / font.unitsPerEm) + pad * 2;
      gc.height = renderSize + pad * 2;
      const gctx = gc.getContext('2d');
      const baseline = renderSize * 0.8;
      const path2d = new Path2D();
      const otPath = glyph.getPath(pad, baseline, renderSize);
      otPath.commands.forEach(cmd => {
        if (cmd.type==='M') path2d.moveTo(cmd.x, cmd.y);
        else if (cmd.type==='L') path2d.lineTo(cmd.x, cmd.y);
        else if (cmd.type==='C') path2d.bezierCurveTo(cmd.x1,cmd.y1,cmd.x2,cmd.y2,cmd.x,cmd.y);
        else if (cmd.type==='Q') path2d.quadraticCurveTo(cmd.x1,cmd.y1,cmd.x,cmd.y);
        else if (cmd.type==='Z') path2d.closePath();
      });
      gctx.fillStyle = '#fff';
      gctx.fill(path2d, 'evenodd');
      const w = gc.width, h = gc.height;
      const idata = gctx.getImageData(0, 0, w, h);
      const bin = new Uint8Array(w * h);
      for (let i = 0; i < bin.length; i++) bin[i] = idata.data[i*4+3] > 128 ? 1 : 0;
      const skel = zhangSuen(bin, w, h);
      const segs = traceSkeleton(skel, w, h);
      const advance = glyph.advanceWidth * FONT_SIZE / font.unitsPerEm;
      // Scale skeleton coords from renderSize space back to FONT_SIZE space
      const scaleF = FONT_SIZE / renderSize;
      const mapped = segs.map(seg => seg.map(p => ({
        x: glyphX + (p.x - pad) * scaleF,
        y: BASELINE + (p.y - baseline) * scaleF
      })));
      if (mapped.length <= 1) return { segments: mapped, advance };
      // Sort longest first, each segment left-to-right
      mapped.sort((a, b) => {
        const ax = Math.min(a[0].x, a.at(-1).x);
        const bx = Math.min(b[0].x, b.at(-1).x);
        if (Math.abs(ax - bx) > 3) return ax - bx; // left strokes first
        return b.length - a.length; // tiebreak by length
      });
      mapped.forEach(seg => {
        const dx = Math.abs(seg.at(-1).x - seg[0].x);
        const dy = Math.abs(seg.at(-1).y - seg[0].y);
        if (dx >= dy) { if (seg[0].x > seg.at(-1).x) seg.reverse(); }   // horizontal: left→right
        else          { if (seg[0].y > seg.at(-1).y) seg.reverse(); }   // vertical: top→bottom
      });
      return { segments: mapped, advance };
    }

    // Load Inter 400 for opentype skeleton extraction
    const otUrl = 'https://cdn.jsdelivr.net/npm/@fontsource/inter/files/inter-latin-400-normal.woff';
    opentype.load(otUrl, (err, font) => {
      if (err) { console.error('Font load failed:', err); return; }

      const totalW = font.getAdvanceWidth(TEXT, FONT_SIZE);
      const textOffsetX = (W - totalW) / 2;

      // Extract skeleton paths for each glyph
      const glyphSkels = [];
      const glyphFillPaths = [];
      let curX = textOffsetX;
      const glyphChars = [...TEXT];
      for (let gi = 0; gi < font.stringToGlyphs(TEXT).length; gi++) {
        const glyph = font.stringToGlyphs(TEXT)[gi];
        const advance = glyph.advanceWidth * FONT_SIZE / font.unitsPerEm;
        let segments;
        if (glyphChars[gi] === '•') {
          // Bullet: inject a small circle path (8 points around center)
          const cx2 = curX + advance / 2, cy2 = BASELINE - FONT_SIZE * 0.35, r = 3;
          const circlePts = Array.from({length: 9}, (_, i) => ({
            x: cx2 + r * Math.cos(i / 8 * Math.PI * 2 - Math.PI/2),
            y: cy2 + r * Math.sin(i / 8 * Math.PI * 2 - Math.PI/2)
          }));
          segments = [circlePts];
        } else {
          const skel = glyphSkeleton(font, glyph.index, curX, FONT_SIZE);
          segments = skel.segments.length > 0 ? skel.segments
            : [[{ x: curX + advance / 2, y: BASELINE - FONT_SIZE * 0.3 }]];
          // For i/j: append a small dot circle for the tittle
          if (glyphChars[gi] === 'i' || glyphChars[gi] === 'j') {
            const cx2 = curX + advance / 2, cy2 = BASELINE - FONT_SIZE * 0.65, r = 1.5;
            segments = [...segments, Array.from({length: 9}, (_, k) => ({
              x: cx2 + r * Math.cos(k / 8 * Math.PI * 2 - Math.PI/2),
              y: cy2 + r * Math.sin(k / 8 * Math.PI * 2 - Math.PI/2)
            }))];
          }
        }
        glyphSkels.push(segments);
        // Pre-build filled Path2D for each glyph
        const fp = new Path2D();
        const otPath = glyph.getPath(curX, BASELINE, FONT_SIZE);
        for (const cmd of otPath.commands) {
          if (cmd.type==='M') fp.moveTo(cmd.x, cmd.y);
          else if (cmd.type==='L') fp.lineTo(cmd.x, cmd.y);
          else if (cmd.type==='C') fp.bezierCurveTo(cmd.x1,cmd.y1,cmd.x2,cmd.y2,cmd.x,cmd.y);
          else if (cmd.type==='Q') fp.quadraticCurveTo(cmd.x1,cmd.y1,cmd.x,cmd.y);
          else if (cmd.type==='Z') fp.closePath();
        }
        glyphFillPaths.push(fp);
        curX = textOffsetX + font.getAdvanceWidth(TEXT.slice(0, glyphSkels.length), FONT_SIZE);      }

      // Flatten skeleton segments into allPts with glyph index
      const allPts = [];
      for (let gi = 0; gi < glyphSkels.length; gi++) {
        for (const seg of glyphSkels[gi]) {
          if (seg.length === 1) {
            // Single isolated pixel — mark as dot (move=false, so it draws a tiny stroke to itself)
            allPts.push({ x: seg[0].x, y: seg[0].y, move: true, gi });
            allPts.push({ x: seg[0].x + 0.5, y: seg[0].y, move: false, gi });
          } else {
            for (let i = 0; i < seg.length; i++) allPts.push({ x: seg[i].x, y: seg[i].y, move: i === 0, gi });
          }
        }
      }

      const textCanvas = document.createElement('canvas');
      textCanvas.width = W; textCanvas.height = H;
      const tctx = textCanvas.getContext('2d');

      const firstPt = allPts[0] ?? { x: textOffsetX, y: BASELINE };
      let state = 'enter', pi = 0, penDown = false;
      let armX = firstPt.x, armY = firstPt.y, botX = -BOT_W;
      let lastX = armX, lastY = armY;
      let fadeAlpha = 1, fadeStart = null;
      let doneGlyphs = new Set();

      const ENTER_TARGET = firstPt.x, EXIT_TARGET = W + BOT_W;
      const ffBtn = document.getElementById('arm-ff-btn');

      function redraw() {
        tctx.clearRect(0, 0, W, H);
        tctx.fillStyle = '#eee';
        for (const gi of doneGlyphs) tctx.fill(glyphFillPaths[gi], 'evenodd');
      }

      function showDone() {
        tctx.clearRect(0, 0, W, H);
        tctx.fillStyle = '#eee';
        for (const p of glyphFillPaths) tctx.fill(p, 'evenodd');
        ctx.clearRect(0, 0, W, H); ctx.drawImage(textCanvas, 0, 0);
        state = 'done'; ffBtn.textContent = '▶';
      }

      function reset() {
        tctx.clearRect(0, 0, W, H); pi = 0; penDown = false; doneGlyphs = new Set();
        armX = firstPt.x; armY = firstPt.y; botX = -BOT_W;
        lastX = armX; lastY = armY; fadeAlpha = 1; fadeStart = null; state = 'enter';
        ffBtn.textContent = '▶▶'; cancelAnimationFrame(rafId); rafId = requestAnimationFrame(frame);
      }

      ffBtn.addEventListener('click', () => { if (state === 'done') reset(); else showDone(); });

      function frame(now) {
        if (state === 'done') return;

        if (state === 'enter') {
          botX += (ENTER_TARGET - botX) * 0.05;
          armX = botX; armY = firstPt.y;
          if (Math.abs(botX - ENTER_TARGET) < 2) {
            botX = ENTER_TARGET; armX = firstPt.x; armY = firstPt.y;
            lastX = armX; lastY = armY; penDown = false;
            state = 'write';
          }

        } else if (state === 'write') {
          let rem = SPEED;
          while (rem > 0 && pi < allPts.length) {
            const t = allPts[pi];
            const dx = t.x - armX, dy = t.y - armY, dist = Math.hypot(dx, dy);
            const step = Math.min(dist, rem);
            const r = dist > 0 ? step / dist : 0;
            armX += dx * r; armY += dy * r;
            if (!t.move && penDown) {
              tctx.beginPath(); tctx.moveTo(lastX, lastY); tctx.lineTo(armX, armY);
              tctx.strokeStyle = '#eee'; tctx.lineWidth = 2.5; tctx.lineCap = 'round'; tctx.stroke();
            }
            lastX = armX; lastY = armY;
            if (step >= dist) {
              const prevGi = allPts[pi].gi; penDown = !t.move; pi++;
              const nextGi = allPts[pi]?.gi;
              if (nextGi !== undefined && nextGi !== prevGi && !doneGlyphs.has(prevGi)) {
                doneGlyphs.add(prevGi); // redraw();
              }
            }
            rem -= step; if (step === 0) break;
          }
          botX += (armX - botX) * 0.03;
          if (pi >= allPts.length) {
            // doneGlyphs = new Set(glyphFillPaths.map((_,i)=>i)); redraw();
            state = 'exit'; fadeStart = null;
          }

        } else if (state === 'exit') {
          botX += (EXIT_TARGET - botX) * 0.05;
          armX += (EXIT_TARGET - armX) * 0.05;
          if (!fadeStart) fadeStart = now;
          const held = now - fadeStart;
          if (held > 3000) fadeAlpha = Math.max(0, 1 - (held - 3000) / 600);
          if (botX > W + BOT_W - 5) { setTimeout(() => { if (state === 'waiting') reset(); }, WAIT_MS); state = 'waiting'; }
        }

        ctx.clearRect(0, 0, W, H);
        ctx.save(); ctx.globalAlpha = fadeAlpha; ctx.drawImage(textCanvas, 0, 0); ctx.restore();
        if (botX > -BOT_W && botX < W + BOT_W) {
          drawBot(botX);
          if (state === 'write' || state === 'enter') drawArm(botX, armX, armY);
        }
        rafId = requestAnimationFrame(frame);
      }

      let rafId = requestAnimationFrame(frame);
    });
  }

  // Escape key closes modals
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal').forEach(m => { m.style.display = 'none'; });
      document.body.style.overflow = '';
    }
  });

  // Scroll-reveal
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });
  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

  // Active nav highlighting
  const navLinks = document.querySelectorAll('.nav-right a[href^="#"]');
  const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        navLinks.forEach(a => a.classList.remove('active'));
        const link = document.querySelector(`.nav-right a[href="#${e.target.id}"]`);
        if (link) link.classList.add('active');
      }
    });
  }, { rootMargin: '-20% 0px -75% 0px', threshold: 0 });
  document.querySelectorAll('section[id]').forEach(s => sectionObserver.observe(s));

  // Back-to-top button
  const btn = document.getElementById('back-to-top');
  const nav = document.querySelector('nav');
  window.addEventListener('scroll', () => {
    if (btn) btn.classList.toggle('visible', window.scrollY > 400);
    if (nav) nav.classList.toggle('scrolled', window.scrollY > 80);
  });
});
