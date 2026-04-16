'use strict';

const CIRC = 238.76;
const FALL_T = 0.55, SZ_T = 0.48;



/* DOM shortcuts */
const $ = id => document.getElementById(id);
const vstream    = $('vstream');
const vidBox     = $('vid-box');
const camLbl     = $('cam-lbl');
const tsLbl      = $('ts-lbl');
const aiLbl      = $('ai-lbl');
const trnOver    = $('trn-overlay');
const ctrlOver   = $('ctrl-overlay');
const trnTxt     = $('trn-txt');
const segProg    = $('seg-prog');
const segTitle   = $('seg-title');
const segPill    = $('seg-pill');
const fpsTxt     = $('fps-txt');
const alertToday = $('alert-today');
const pAv        = $('p-av');
const pName      = $('p-name');
const pRoom      = $('p-room');
const pBadge     = $('p-badge');
const pStatus    = $('p-status');
const vHr        = $('v-hr');
const vSpo2      = $('v-spo2');
const vBp        = $('v-bp');
const vTemp      = $('v-temp');
const fallArc    = $('fall-arc');
const szArc      = $('sz-arc');
const fallPct    = $('fall-pct');
const szPct      = $('sz-pct');
const gemCard    = $('gem-card');
const gIdle      = $('g-idle');
const gThink     = $('g-think');
const gResult    = $('g-result');
const gVerdict   = $('g-verdict');
const gHl        = $('g-hl');
const gTxt       = $('g-txt');
const gSev       = $('g-sev');
const gActs      = $('g-acts');
const alertCount = $('alert-count');
const logFeed    = $('log-feed');
const logEmpty   = $('log-empty');
const reviewOver = $('review-overlay');
const rvBar      = $('rv-bar');
const rvCount    = $('rv-count');

/* Click-to-continue disabled — use 'Next Patient' button or navbar button */

const btnPause   = $('btn-pause');
const btnSkip    = $('btn-skip');
const sidebar    = $('sidebar');
const viHr       = $('vi-hr');
const navClock   = $('nav-clock');
const frameCount = $('frame-count');
const gemTime    = $('gem-time');
const statsDone  = $('stats-overlay');
const patPlate   = $('pat-nameplate');

/* State */
let totalAlerts = 0, isPaused = false;
let curHR = 78, hrInterval = null, clockInterval = null, hbInterval = null;
let curSegType = 'normal';
let frameTotal = 0;
let gemAlertStartTime = 0;
/* #16 — Session stats */
let sStats = { detected: 0, confirmed: 0, suppressed: 0 };
let sessionStartTime = Date.now();

// Ensure route guard is fully enforced logically as well
if (!localStorage.getItem('vg_token')) {
    window.location.href = '/static/login.html';
}

function logoutNurse() {
    localStorage.removeItem('vg_token');
    localStorage.removeItem('vg_nurse_name');
    localStorage.removeItem('vg_staff_id');
    window.location.href = '/static/login.html';
}

// Set nurse name in DOM
document.addEventListener('DOMContentLoaded', () => {
    const el = document.getElementById('nurse-name-display');
    if (el) el.textContent = localStorage.getItem('vg_nurse_name') || 'Staff';
    
    // RBAC: Hide Admin Hub if not admin
    const staffId = localStorage.getItem('vg_staff_id');
    const adminBtn = document.getElementById('admin-hub-btn');
    if (adminBtn && staffId !== 'admin') {
        adminBtn.style.display = 'none';
    }
});

/* ── #18 Fall Gauge Simulation ────────────────────────────────────────
   Mirrors SIM for seizure. Ramps fall gauge smoothly on fall segments.
───────────────────────────────────────────────────────────────────── */
const FSIM = {
  val: 0, target: 0, active: false, spiked: false, _tid: null,
};

function _fallTick() {
  if (!FSIM.active) return;
  const noise = (Math.random() - 0.5) * (FSIM.spiked ? 1.5 : 2.5);
  FSIM.val += (FSIM.target - FSIM.val) * (FSIM.spiked ? 0.14 : 0.055) + noise * 0.25;
  FSIM.val  = Math.max(0, Math.min(100, FSIM.val));
  if (!FSIM.spiked) {
    FSIM.target += (Math.random() - 0.5) * 1.4;
    FSIM.target  = Math.max(30, Math.min(52, FSIM.target));
  } else {
    FSIM.target += (Math.random() - 0.5) * 0.9;
    FSIM.target  = Math.max(84, Math.min(97, FSIM.target));
  }
  _renderFall(FSIM.val / 100);
  FSIM._tid = setTimeout(_fallTick, 110);
}

function fallSimStart() {
  if (FSIM._tid) clearTimeout(FSIM._tid);
  FSIM.val    = 0;
  FSIM.target = 34 + Math.random() * 14;
  FSIM.active = true;
  FSIM.spiked = false;
  FSIM._tid   = setTimeout(_fallTick, 110);
}

function fallSimSpike() {
  FSIM.spiked = true;
  FSIM.target = 87 + Math.random() * 10;
}

function fallSimStop() {
  FSIM.active = false; FSIM.spiked = false;
  if (FSIM._tid) { clearTimeout(FSIM._tid); FSIM._tid = null; }
  const d = setInterval(() => {
    FSIM.val = Math.max(0, FSIM.val - 2.5);
    _renderFall(FSIM.val / 100);
    if (FSIM.val <= 0) clearInterval(d);
  }, 80);
}

/* #8 — Gradient color for fall arc based on value */
function _fallColor(v) {
  if (v < 0.35) return '#f59e0b';            // amber
  if (v < 0.55) return '#f97316';            // orange
  return '#ef4444';                          // red
}

function _renderFall(v) {
  const color = _fallColor(v);
  fallArc.style.stroke       = color;
  fallArc.style.strokeDashoffset = CIRC * (1 - v);
  /* #23 — Smooth number */
  _animNum(fallPct, parseInt(fallPct.textContent) || 0, Math.round(v * 100), '%');
  fallArc.classList.toggle('hot', v >= FALL_T);
  fallPct.style.color = color;
}

/* ── Seizure Gauge Simulation ────────────────────────── */
const SIM = {
  val: 0, target: 0, active: false, spiked: false, _tid: null,
};

function _szTick() {
  if (!SIM.active) return;
  const noise = (Math.random() - 0.5) * (SIM.spiked ? 1.5 : 2.2);
  SIM.val += (SIM.target - SIM.val) * (SIM.spiked ? 0.14 : 0.055) + noise * 0.25;
  SIM.val  = Math.max(0, Math.min(100, SIM.val));
  if (!SIM.spiked) {
    SIM.target += (Math.random() - 0.5) * 1.2;
    SIM.target  = Math.max(28, Math.min(46, SIM.target));
  } else {
    SIM.target += (Math.random() - 0.5) * 0.8;
    SIM.target  = Math.max(82, Math.min(96, SIM.target));
  }
  _renderSz(SIM.val / 100);
  SIM._tid = setTimeout(_szTick, 110);
}

function szSimStart() {
  if (SIM._tid) clearTimeout(SIM._tid);
  SIM.val    = 0;
  SIM.target = 32 + Math.random() * 12;
  SIM.active = true;
  SIM.spiked = false;
  SIM._tid   = setTimeout(_szTick, 110);
}

function szSimSpike() {
  SIM.spiked = true;
  SIM.target = 88 + Math.random() * 8;
}

function szSimStop() {
  SIM.active = false; SIM.spiked = false;
  if (SIM._tid) { clearTimeout(SIM._tid); SIM._tid = null; }
  const d = setInterval(() => {
    SIM.val = Math.max(0, SIM.val - 2.5);
    _renderSz(SIM.val / 100);
    if (SIM.val <= 0) clearInterval(d);
  }, 80);
}

/* #8 — Gradient color for seizure arc */
function _szColor(v) {
  if (v < 0.40) return '#a855f7';            // purple
  if (v < SZ_T) return '#c084fc';            // light purple
  if (v < 0.75) return '#f97316';            // orange (warning)
  return '#ef4444';                          // red (critical)
}

function _renderSz(v) {
  const color = _szColor(v);
  szArc.style.stroke          = color;
  szArc.style.strokeDashoffset = CIRC * (1 - v);
  _animNum(szPct, parseInt(szPct.textContent) || 0, Math.round(v * 100), '%');
  szArc.classList.toggle('hot', v >= SZ_T);
  szPct.style.color = color;
}

/* #23 — Smooth number counter animation */
function _animNum(el, from, to, suffix = '') {
  if (from === to) return;
  const diff  = to - from;
  const steps = Math.min(Math.abs(diff), 18);
  const step  = diff / steps;
  let cur = from, count = 0;
  const old = el._numInterval;
  if (old) clearInterval(old);
  el._numInterval = setInterval(() => {
    count++;
    cur += step;
    el.textContent = Math.round(cur) + suffix;
    if (count >= steps) {
      el.textContent = to + suffix;
      clearInterval(el._numInterval);
    }
  }, 14);
}

/* ── Audio Alarms ── */
let alertAudioCtx = null;
let alertInterval = null;

function bumpAudio() {
  if (!alertAudioCtx) alertAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
}

function startAudioAlarm() {
  bumpAudio();
  if (alertInterval) return;
  alertInterval = setInterval(() => {
    if(alertAudioCtx.state === 'suspended') alertAudioCtx.resume();
    const osc = alertAudioCtx.createOscillator();
    const gain = alertAudioCtx.createGain();
    osc.connect(gain);
    gain.connect(alertAudioCtx.destination);
    osc.frequency.setValueAtTime(880, alertAudioCtx.currentTime); 
    osc.frequency.exponentialRampToValueAtTime(440, alertAudioCtx.currentTime + 0.3); 
    osc.start();
    
    gain.gain.setValueAtTime(0.3, alertAudioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, alertAudioCtx.currentTime + 0.3);
    osc.stop(alertAudioCtx.currentTime + 0.3);
  }, 800);
}

function stopAudioAlarm() {
  if (alertInterval) {
    clearInterval(alertInterval);
    alertInterval = null;
  }
}

/* ── WebSocket ── */
const urlParams = new URLSearchParams(window.location.search);
const pId = parseInt(urlParams.get('patient') || '1');

const ws = new WebSocket('ws://' + location.host + '/stream');
ws.onopen  = () => { 
  startClock(); startVitals(); sessionStartTime = Date.now(); 
  ws.send(JSON.stringify({ action: "start", patient_id: pId }));
};
ws.onclose = () => {
  pName.textContent = 'CONNECTION LOST';
  fpsTxt.textContent = '0 FPS'; fpsTxt.className = 'fps-r';
};
ws.onmessage = e => {
  let d; try { d = JSON.parse(e.data); } catch { return; }
  if      (d.type === 'segment_start') onSegStart(d);
  else if (d.type === 'transition')    onTransition(d);
  else if (d.type === 'seizure_spike') { szSimSpike(); }
  else if (d.type === 'frame_update')  onFrame(d);
  else if (d.type === 'gemini_tier2')  onT2(d);
  else if (d.type === 'gemini_report') onReport(d);
  else if (d.type === 'alert_review') onReview(d);
  else if (d.type === 'review_tick')  onReviewTick(d);
  else if (d.type === 'demo_complete') onDone();
};

/* ── Handlers ── */
function onSegStart(d) {
  trnOver.classList.add('hide');
  ctrlOver.classList.add('hide');
  reviewOver.classList.remove('show'); // hide review overlay for new segment
  _restoreSkipBtn();                   // restore Skip button label
  const lbl  = (d.label || '').toLowerCase();
  const type = d.seg_type || (lbl.includes('fall') ? 'fall' : lbl.includes('seiz') || lbl.includes('unusual') ? 'seizure' : 'normal');
  curSegType = type;

  segProg.textContent  = '[' + (d.progress || '?/?') + ']';
  segTitle.textContent = d.label || '';
  segPill.textContent  = type === 'fall' ? 'FALL EVENT' : type === 'seizure' ? 'SEIZURE EPISODE' : 'NORMAL ACTIVITY';
  segPill.className    = 'seg-pill' + (type === 'fall' ? ' fall' : type === 'seizure' ? ' seizure' : '');

  vidBox.className = '';
  aiLbl.style.display = 'none';
  setPatient(d.patient || 'Unknown Patient', d.label || 'ICU Ward');
  setStatus('monitor', 'MONITORING');
  showGem('idle');
  gemCard.className = '';
  gemTime.classList.remove('show');

  /* Reset gauges and start simulation for the segment type */
  if (type === 'fall') {
    fallSimStart();
    szSimStop();
  } else if (type === 'seizure') {
    szSimStart();
    fallSimStop();
  } else {
    fallSimStop();
    szSimStop();
  }

  /* #17 — Show patient nameplate for 3s */
  patPlate.classList.add('show');
  setTimeout(() => patPlate.classList.remove('show'), 3200);
}

function onTransition(d) {
  trnTxt.textContent = d.message || 'Switching Camera…';
  $('trn-sub').textContent = 'Preparing next patient segment';
  trnOver.classList.remove('hide');
}

function onFrame(d) {
  if (d.frame_b64) vstream.src = 'data:image/jpeg;base64,' + d.frame_b64;
  const fps = d.fps || 0;
  fpsTxt.textContent = fps + ' FPS';
  fpsTxt.className   = fps > 20 ? 'fps-g' : fps > 10 ? 'fps-a' : 'fps-r';

  /* #21 — Frame counter */
  frameTotal++;
  frameCount.textContent = frameTotal.toLocaleString();

  setGauge('fall',    d.fall_risk    || 0);
  setGauge('seizure', d.seizure_risk || 0);
  if (d.analyzing) aiLbl.style.display = 'flex';
  if (d.alert) fireAlert(d.alert);
}

function onT2(d) {
  const ok   = d.decision === 'CONFIRMED';
  const mini = $('mini-' + d.alert_id);
  if (mini) {
    mini.className = 'ac-gem';
    mini.innerHTML = '<div class="mbadge ' + (ok?'ok':'no') + '">' + (ok?'✓ CONFIRMED':'✗ SUPPRESSED') + '</div><div class="mtxt">' + (d.reason||'') + '</div>';
  }
  showGem('result');
  gemCard.className = ok ? 'gc' : 'gs';
  gVerdict.textContent = ok ? '✓ CONFIRMED' : '✗ SUPPRESSED';
  gVerdict.className   = 'gv ' + (ok?'ok':'no');
  gHl.textContent  = ok ? 'Confirming — full analysis in progress…' : 'Alert Suppressed — False Alarm';
  gTxt.textContent = d.reason || '';
  gSev.textContent = ''; gSev.className = 'sv';
  gActs.innerHTML  = '';
}

function onReport(d) {
  const r  = d.report || {}, ok = r.decision === 'CONFIRMED';
  const mini = $('mini-' + d.alert_id);

  /* #15 — Show verify time */
  if (gemAlertStartTime) {
    const elapsed = ((Date.now() - gemAlertStartTime) / 1000).toFixed(1);
    gemTime.textContent = '⚡ ' + elapsed + 's';
    gemTime.classList.add('show');
    gemAlertStartTime = 0;
  }

  /* #16 — Track confirmed/suppressed */
  if (ok) sStats.confirmed++; else sStats.suppressed++;

  if (mini) {
    mini.className = 'ac-gem';
    const acts = (r.actions||[]).slice(0,2).map(a => '<div style="margin-top:2px;font-size:0.65rem;color:#8b9ab7">→ '+a+'</div>').join('');
    mini.innerHTML = '<div class="mbadge '+(ok?'ok':'no')+'">'+(ok?'✓ CONFIRMED':'✗ SUPPRESSED')+'</div><div class="mtxt">'+(r.headline||'')+'</div>'+acts;
  }
  if (!ok) {
    const c = $('ac-' + d.alert_id);
    if (c) { c.style.opacity='0.5'; c.style.borderLeftColor='#4b5a72'; c.querySelector('.ac-type').style.color='#4b5a72'; }
  }

  showGem('result');
  stopAudioAlarm();
  const sev = (r.severity||'moderate').toLowerCase();
  gemCard.className = ok ? 'gc' : 'gs';
  gVerdict.textContent = ok ? '✓ CONFIRMED' : '✗ SUPPRESSED';
  gVerdict.className   = 'gv ' + (ok?'ok':'no');
  gHl.textContent  = r.headline  || 'Analysis Complete';

  /* #6 — Typewriter for narrative */
  typewriterSet(gTxt, r.narrative || '');

  gSev.textContent = sev.toUpperCase();
  gSev.className   = 'sv ' + ({low:'low',moderate:'mod',high:'hi',critical:'crit'}[sev]||'mod');
  gActs.innerHTML  = (r.actions||[]).slice(0,3).map(a => '<li>'+a+'</li>').join('');
  aiLbl.style.display = 'none';
}

function onDone() {
  const s = trnOver.querySelector('.spin');
  if (s) s.style.display = 'none';
  trnTxt.textContent = '✓ Demo Complete';
  $('trn-sub').textContent = totalAlerts + ' alert' + (totalAlerts!==1?'s':'') + ' recorded';
  trnOver.classList.remove('hide');

  /* #16 — Show session stats overlay */
  $('stat-detected').textContent  = sStats.detected;
  $('stat-confirmed').textContent = sStats.confirmed;
  $('stat-suppressed').textContent= sStats.suppressed;
  const elapsed = Math.round((Date.now() - sessionStartTime) / 1000);
  const mins    = Math.floor(elapsed / 60), secs = elapsed % 60;
  $('stat-time').textContent = `Session duration: ${mins}m ${secs}s  ·  ${frameTotal.toLocaleString()} frames processed`;
  setTimeout(() => statsDone.classList.add('show'), 1200);
}

/* ── Alert Review Hold ── */
let _reviewDuration = 0;
function onReview(d) {
  _reviewDuration = d.duration || 0;
  const wasConfirmed = (gVerdict.textContent || '').includes('CONFIRMED');
  const iconEl  = $('rv-icon');
  const titleEl = $('rv-title');
  const hintEl  = $('rv-hint');
  const barWrap = $('rv-bar-wrap');
  const cntWrap = $('rv-count-wrap');
  const nextBtn = $('rv-next-btn');

  if (iconEl)  iconEl.textContent = wasConfirmed ? '✦' : '⚬';
  if (titleEl) {
    titleEl.textContent = wasConfirmed
      ? '✓ ALERT CONFIRMED — Nursing Team Notified'
      : '✗ ALERT SUPPRESSED — False Positive';
    titleEl.style.color = wasConfirmed ? '#34d399' : '#f59e0b';
    titleEl.style.textShadow = wasConfirmed
      ? '0 0 16px rgba(52,211,153,0.4)'
      : '0 0 16px rgba(245,158,11,0.4)';
  }

  if (_reviewDuration === 0) {
    /* Manual mode — show Next Patient button, hide bar/countdown */
    if (barWrap) barWrap.style.display = 'none';
    if (cntWrap) cntWrap.style.display = 'none';
    if (nextBtn) nextBtn.style.display  = 'inline-flex';
    if (hintEl)  hintEl.textContent = 'Review complete — proceed when ready';
    /* Also update the navbar Skip button */
    btnSkip.textContent = '▶ Next Patient';
    btnSkip.style.background = '#34d399';
    btnSkip.style.color      = '#000';
    btnSkip.style.border     = 'none';
  } else {
    /* Timed mode */
    if (barWrap) barWrap.style.display = '';
    if (cntWrap) cntWrap.style.display = '';
    if (nextBtn) nextBtn.style.display  = 'none';
    if (hintEl)  hintEl.textContent = 'Click anywhere to continue immediately';
    rvCount.textContent = _reviewDuration;
    rvBar.style.transition = 'none';
    rvBar.style.width = '100%';
    requestAnimationFrame(() => requestAnimationFrame(() => {
      rvBar.style.transition = 'width 0.95s linear';
    }));
  }
  reviewOver.classList.add('show');
}

function onReviewTick(d) {
  if (_reviewDuration === 0) return; // manual mode — no ticks needed
  const rem = d.remaining;
  rvCount.textContent = rem;
  rvBar.style.width = Math.round((rem / _reviewDuration) * 100) + '%';
  if (rem <= 0) setTimeout(() => reviewOver.classList.remove('show'), 400);
}

function onClickNext() {
  /* Dismiss overlay and signal backend to proceed */
  reviewOver.classList.remove('show');
  _restoreSkipBtn();
  if (ws.readyState === 1) ws.send(JSON.stringify({action: 'skip'}));
}

function _restoreSkipBtn() {
  btnSkip.textContent       = '⏭ Skip';
  btnSkip.style.background  = '';
  btnSkip.style.color       = '';
  btnSkip.style.border      = '';
}

/* ── Alert ── */
function fireAlert(a) {
  const { alert_id: id, event_type: type, confidence: conf, timestamp: ts } = a;
  if ($('ac-' + id)) return;
  totalAlerts++;
  sStats.detected++;
  alertToday.textContent = totalAlerts;
  alertCount.textContent = totalAlerts + ' alert' + (totalAlerts!==1?'s':'');
  logEmpty.style.display = 'none';
  setStatus(type, type === 'seizure' ? '⚠ SEIZURE ALERT' : '⚠ FALL ALERT');
  vidBox.className = type === 'seizure' ? 'as' : 'af';
  showGem('think');
  aiLbl.style.display = 'flex';

  /* #15 — Start timer for verify time */
  gemAlertStartTime = Date.now();

  /* Spike gauges on alert */
  if (type === 'seizure') szSimSpike();
  else fallSimSpike();
  
  startAudioAlarm();

  /* #9 — Avatar alert color */
  pAv.className = 'p-avatar ' + (type === 'seizure' ? 'av-seizure' : 'av-fall');

  /* #5 — Sidebar shake */
  sidebar.classList.remove('shaking');
  void sidebar.offsetWidth;
  sidebar.classList.add('shaking');
  setTimeout(() => sidebar.classList.remove('shaking'), 400);

  const card = document.createElement('div');
  card.id = 'ac-' + id;
  card.className = 'ac ' + type;
  card.innerHTML = `
    <div class="ac-top">
      <div>
        <div class="ac-type">${type==='seizure'?'⚠ SEIZURE DETECTED':'⚠ FALL DETECTED'}</div>
        <div class="ac-conf">Confidence: ${Math.round(conf*100)}%</div>
      </div>
      <div class="ac-r">
        <span class="ac-time">${ts}</span>
        <button class="ac-x" onclick="rmCard('ac-${id}')">×</button>
      </div>
    </div>
    <div class="ac-gem thinking" id="mini-${id}">
      <div class="mpulse"></div>
      <span>Cognitive Core verifying…</span>
    </div>`;
  logFeed.insertBefore(card, logFeed.firstChild);
}

function rmCard(id) {
  const c = $(id); if (c) c.remove();
  if (!logFeed.querySelector('.ac')) logEmpty.style.display = '';
}

/* ── Gauges ── */
function setGauge(type, val) {
  const v = Math.min(1, Math.max(0, val));
  if (type === 'fall') {
    const display = (curSegType === 'seizure') ? Math.min(v, 0.20) : v;
    if (!FSIM.active) {
      _renderFall(display);
    } else {
      if (!FSIM.spiked && v > 0.60) fallSimSpike();
    }
  } else {
    /* Seizure gauge: simulation drives display; backend signal only used at alert */
    if (!SIM.active) _renderSz(v);
    /* No direct injection — szSimSpike() is called via seizure_spike event */
  }
}

/* ── Patient profile ── */
function setPatient(name, room) {
  curHR = 78 + Math.floor(Math.random()*15);
  pName.textContent  = name;
  pRoom.textContent  = room;
  pAv.textContent    = name.charAt(0);
  pAv.className      = 'p-avatar av-normal';
  if (camLbl) camLbl.textContent = 'ICU CAM - Live';
  vHr.textContent   = curHR;
  vSpo2.textContent = 96 + Math.floor(Math.random()*4);
  vBp.textContent   = (110 + Math.floor(Math.random()*25)) + '/' + (70 + Math.floor(Math.random()*20));
  vTemp.textContent = (36.5 + Math.random()).toFixed(1);
  
  const pnpName = document.getElementById('pnp-name');
  if(pnpName) pnpName.textContent = name;
  const pnpRoom = document.getElementById('pnp-room');
  if(pnpRoom) pnpRoom.textContent = room + ' · Live Feed';
  
  startVitals();
}

/* ── Status badge ── */
function setStatus(type, txt) {
  pStatus.textContent = txt;
  pBadge.className = 'p-badge' + (type==='fall'?' pf':type==='seizure'?' ps':'');
  if (type==='fall'||type==='seizure') {
    pBadge.classList.add('pa');
    /* #12 — Bounce animation */
    pBadge.classList.remove('bouncing');
    void pBadge.offsetWidth;
    pBadge.classList.add('bouncing');
  }
}

/* ── Gemini state ── */
function showGem(s) {
  gIdle.classList.add('hide');
  gThink.classList.add('hide');
  gResult.classList.add('hide');
  if (s==='idle')   gIdle.classList.remove('hide');
  else if (s==='think')  gThink.classList.remove('hide');
  else if (s==='result') gResult.classList.remove('hide');
}

/* ── #6 Typewriter effect ── */
function typewriterSet(el, text, speed = 16) {
  el.textContent = '';
  let i = 0;
  if (el._twInterval) clearInterval(el._twInterval);
  el._twInterval = setInterval(() => {
    el.textContent += text[i] || '';
    i++;
    if (i >= text.length) clearInterval(el._twInterval);
  }, speed);
}

/* ── #2 + #3 Vitals simulation ── */
function startVitals() {
  if (hrInterval) clearInterval(hrInterval);
  if (hbInterval) clearInterval(hbInterval);

  hrInterval = setInterval(() => {
    /* HR drift */
    const d = (Math.random()>0.5?1:-1) * Math.round(Math.random()*2);
    curHR = Math.max(curProfile.hr-4, Math.min(curProfile.hr+4, curHR+d));
    _animNum(vHr, parseInt(vHr.textContent)||curHR, curHR, '');

    /* #3 — SpO₂ drift */
    const newSpo2 = curProfile.spo2 + Math.round((Math.random()-0.5)*2);
    const spo2 = Math.max(curProfile.spo2-2, Math.min(curProfile.spo2+1, newSpo2));
    vSpo2.textContent = spo2;

    /* #3 — Temp drift */
    const curT = parseFloat(vTemp.textContent) || curProfile.temp;
    const newT = Math.max(curProfile.temp-0.3, Math.min(curProfile.temp+0.3, curT + (Math.random()-0.5)*0.15));
    vTemp.textContent = newT.toFixed(1);
  }, 2200);

  /* #2 — Heartbeat pulse in sync with BPM */
  hbInterval = setInterval(() => {
    if (!viHr) return;
    viHr.style.transform = 'scale(1.45)';
    setTimeout(() => { if (viHr) viHr.style.transform = 'scale(1)'; }, 110);
  }, 60000 / curHR);
}

/* ── #20 Clock ── */
function startClock() {
  if (clockInterval) clearInterval(clockInterval);
  const tick = () => {
    const n  = new Date();
    const t  = String(n.getHours()).padStart(2,'0') + ':' + String(n.getMinutes()).padStart(2,'0') + ':' + String(n.getSeconds()).padStart(2,'0');
    tsLbl.textContent    = t;
    navClock.textContent = t;
  };
  tick(); clockInterval = setInterval(tick, 1000);
}

/* ── Buttons ── */
function onPause() {
  isPaused = !isPaused;
  btnPause.textContent = isPaused ? '▶ Resume' : '⏸ Pause';
  if (ws.readyState===1) ws.send(JSON.stringify({action: isPaused?'pause':'resume'}));
}
function onSkip() {
  /* If we're in the review hold, treat Skip as Next Patient */
  if (reviewOver.classList.contains('show')) { onClickNext(); return; }
  trnTxt.textContent = 'Skipping to next patient…';
  $('trn-sub').textContent = '';
  trnOver.classList.remove('hide');
  if (ws.readyState===1) ws.send(JSON.stringify({action:'skip'}));
}
function onResume() {
  ctrlOver.classList.add('hide');
  isPaused = false;
  btnPause.textContent = '⏸ Pause';
  if (ws.readyState===1) ws.send(JSON.stringify({action:'resume'}));
}

/* Init */
setPatient('Patient A');
