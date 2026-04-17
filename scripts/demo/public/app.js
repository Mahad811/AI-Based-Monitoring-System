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
    const nurseName = localStorage.getItem('vg_nurse_name') || 'Staff';
    const el = document.getElementById('nurse-name-display');
    if (el) el.textContent = nurseName;

    // Navbar avatar
    if (window.VG) {
        const av    = VG.avatar(nurseName);
        const avEl  = document.getElementById('idx-nav-av');
        if (avEl) { avEl.textContent = av.initials; avEl.style.background = av.bgColor; }
    }

    // RBAC: Hide Admin Hub if not admin
    const staffId  = localStorage.getItem('vg_staff_id');
    const adminBtn = document.getElementById('admin-hub-btn');
    if (adminBtn && staffId !== 'admin') {
        adminBtn.style.display = 'none';
    }

    // Sync sound toggle pill to persisted state
    VGAudio._syncBtn();

    // Unlock AudioContext on first user gesture
    const _unlock = () => { VGAudio.unlock(); };
    document.addEventListener('click',      _unlock, { once: true });
    document.addEventListener('touchstart', _unlock, { once: true });

    // Space bar → acknowledge active alert banner
    document.addEventListener('keydown', e => {
        if (e.code === 'Space' && !e.target.matches('input,textarea,button,select')) {
            const banner = document.getElementById('alert-banner');
            if (banner && banner.classList.contains('ab-show')) {
                e.preventDefault();
                ackAlert();
            }
        }
    });
});

/* ── Relative time auto-updater for alert cards ── */
setInterval(() => {
    document.querySelectorAll('.ac-reltime').forEach(el => {
        const epoch = parseInt(el.dataset.epoch, 10);
        if (!epoch) return;
        const s = Math.floor((Date.now() - epoch) / 1000);
        el.textContent = s < 5 ? 'Just now' : s < 60 ? s + 's ago'
            : s < 3600 ? Math.floor(s / 60) + 'm ago'
            : Math.floor(s / 3600) + 'h ago';
    });
}, 12000);

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

/* ── #19 Seizure Gauge Simulator ────────────────────────────────────────
   Mirrors FSIM exactly. Idles at a low baseline during resting clips,
   then executes a cinematic ramp to critical on spike. Only runs during
   seizure segments — all other segments call szSimStop() to reset to 0.
───────────────────────────────────────────────────────────────────── */
const SZSIM = {
  val: 0, target: 0, active: false, spiked: false, _tid: null,
};

function _szSimTick() {
  if (!SZSIM.active) return;
  // Idle: tight noise for a steady flat line. Spiked: finer noise so the climb looks smooth.
  const noise = (Math.random() - 0.5) * (SZSIM.spiked ? 0.9 : 1.4);
  // Idle factor 0.038 → very lazy drift. Spike factor 0.052 → ~4-5s cinematic ramp to critical.
  SZSIM.val += (SZSIM.target - SZSIM.val) * (SZSIM.spiked ? 0.052 : 0.038) + noise * 0.14;
  SZSIM.val  = Math.max(0, Math.min(100, SZSIM.val));
  if (!SZSIM.spiked) {
    SZSIM.target += (Math.random() - 0.5) * 0.8;
    SZSIM.target  = Math.max(3, Math.min(10, SZSIM.target));
  } else {
    SZSIM.target += (Math.random() - 0.5) * 0.6;
    SZSIM.target  = Math.max(88, Math.min(97, SZSIM.target));
  }
  _renderSz(SZSIM.val / 100);
  SZSIM._tid = setTimeout(_szSimTick, 110);
}

function szSimStart() {
  if (SZSIM._tid) clearTimeout(SZSIM._tid);
  SZSIM.val    = 0;
  SZSIM.target = 4 + Math.random() * 5;   // calm 4-9% idle baseline
  SZSIM.active = true;
  SZSIM.spiked = false;
  SZSIM._tid   = setTimeout(_szSimTick, 110);
}

function szSimElevate() {
  // Called on seizure_spike (clip transition): mildly raises the baseline to
  // signal the system is analysing seizure-pattern footage. Stays clearly
  // sub-threshold so no false alarm impression is given before the alert fires.
  if (!SZSIM.active || SZSIM.spiked) return;
  SZSIM.target = 20 + Math.random() * 8;   // elevate to 20-28%, not critical
}

function szSimSpike() {
  // Called only by fireAlert() — ties the full climb to the confirmed alert.
  if (SZSIM.spiked) return;
  SZSIM.spiked = true;
  SZSIM.target = 89 + Math.random() * 8;
}

function szSimStop() {
  SZSIM.active = false; SZSIM.spiked = false;
  if (SZSIM._tid) { clearTimeout(SZSIM._tid); SZSIM._tid = null; }
  const d = setInterval(() => {
    SZSIM.val = Math.max(0, SZSIM.val - 2.5);
    _renderSz(SZSIM.val / 100);
    if (SZSIM.val <= 0) clearInterval(d);
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

/* ══════════════════════════════════════════════════════════════════
   VGAudio — Hospital-grade layered audio system
   Fall:    descending A5→F5→D5, triangle wave, 1.05s cadence
   Seizure: rapid E5→C5 pulse,   square  wave, 0.65s cadence
   TTS voice fires for alert + verdict via Web Speech API
══════════════════════════════════════════════════════════════════ */
const VGAudio = (() => {
  let _ctx        = null;
  let _soundOn    = localStorage.getItem('vg_sound_on') !== 'false';
  let _alarmTimer = null;
  let _voice      = null;
  let _voiceReady = false;

  function _getCtx() {
    if (!_ctx) _ctx = new (window.AudioContext || window.webkitAudioContext)();
    return _ctx;
  }

  function _loadVoice() {
    if (_voiceReady) return;
    const voices = speechSynthesis.getVoices();
    if (!voices.length) return;
    _voice = voices.find(v => v.name === 'Google UK English Female')
          || voices.find(v => v.name === 'Samantha')
          || voices.find(v => /en[-_]GB/i.test(v.lang))
          || voices.find(v => /en/i.test(v.lang) && /female/i.test(v.name))
          || voices.find(v => /en/i.test(v.lang))
          || voices[0] || null;
    _voiceReady = true;
  }

  if (typeof speechSynthesis !== 'undefined') {
    speechSynthesis.addEventListener('voiceschanged', _loadVoice);
    _loadVoice();
  }

  function unlock() {
    try {
      const ctx = _getCtx();
      if (ctx.state === 'suspended') ctx.resume();
    } catch (_) {}
  }

  /* Core tone generator — ADSR envelope */
  function _tone(freq, wave, dur, peak, startAt) {
    const ctx = _getCtx();
    if (ctx.state === 'suspended') ctx.resume();
    const osc = ctx.createOscillator();
    const gn  = ctx.createGain();
    osc.type = wave;
    osc.frequency.setValueAtTime(freq, ctx.currentTime + startAt);
    gn.gain.setValueAtTime(0, ctx.currentTime + startAt);
    gn.gain.linearRampToValueAtTime(peak, ctx.currentTime + startAt + 0.08);
    gn.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + startAt + dur);
    osc.connect(gn);
    gn.connect(ctx.destination);
    osc.start(ctx.currentTime + startAt);
    osc.stop(ctx.currentTime + startAt + dur + 0.02);
  }

  function chime(type) {
    if (!_soundOn) return;
    unlock();
    if (type === 'pre-alert') {
      _tone(1046.5, 'sine', 0.38, 0.18, 0.00);  // C6
      _tone(783.99, 'sine', 0.38, 0.14, 0.22);  // G5
    } else if (type === 'resolved') {
      _tone(880.00, 'sine', 0.32, 0.16, 0.00);  // A5
      _tone(659.25, 'sine', 0.32, 0.13, 0.24);  // E5
      _tone(523.25, 'sine', 0.42, 0.11, 0.50);  // C5
    }
  }

  function alarm(type) {
    stopAlarm();
    const isSz    = type === 'seizure';
    const cadence = isSz ? 650 : 1050;
    const maxBursts = 1;   // play 1 short burst then stop automatically
    let count = 0;
    const fire = () => {
      if (!_soundOn) return;
      unlock();
      if (isSz) {
        _tone(659.25, 'square',   0.18, 0.22, 0.00);
        _tone(523.25, 'square',   0.18, 0.22, 0.22);
      } else {
        _tone(880.00, 'triangle', 0.18, 0.26, 0.00);
        _tone(698.46, 'triangle', 0.18, 0.22, 0.22);
        _tone(587.33, 'triangle', 0.18, 0.18, 0.44);
      }
      count++;
      if (count >= maxBursts) stopAlarm();
    };
    fire();
    _alarmTimer = setInterval(fire, cadence);
  }

  function stopAlarm() {
    if (_alarmTimer) { clearInterval(_alarmTimer); _alarmTimer = null; }
  }

  /* Low-frequency 60 Hz thump — pairs with the video flash */
  function thump() {
    if (!_soundOn) return;
    unlock();
    _tone(60, 'sine', 0.14, 0.45, 0.00);
    _tone(80, 'sine', 0.10, 0.25, 0.04);
  }

  function say(text) {
    if (!_soundOn || typeof speechSynthesis === 'undefined') return;
    speechSynthesis.cancel();
    _loadVoice();
    const utt = new SpeechSynthesisUtterance(text);
    if (_voice) utt.voice = _voice;
    utt.rate   = 0.90;
    utt.pitch  = 1.05;
    utt.volume = 0.88;
    setTimeout(() => speechSynthesis.speak(utt), 900);
  }

  function setSound(on) {
    _soundOn = on;
    localStorage.setItem('vg_sound_on', on ? 'true' : 'false');
    if (!on) { stopAlarm(); if (typeof speechSynthesis !== 'undefined') speechSynthesis.cancel(); }
    _syncBtn();
  }

  function toggle() { setSound(!_soundOn); return _soundOn; }
  function isOn()   { return _soundOn; }

  function _syncBtn() {
    const btn = document.getElementById('snd-toggle');
    if (!btn) return;
    btn.textContent    = _soundOn ? '🔊 Sound' : '🔇 Muted';
    btn.style.opacity  = _soundOn ? '1' : '0.55';
    btn.style.color    = _soundOn ? '#34d399' : '#4b5a72';
    btn.style.borderColor = _soundOn ? 'rgba(52,211,153,0.3)' : 'rgba(255,255,255,0.1)';
  }

  return { unlock, chime, alarm, stopAlarm, thump, say, toggle, isOn, _syncBtn };
})();

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
  if      (d.type === 'segment_start')  onSegStart(d);
  else if (d.type === 'transition')     onTransition(d);
  else if (d.type === 'frame_update')   onFrame(d);
  else if (d.type === 'audio_alert')    onAudioAlert(d);
  else if (d.type === 'seizure_spike')  szSimElevate();
  else if (d.type === 'gemini_tier2')   onT2(d);
  else if (d.type === 'gemini_report')  onReport(d);
  else if (d.type === 'alert_review')   onReview(d);
  else if (d.type === 'review_tick')    onReviewTick(d);
  else if (d.type === 'demo_complete')  onDone();
};

/* ── Handlers ── */
function onSegStart(d) {
  trnOver.classList.add('hide');
  ctrlOver.classList.add('hide');
  reviewOver.classList.remove('show'); // hide review overlay for new segment

  /* Clear any lingering alert state from previous segment */
  document.body.classList.remove('sev-fall', 'sev-seizure');
  hideAlertBanner();
  VGAudio.stopAlarm();

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

  /* Establish environment per segment type */
  if (type === 'fall') {
    fallSimStart();
    szSimStop();
  } else if (type === 'seizure') {
    fallSimStop();
    szSimStart();
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
  /* Fade shimmer bar on preliminary verdict */
  const shimBar = $('ac-shim-' + d.alert_id);
  if (shimBar) shimBar.classList.add('done');

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
  /* Fade shimmer bar + dim suppressed cards */
  const shimBar = $('ac-shim-' + d.alert_id);
  if (shimBar) shimBar.classList.add('done');
  if (!ok) {
    const c = $('ac-' + d.alert_id);
    if (c) { c.style.opacity = '0.55'; c.style.filter = 'grayscale(0.35)'; }
  }

  showGem('result');

  /* Hospital-grade audio: stop alarm, play resolution chime, speak verdict */
  VGAudio.stopAlarm();
  VGAudio.chime('resolved');
  const sev = (r.severity || 'moderate').toLowerCase();
  if (ok) {
    VGAudio.say(`Confirmed. Severity ${sev}.`);
  } else {
    VGAudio.say('False alarm.');
  }

  /* Dismiss alert banner + strip severity accent */
  hideAlertBanner();
  document.body.classList.remove('sev-fall', 'sev-seizure');

  gemCard.className = ok ? 'gc' : 'gs';
  gVerdict.textContent = ok ? '✓ CONFIRMED' : '✗ SUPPRESSED';
  gVerdict.className   = 'gv ' + (ok?'ok':'no');
  gHl.textContent  = r.headline || 'Analysis Complete';

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

function onAudioAlert(d) {
  // Use integrated VGAudio system from the UI overhaul
  if (!d.classification || d.classification === 'noise') return;
  const isCritical = d.panic > 0.5;
  VGAudio.say(`${d.classification} detected in audio feed.`);
  if (isCritical) {
      VGAudio.chime('pre-alert');
  }
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

  /* Spike gauges */
  if (type === 'seizure') szSimSpike();
  else fallSimSpike();
  /* ── Severity accent system ── */
  document.body.classList.remove('sev-fall', 'sev-seizure');
  document.body.classList.add('sev-' + type);

  /* ── Cinematic flash + low-frequency thump ── */
  triggerVideoFlash(type);
  VGAudio.thump();

  /* ── Hospital-grade alarm + voice announcement ── */
  VGAudio.alarm(type);
  const room = pRoom.textContent || 'ICU';
  const name = pName.textContent || 'Patient';
  VGAudio.say(`${type === 'seizure' ? 'Seizure' : 'Fall'} detected. ${room}.`);

  /* ── Slide-in alert banner ── */
  showAlertBanner(type, conf);

  /* ── HUD target label ── */
  updateHUDLabel(conf);

  /* #9 — Avatar alert color */
  pAv.className = 'p-avatar ' + (type === 'seizure' ? 'av-seizure' : 'av-fall');

  /* #5 — Sidebar shake */
  sidebar.classList.remove('shaking');
  void sidebar.offsetWidth;
  sidebar.classList.add('shaking');
  setTimeout(() => sidebar.classList.remove('shaking'), 400);

  /* ── Hero alert card ── */
  const confPct  = Math.round(conf * 100);
  const ringR    = 16;
  const ringC    = +(2 * Math.PI * ringR).toFixed(2);
  const arcColor = type === 'seizure' ? '#a855f7' : '#f59e0b';
  const card = document.createElement('div');
  card.id = 'ac-' + id;
  card.className = 'ac ' + type;
  card.innerHTML = `
    <div class="ac-stripe"></div>
    <div class="ac-hero">
      <svg class="ac-conf-ring" width="44" height="44" viewBox="0 0 44 44">
        <circle cx="22" cy="22" r="${ringR}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="4"/>
        <circle cx="22" cy="22" r="${ringR}" fill="none" stroke="${arcColor}" stroke-width="4"
                stroke-linecap="round"
                stroke-dasharray="${ringC}"
                stroke-dashoffset="${(ringC * (1 - conf)).toFixed(2)}"
                style="transform:rotate(-90deg);transform-origin:center;transform-box:fill-box;"/>
        <text x="22" y="26" text-anchor="middle" font-size="8.5" font-weight="800"
              fill="${arcColor}" font-family="'JetBrains Mono',monospace">${confPct}</text>
      </svg>
      <div class="ac-hero-body">
        <div class="ac-type">${type==='seizure'?'⚠ SEIZURE DETECTED':'⚠ FALL DETECTED'}</div>
        <div class="ac-subline">
          <span class="ac-reltime" data-epoch="${Date.now()}">Just now</span>
          <span class="ac-time"> · ${ts}</span>
        </div>
      </div>
      <button class="ac-x" onclick="rmCard('ac-${id}')">×</button>
    </div>
    <div class="ac-shimmer-bar" id="ac-shim-${id}"></div>
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

/* ── Audio Alerts ── */
function onAudioAlert(a) {
  const { alert_id, event_type: type, sound_type: sound, confidence: conf, timestamp: ts } = a;
  const id = alert_id;
  
  totalAlerts++;
  sStats.detected++;
  alertToday.textContent = totalAlerts;
  alertCount.textContent = totalAlerts + ' alert' + (totalAlerts!==1?'s':'');
  logEmpty.style.display = 'none';

  // Auditory Siren moved to LLM Validation output to prevent false-alarm fatigue

  /* Sidebar shake */
  sidebar.classList.remove('shaking');
  void sidebar.offsetWidth;
  sidebar.classList.add('shaking');
  setTimeout(() => sidebar.classList.remove('shaking'), 400);

  const card = document.createElement('div');
  card.id = 'ac-' + id;
  card.className = 'ac';
  card.style.borderLeftColor = '#8b5cf6';
  card.style.background = 'rgba(139, 92, 246, 0.1)';
  
  card.innerHTML = `
    <div class="ac-top">
      <div>
        <div class="ac-type" style="color: #a78bfa;">🎙 AUDIO ALERT: ${type.toUpperCase()}</div>
        <div class="ac-conf">Detected Sound: <span style="color:#fff; font-weight:bold;">${sound}</span> (Conf: ${Math.round(conf*100)}%)</div>
      </div>
      <div class="ac-r">
        <span class="ac-time">${ts}</span>
        <button class="ac-x" style="color:#a78bfa" onclick="rmCard('ac-${id}')">×</button>
      </div>
    </div>
    <div class="ac-gem thinking" id="mini-${id}" style="margin-top:8px;">
      <div class="mpulse"></div>
      <span>Cognitive Core tracking context…</span>
    </div>`;
  logFeed.insertBefore(card, logFeed.firstChild);
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
    /* Seizure gauge is driven entirely by SZSIM during seizure segments.
       Spike is triggered by seizure_spike WS (clip transition) or fireAlert —
       never by raw backend confidence values, which can be noisy on resting clips. */
  }
}

/* ── Alert Banner helpers ── */
function showAlertBanner(type, conf) {
  const banner = document.getElementById('alert-banner');
  if (!banner) return;
  banner.className = 'ab-show ab-' + type;
  const typeEl = document.getElementById('ab-type');
  const roomEl = document.getElementById('ab-room');
  const nameEl = document.getElementById('ab-name');
  const confEl = document.getElementById('ab-conf');
  if (typeEl) typeEl.textContent = type === 'seizure' ? '⚠ SEIZURE DETECTED' : '⚠ FALL DETECTED';
  if (roomEl) roomEl.textContent = pRoom.textContent || 'ICU';
  if (nameEl) nameEl.textContent = pName.textContent || 'Patient';
  if (confEl) confEl.textContent = Math.round(conf * 100) + '%';
}

function hideAlertBanner() {
  const banner = document.getElementById('alert-banner');
  if (!banner || !banner.classList.contains('ab-show')) return;
  banner.classList.remove('ab-show');
  // Clear colour classes after slide-out transition (320ms) completes
  setTimeout(() => { banner.className = ''; }, 380);
}

function ackAlert() {
  VGAudio.stopAlarm();
  if (typeof speechSynthesis !== 'undefined') speechSynthesis.cancel();
  hideAlertBanner();
}

/* ── Video Flash ── */
function triggerVideoFlash(type) {
  const fl = document.getElementById('vg-flash');
  if (!fl) return;
  fl.className = type === 'seizure' ? 'flash-seizure' : 'flash-fall';
  void fl.offsetWidth;
  fl.classList.add('firing');
  setTimeout(() => { fl.className = ''; }, 280);
}

/* ── HUD label update ── */
function updateHUDLabel(conf) {
  const lbl = document.getElementById('hud-label');
  if (!lbl) return;
  const raw  = (pName.textContent || 'Patient A').toUpperCase();
  const tag  = raw.split(' ').slice(0, 2).join('_');
  lbl.textContent = `TARGET: ${tag} · CONF ${Math.round(conf * 100)}% · MoViNet-A2`;
}

/* ── Patient profile ── */
let curProfile = { hr: 78, spo2: 97, temp: 36.8 };

function setPatient(name, room) {
  curHR = 72 + Math.floor(Math.random()*18);
  curProfile = {
    hr:   curHR,
    spo2: 96 + Math.floor(Math.random()*4),
    temp: parseFloat((36.3 + Math.random() * 0.9).toFixed(1)),
  };
  pName.textContent  = name;
  pRoom.textContent  = room;
  pAv.textContent    = name.charAt(0);
  pAv.className      = 'p-avatar av-normal';
  if (camLbl) camLbl.textContent = 'ICU CAM - Live';
  vHr.textContent   = curProfile.hr;
  vSpo2.textContent = curProfile.spo2;
  vBp.textContent   = (110 + Math.floor(Math.random()*25)) + '/' + (70 + Math.floor(Math.random()*20));
  vTemp.textContent = curProfile.temp.toFixed(1);

  // Navbar avatar via VG utility
  if (window.VG) {
    const av   = VG.avatar(name);
    const avEl = document.getElementById('idx-nav-av');
    if (avEl) { avEl.textContent = av.initials; avEl.style.background = av.bgColor; }
  }

  const pnpName = document.getElementById('pnp-name');
  if (pnpName) pnpName.textContent = name;
  const pnpRoom = document.getElementById('pnp-room');
  if (pnpRoom) pnpRoom.textContent = room + ' · Live';

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
