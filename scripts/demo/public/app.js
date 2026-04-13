/**
 * Vital Guardian — Dashboard JavaScript
 * Handles WebSocket streaming, gauge animations, patient vitals,
 * Gemini decision card state machine, and alert log management.
 */

'use strict';

// ─────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────
const FALL_THRESH    = 0.55;
const SZ_THRESH      = 0.48;
const GAUGE_CIRCUM   = 238.76; // 2π × r(38)

// ─────────────────────────────────────────────────────────────
// PATIENT DATA PROFILES
// Static hospital-plausible profiles per patient.
// Vitals subtly animate to look alive.
// ─────────────────────────────────────────────────────────────
const PATIENT_PROFILES = {
    'Patient A': {
        fullName:  'Patient A · Hassan M.',
        room:      'Room 304 · Bed 2',
        avatar:    'A',
        camera:    'ICU CAM 01',
        hr:        78,
        spo2:      96,
        bp:        '132/84',
        temp:      36.9,
    },
    'Patient B': {
        fullName:  'Patient B · Fatima K.',
        room:      'Room 107 · Bed 1',
        avatar:    'B',
        camera:    'ICU CAM 02',
        hr:        88,
        spo2:      94,
        bp:        '118/76',
        temp:      37.2,
    },
    'Patient C': {
        fullName:  'Patient C · Tariq R.',
        room:      'Room 212 · Bed 3',
        avatar:    'C',
        camera:    'ICU CAM 03',
        hr:        72,
        spo2:      98,
        bp:        '120/80',
        temp:      36.6,
    },
};

// ─────────────────────────────────────────────────────────────
// DOM REFERENCES
// ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// Video
const videoStream      = $('video-stream');
const videoContainer   = $('video-container');
const cameraLabel      = $('camera-label');
const videoTimestamp   = $('video-timestamp');
const videoAnalyzingTag= $('video-analyzing-tag');

// Overlays
const overlay          = $('transition-overlay');
const overlayText      = $('transition-text');
const controlsOverlay  = $('controls-overlay');
const controlStatus    = $('control-status');

// Buttons
const btnResume        = $('btn-resume');
const btnSkip          = $('btn-skip');
const btnManualPause   = $('btn-manual-pause');
const btnManualSkip    = $('btn-manual-skip');

// Header stats
const fpsCounter       = $('fps-counter');
const alertCountToday  = $('alert-count-today');

// Patient card
const patientName      = $('patient-name');
const patientRoom      = $('patient-room');
const patientAvatar    = $('patient-avatar');
const patientStatusBadge = $('patient-status-badge');
const statusDot        = $('status-dot');
const statusText       = $('status-text');
const hrValue          = $('hr-value');
const spo2Value        = $('spo2-value');
const bpValue          = $('bp-value');
const tempValue        = $('temp-value');

// Gauges
const fallGaugeFill    = $('fall-gauge-fill');
const seizureGaugeFill = $('seizure-gauge-fill');
const fallPct          = $('fall-pct');
const seizurePct       = $('seizure-pct');

// Segment info
const segmentProgress  = $('segment-progress');
const segmentTitle     = $('segment-title');
const segmentTypePill  = $('segment-type-pill');

// Gemini decision card
const geminiCard       = $('gemini-decision-card');
const geminiIdleState  = $('gemini-idle-state');
const geminiAnalyzing  = $('gemini-analyzing-state');
const geminiResult     = $('gemini-result-state');
const geminiVerdict    = $('gemini-verdict-badge');
const geminiHeadline   = $('gemini-headline');
const geminiNarrative  = $('gemini-narrative');
const geminiSeverity   = $('gemini-severity');
const geminiActions    = $('gemini-actions-list');

// Alert feed
const alertFeed        = $('alert-feed');
const emptyAlerts      = $('empty-alerts');
const alertCountBadge  = $('alert-count-badge');

// ─────────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────────
let totalAlerts       = 0;
let isManualPaused    = false;
let currentSegType    = 'normal';
let activeAlertType   = null;  // null | 'fall' | 'seizure'
let vitalsInterval    = null;
let clockInterval     = null;
let currentProfile    = PATIENT_PROFILES['Patient A'];
let currentHR         = 78;

// ─────────────────────────────────────────────────────────────
// WEBSOCKET
// ─────────────────────────────────────────────────────────────
const ws = new WebSocket(`ws://${window.location.host}/stream`);

ws.onopen = () => {
    console.log('[VG] WebSocket connected');
    startClock();
    startVitalsSimulation();
};

ws.onclose = () => {
    console.log('[VG] WebSocket disconnected');
    patientName.textContent = 'CONNECTION LOST';
    fpsCounter.textContent  = '0 FPS';
    fpsCounter.className    = 'fps fps-slow';
    setSystemStatus('disconnected', 'DISCONNECTED', '');
};

ws.onerror = err => console.error('[VG] WebSocket error:', err);

ws.onmessage = evt => {
    let data;
    try { data = JSON.parse(evt.data); }
    catch { return; }

    switch (data.type) {

        case 'segment_start':
            handleSegmentStart(data);
            break;

        case 'transition':
            handleTransition(data);
            break;

        case 'frame_update':
            handleFrameUpdate(data);
            break;

        case 'gemini_tier2':
            handleGeminiTier2(data);
            break;

        case 'gemini_report':
            handleGeminiReport(data);
            break;

        case 'demo_complete':
            handleDemoComplete();
            break;

        default:
            break;
    }
};

// ─────────────────────────────────────────────────────────────
// HANDLERS
// ─────────────────────────────────────────────────────────────

function handleSegmentStart(data) {
    overlay.classList.add('hidden');
    controlsOverlay.classList.add('hidden');

    const patient  = data.patient || 'Patient A';
    const label    = data.label   || 'Monitoring';
    const progress = data.progress|| '?/?';

    // Determine type from label
    const lbl = label.toLowerCase();
    if (lbl.includes('fall'))    currentSegType = 'fall';
    else if (lbl.includes('seiz') || lbl.includes('unusual')) currentSegType = 'seizure';
    else currentSegType = 'normal';

    // Update segment info bar
    segmentProgress.textContent = `[${progress}]`;
    segmentTitle.textContent    = label;
    updateSegmentTypePill(currentSegType);

    // Reset video border and alert state
    videoContainer.className = 'video-container';
    activeAlertType = null;
    videoAnalyzingTag.style.display = 'none';

    // Update patient profile
    loadPatientProfile(patient);

    // Reset status
    setSystemStatus('normal', 'MONITORING', '');

    // Reset Gemini decision card to idle
    showGeminiState('idle');
    geminiCard.className = 'gemini-decision-card glass-card';
}

function handleTransition(data) {
    overlayText.textContent = data.message || 'Switching Camera…';
    overlay.classList.remove('hidden');
}

function handleFrameUpdate(data) {
    // Update video frame
    if (data.frame_b64) {
        videoStream.src = `data:image/jpeg;base64,${data.frame_b64}`;
    }

    // FPS counter
    const fps = data.fps || 0;
    fpsCounter.textContent = `${fps} FPS`;
    fpsCounter.className   = 'fps ' + (fps > 20 ? 'fps-fast' : fps > 10 ? 'fps-med' : 'fps-slow');

    // Update gauges
    const fallRisk    = data.fall_risk    || 0;
    const seizureRisk = data.seizure_risk || 0;
    updateGauge('fall',    fallRisk);
    updateGauge('seizure', seizureRisk);

    // Analyzing tag
    if (data.analyzing) {
        videoAnalyzingTag.style.display = 'flex';
    }

    // Instant alert
    if (data.alert) {
        handleInstantAlert(data.alert);
    }
}

function handleInstantAlert(alertData) {
    const { alert_id, event_type, confidence, timestamp } = alertData;

    totalAlerts++;
    alertCountToday.textContent   = totalAlerts;
    alertCountBadge.textContent   = `${totalAlerts} alert${totalAlerts !== 1 ? 's' : ''}`;

    // Hide empty state
    if (emptyAlerts) emptyAlerts.style.display = 'none';

    // Update system status
    activeAlertType = event_type;
    if (event_type === 'seizure') {
        setSystemStatus('seizure', '⚠  SEIZURE ALERT', 'status-seizure status-alert');
        videoContainer.className = 'video-container alert-active-seizure';
    } else {
        setSystemStatus('fall', '⚠  FALL ALERT', 'status-fall status-alert');
        videoContainer.className = 'video-container alert-active-fall';
    }

    // Show Gemini card analyzing
    showGeminiState('analyzing');
    videoAnalyzingTag.style.display = 'flex';

    // Build alert card
    const confPct  = Math.round(confidence * 100);
    const typeLabel = event_type === 'seizure' ? '⚠ SEIZURE DETECTED' : '⚠ FALL DETECTED';
    const cardClass = event_type === 'seizure' ? 'seizure' : 'fall';

    const el = document.createElement('div');
    el.className = `alert-card ${cardClass}`;
    el.id        = `alert-card-${alert_id}`;

    el.innerHTML = `
        <div class="alert-card-header">
            <div>
                <div class="alert-type-label">${typeLabel}</div>
                <div class="alert-conf">Confidence: ${confPct}%</div>
            </div>
            <div class="alert-header-right">
                <span class="alert-time">${timestamp}</span>
                <button class="alert-dismiss-btn" onclick="this.closest('.alert-card').remove(); updateAlertCount();" aria-label="Dismiss alert">×</button>
            </div>
        </div>
        <div class="alert-gemini-mini analyzing" id="gemini-mini-${alert_id}">
            <div class="mini-pulse"></div>
            <span>Cognitive Core verifying…</span>
        </div>
    `;

    // Insert at top of feed
    alertFeed.insertBefore(el, alertFeed.firstChild);
    el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function handleGeminiTier2(data) {
    // Fast binary result arrives first — update the mini box in the card
    const { alert_id, decision, reason } = data;

    const miniBox = document.getElementById(`gemini-mini-${alert_id}`);
    if (miniBox) {
        const isConfirmed = decision === 'CONFIRMED';
        miniBox.className = 'alert-gemini-mini';
        miniBox.innerHTML = `
            <div class="mini-verdict ${isConfirmed ? 'confirmed' : 'suppressed'}">
                ${isConfirmed ? '✓ CONFIRMED' : '✗ SUPPRESSED'}
            </div>
            <div class="mini-narrative">${reason || ''}</div>
        `;
    }

    // Update the big Gemini decision card (Tier2 = fast preview)
    if (decision === 'CONFIRMED') {
        showGeminiState('result');
        geminiCard.className = 'gemini-decision-card glass-card state-confirmed';
        geminiVerdict.className  = 'gemini-verdict verdict-confirmed';
        geminiVerdict.textContent = '✓ CONFIRMED';
        geminiHeadline.textContent = 'Verifying clinical details…';
        geminiNarrative.textContent = reason || 'Alert confirmed. Full clinical analysis in progress…';
        geminiSeverity.textContent  = '';
        geminiSeverity.className    = 'severity-value';
        geminiActions.innerHTML     = '';
    } else {
        showGeminiState('result');
        geminiCard.className = 'gemini-decision-card glass-card state-suppressed';
        geminiVerdict.className  = 'gemini-verdict verdict-suppressed';
        geminiVerdict.textContent = '✗ SUPPRESSED';
        geminiHeadline.textContent = 'Alert Suppressed — False Alarm';
        geminiNarrative.textContent = reason || 'Cognitive Core determined this is not a clinical event.';
        geminiSeverity.textContent  = '';
        geminiActions.innerHTML     = '';
    }
}

function handleGeminiReport(data) {
    const { alert_id, report } = data;

    // Update mini box in the alert card
    const miniBox = document.getElementById(`gemini-mini-${alert_id}`);
    if (miniBox) {
        const isConfirmed = report.decision === 'CONFIRMED';
        const actHtml = (report.actions || []).slice(0, 2).map(a => `<div style="margin-top:3px;opacity:0.9">→ ${a}</div>`).join('');
        miniBox.className = 'alert-gemini-mini';
        miniBox.innerHTML = `
            <div class="mini-verdict ${isConfirmed ? 'confirmed' : 'suppressed'}">
                ${isConfirmed ? '✓ CONFIRMED' : '✗ SUPPRESSED'}
            </div>
            <div class="mini-narrative">${report.headline || ''}</div>
            ${actHtml}
        `;

        // Dim card if suppressed
        if (!isConfirmed) {
            const card = document.getElementById(`alert-card-${alert_id}`);
            if (card) {
                card.style.opacity = '0.55';
                card.style.borderLeftColor = 'var(--text-3)';
                card.querySelector('.alert-type-label').style.color = 'var(--text-3)';
            }
        }
    }

    // Update the big Gemini decision card with full report
    const isConfirmed   = report.decision === 'CONFIRMED';
    const actions       = report.actions || [];
    const severity      = (report.severity || 'moderate').toLowerCase();
    const severityClass = `severity-${severity}`;

    if (isConfirmed) {
        geminiCard.className = 'gemini-decision-card glass-card state-confirmed';
        geminiVerdict.className  = 'gemini-verdict verdict-confirmed';
        geminiVerdict.textContent = '✓ CONFIRMED';
    } else {
        geminiCard.className = 'gemini-decision-card glass-card state-suppressed';
        geminiVerdict.className  = 'gemini-verdict verdict-suppressed';
        geminiVerdict.textContent = '✗ SUPPRESSED';
    }

    showGeminiState('result');
    geminiHeadline.textContent  = report.headline  || 'Analysis Complete';
    geminiNarrative.textContent = report.narrative || 'Clinical analysis complete.';

    geminiSeverity.textContent  = severity.toUpperCase();
    geminiSeverity.className    = `severity-value ${severityClass}`;

    geminiActions.innerHTML = actions
        .slice(0, 3)
        .map(a => `<li>${a}</li>`)
        .join('');

    videoAnalyzingTag.style.display = 'none';
}

function handleDemoComplete() {
    document.querySelector('.spinner').style.display = 'none';
    overlayText.textContent = '✓ Demo Complete — All Incidents Logged';
    const sub = document.querySelector('.transition-sub');
    if (sub) sub.textContent = `${totalAlerts} alert${totalAlerts !== 1 ? 's' : ''} recorded`;
    overlay.classList.remove('hidden');
}

// ─────────────────────────────────────────────────────────────
// GAUGE UPDATES
// ─────────────────────────────────────────────────────────────
function updateGauge(type, value) {
    const clampedVal = Math.min(1, Math.max(0, value));
    const offset     = GAUGE_CIRCUM * (1 - clampedVal);
    const pct        = Math.round(clampedVal * 100);

    if (type === 'fall') {
        fallGaugeFill.style.strokeDashoffset = offset;
        fallPct.textContent = `${pct}%`;
        const isDanger = clampedVal >= FALL_THRESH;
        fallGaugeFill.classList.toggle('danger', isDanger);
    } else {
        seizureGaugeFill.style.strokeDashoffset = offset;
        seizurePct.textContent = `${pct}%`;
        const isDanger = clampedVal >= SZ_THRESH;
        seizureGaugeFill.classList.toggle('danger', isDanger);
    }
}

// ─────────────────────────────────────────────────────────────
// SYSTEM STATUS
// ─────────────────────────────────────────────────────────────
function setSystemStatus(type, text, classes) {
    statusText.textContent               = text;
    patientStatusBadge.className         = 'patient-status-badge ' + classes;
}

// ─────────────────────────────────────────────────────────────
// PATIENT PROFILE LOADER
// ─────────────────────────────────────────────────────────────
function loadPatientProfile(patientKey) {
    const profile = PATIENT_PROFILES[patientKey] || PATIENT_PROFILES['Patient A'];
    currentProfile = profile;
    currentHR      = profile.hr;

    patientName.textContent = profile.fullName;
    patientRoom.textContent = profile.room;
    patientAvatar.textContent = profile.avatar;
    if (cameraLabel) cameraLabel.textContent = profile.camera;

    hrValue.textContent   = profile.hr;
    spo2Value.textContent = profile.spo2;
    bpValue.textContent   = profile.bp;
    tempValue.textContent = profile.temp.toFixed(1);
}

// ─────────────────────────────────────────────────────────────
// VITALS SIMULATION (cosmetic ±2 HR variation)
// ─────────────────────────────────────────────────────────────
function startVitalsSimulation() {
    if (vitalsInterval) clearInterval(vitalsInterval);
    vitalsInterval = setInterval(() => {
        const delta = (Math.random() > 0.5 ? 1 : -1) * Math.round(Math.random() * 2);
        currentHR   = Math.max(currentProfile.hr - 4, Math.min(currentProfile.hr + 4, currentHR + delta));
        hrValue.textContent = currentHR;
    }, 2000);
}

// ─────────────────────────────────────────────────────────────
// LIVE CLOCK (top-right of video)
// ─────────────────────────────────────────────────────────────
function startClock() {
    if (clockInterval) clearInterval(clockInterval);
    const tick = () => {
        const now = new Date();
        videoTimestamp.textContent =
            String(now.getHours()).padStart(2, '0')   + ':' +
            String(now.getMinutes()).padStart(2, '0') + ':' +
            String(now.getSeconds()).padStart(2, '0');
    };
    tick();
    clockInterval = setInterval(tick, 1000);
}

// ─────────────────────────────────────────────────────────────
// GEMINI STATE MACHINE
// ─────────────────────────────────────────────────────────────
function showGeminiState(state) {
    geminiIdleState.classList.add('hidden');
    geminiAnalyzing.classList.add('hidden');
    geminiResult.classList.add('hidden');

    if (state === 'idle')      geminiIdleState.classList.remove('hidden');
    else if (state === 'analyzing') geminiAnalyzing.classList.remove('hidden');
    else if (state === 'result')    geminiResult.classList.remove('hidden');
}

// ─────────────────────────────────────────────────────────────
// SEGMENT TYPE PILL
// ─────────────────────────────────────────────────────────────
function updateSegmentTypePill(type) {
    segmentTypePill.className = 'segment-type-pill';
    if (type === 'fall')    { segmentTypePill.textContent = 'FALL EVENT';    segmentTypePill.classList.add('type-fall'); }
    else if (type === 'seizure') { segmentTypePill.textContent = 'SEIZURE EPISODE'; segmentTypePill.classList.add('type-seizure'); }
    else                         { segmentTypePill.textContent = 'NORMAL ACTIVITY'; }
}

// ─────────────────────────────────────────────────────────────
// ALERT COUNT UPDATE (called when alert cards are dismissed)
// ─────────────────────────────────────────────────────────────
function updateAlertCount() {
    const remaining = alertFeed.querySelectorAll('.alert-card').length;
    if (remaining === 0 && emptyAlerts) {
        emptyAlerts.style.display = '';
    }
}

// ─────────────────────────────────────────────────────────────
// BUTTON EVENTS
// ─────────────────────────────────────────────────────────────
btnResume.addEventListener('click', () => {
    controlsOverlay.classList.add('hidden');
    isManualPaused = false;
    btnManualPause.textContent = '⏸ Pause';
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'resume' }));
});

btnSkip.addEventListener('click', () => {
    controlsOverlay.classList.add('hidden');
    overlayText.textContent = 'Skipping to next patient…';
    overlay.classList.remove('hidden');
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'skip' }));
});

btnManualPause.addEventListener('click', () => {
    isManualPaused = !isManualPaused;
    if (isManualPaused) {
        btnManualPause.textContent = '▶ Resume';
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'pause' }));
    } else {
        btnManualPause.textContent = '⏸ Pause';
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'resume' }));
    }
});

btnManualSkip.addEventListener('click', () => {
    overlayText.textContent = 'Skipping to next patient…';
    overlay.classList.remove('hidden');
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'skip' }));
});
