// Constants for thresholds (sync with backend)
const FALL_THRESH = 0.55;
const SZ_THRESH = 0.48;

// DOM Elements
const videoStream = document.getElementById('video-stream');
const overlay = document.getElementById('transition-overlay');
const overlayText = document.getElementById('transition-text');
const fpsCounter = document.getElementById('fps-counter');

const controlsOverlay = document.getElementById('controls-overlay');
const controlStatus = document.getElementById('control-status');
const btnResume = document.getElementById('btn-resume');
const btnSkip = document.getElementById('btn-skip');

const btnManualPause = document.getElementById('btn-manual-pause');
const btnManualSkip = document.getElementById('btn-manual-skip');
let isManualPaused = false;

const patientName = document.getElementById('patient-name');
const segmentBadge = document.getElementById('segment-progress');
const segmentTitle = document.getElementById('segment-title');

const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');

const alertFeed = document.getElementById('alert-feed');
const emptyAlerts = document.getElementById('empty-alerts');

// State
let currentStatus = "SYSTEM STANDBY";
let statusClass = "";

// Initialize WebSocket
const ws = new WebSocket(`ws://${window.location.host}/stream`);

ws.onopen = () => {
    console.log("WebSocket connected. Waiting for stream...");
};

ws.onclose = () => {
    console.log("WebSocket disconnected.");
    patientName.textContent = "CONNECTION LOST";
    fpsCounter.textContent = "0 FPS";
};

ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);

    if (data.type === 'segment_start') {
        // Hide transition, update headers
        overlay.classList.add('hidden');
        controlsOverlay.classList.add('hidden');
        patientName.textContent = data.patient;
        segmentBadge.textContent = `[${data.progress}]`;
        segmentTitle.textContent = data.label;
        
        // Reset status to normal if switching patient
        setStatus("NORMAL", "MONITORING", "");
    }
    else if (data.type === 'transition') {
        // Show transition spinner
        overlayText.textContent = data.message;
        overlay.classList.remove('hidden');
    }
    else if (data.type === 'frame_update') {
        // Update Video Base64 Image
        videoStream.src = `data:image/jpeg;base64,${data.frame_b64}`;
        
        // Update FPS
        fpsCounter.textContent = `${data.fps} FPS`;
        fpsCounter.className = 'fps ' + (data.fps > 20 ? 'fps-fast' : data.fps > 10 ? 'fps-med' : 'fps-slow');
        
        // Handle simultaneous alerts pushed with frame (instant Reflex)
        if (data.alert) {
            handleInstantAlert(data.alert);
        }
    }
    else if (data.type === 'gemini_report') {
        // Upgrade the alert card with AI verification
        handleGeminiReport(data.alert_id, data.report);
    }
    else if (data.type === 'demo_complete') {
        overlayText.textContent = 'Demo Complete. Alerts Logged.';
        document.querySelector('.spinner').style.display = 'none';
        overlay.classList.remove('hidden');
    }
};

function setStatus(type, text, className) {
    statusText.textContent = text;
    statusIndicator.className = "status-indicator " + className;
}

function handleInstantAlert(alert_data) {
    const { alert_id, event_type, confidence, timestamp } = alert_data;
    
    // Hide empty state
    if (emptyAlerts) emptyAlerts.style.display = 'none';

    // Update global status
    if (event_type === 'seizure') {
        setStatus("SEIZURE", "SEIZURE ALERT", "status-seizure");
    } else if (event_type === 'fall' && currentStatus !== "SEIZURE") {
        setStatus("FALL", "FALL ALERT", "status-fall");
    }

    // Create the card
    const cardTitle = event_type === 'seizure' ? "SEIZURE DETECTED" : "FALL DETECTED";
    const cardClass = event_type === 'seizure' ? 'seizure' : 'fall';
    const confStr = `${Math.round(confidence * 100)}%`;

    const el = document.createElement('div');
    el.className = `alert-card ${cardClass}`;
    el.id = `alert-card-${alert_id}`;
    
    el.innerHTML = `
        <div class="alert-header">
            <span class="alert-title">${cardTitle} (${confStr})</span>
            <div>
                <span class="alert-time">${timestamp}</span>
                <button onclick="this.closest('.alert-card').remove()" style="background:none; border:none; color:var(--text-muted); font-size: 1.2rem; cursor:pointer; margin-left:8px; line-height: 1;">&times;</button>
            </div>
        </div>
        <div class="gemini-box analyzing" id="gemini-box-${alert_id}">
            <div class="gemini-pulse"></div>
            <span>Cognitive Core Cross-Validating Signals...</span>
        </div>
    `;

    // Append to top of feed
    alertFeed.insertBefore(el, alertFeed.firstChild);
}

function handleGeminiReport(alert_id, report) {
    const box = document.getElementById(`gemini-box-${alert_id}`);
    if (!box) return;

    box.className = 'gemini-box'; // remove 'analyzing'
    
    // Build actions list
    const actionsHtml = report.actions.map(a => `<li>${a}</li>`).join('');
    
    box.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div class="gemini-headline">🧠 ${report.headline}</div>
            <div class="decision-badge decision-${report.decision}">${report.decision}</div>
        </div>
        <div class="gemini-narrative"><strong>Cognitive Core Analysis:</strong> ${report.narrative}</div>
        <ul class="gemini-actions">
            ${actionsHtml}
        </ul>
    `;
    
    // If Gemini suppressed it, dim the alert card
    if (report.decision === 'SUPPRESSED') {
        const card = document.getElementById(`alert-card-${alert_id}`);
        card.style.opacity = '0.5';
        card.style.borderLeftColor = 'var(--text-muted)';
        card.querySelector('.alert-title').style.color = 'var(--text-muted)';
    }
}

// Button Events
btnResume.addEventListener('click', () => {
    controlsOverlay.classList.add('hidden');
    
    // Reset manual pause button state if it was paused
    if (isManualPaused) {
        isManualPaused = false;
        btnManualPause.textContent = "⏸ Pause";
    }

    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'resume' }));
    }
});

btnSkip.addEventListener('click', () => {
    controlsOverlay.classList.add('hidden');
    overlayText.textContent = "Skipping to next patient...";
    overlay.classList.remove('hidden');
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'skip' }));
    }
});

function pauseVideoForVerification() {
    controlsOverlay.classList.remove('hidden');
}

btnManualPause.addEventListener('click', () => {
    isManualPaused = !isManualPaused;
    if (isManualPaused) {
        btnManualPause.textContent = "▶ Resume";
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'pause' }));
    } else {
        btnManualPause.textContent = "⏸ Pause";
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'resume' }));
    }
});

btnManualSkip.addEventListener('click', () => {
    overlayText.textContent = "Skipping to next patient...";
    overlay.classList.remove('hidden');
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'skip' }));
});
