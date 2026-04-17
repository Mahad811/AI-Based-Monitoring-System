'use strict';
/* ═══════════════════════════════════════════════════════════════
   VG-UI — Vital Guardian Shared UI Utilities
   Usage: window.VG.toast / VG.modal / VG.relTime / VG.avatar / etc.
═══════════════════════════════════════════════════════════════ */

(function () {
  // ── Toast container ────────────────────────────────────────────
  let _toastContainer = null;
  function _ensureToastContainer() {
    if (_toastContainer) return;
    _toastContainer = document.createElement('div');
    _toastContainer.id = 'vg-toast-container';
    document.body.appendChild(_toastContainer);
  }

  const TOAST_ICONS = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' };

  function toast(msg, type = 'info', duration = 3200) {
    _ensureToastContainer();
    const el = document.createElement('div');
    el.className = `vg-toast ${type}`;
    el.innerHTML = `<span class="vg-toast-icon">${TOAST_ICONS[type] || 'ℹ'}</span>
                    <span class="vg-toast-msg">${msg}</span>`;
    _toastContainer.appendChild(el);
    const dismiss = () => {
      el.classList.add('vg-toast-out');
      el.addEventListener('animationend', () => el.remove(), { once: true });
    };
    const tid = setTimeout(dismiss, duration);
    el.addEventListener('click', () => { clearTimeout(tid); dismiss(); });
  }

  // ── Modal ─────────────────────────────────────────────────────
  const modal = {
    /**
     * Shows a confirmation dialog. Returns Promise<boolean>.
     * confirmLabel / cancelLabel are optional.
     */
    confirm(title, body = '', { confirmLabel = 'Confirm', cancelLabel = 'Cancel', danger = false } = {}) {
      return new Promise(resolve => {
        const backdrop = document.createElement('div');
        backdrop.className = 'vg-modal-backdrop';
        backdrop.innerHTML = `
          <div class="vg-modal" role="dialog" aria-modal="true">
            <div class="vg-modal-title">${title}</div>
            ${body ? `<div class="vg-modal-body">${body}</div>` : ''}
            <div class="vg-modal-actions">
              <button class="vg-btn" id="vg-modal-cancel">${cancelLabel}</button>
              <button class="vg-btn vg-btn-primary${danger ? ' vg-btn-danger' : ''}" id="vg-modal-ok">${confirmLabel}</button>
            </div>
          </div>`;
        document.body.appendChild(backdrop);
        const close = (result) => {
          backdrop.classList.add('vg-hide');
          backdrop.addEventListener('animationend', () => backdrop.remove(), { once: true });
          resolve(result);
        };
        backdrop.querySelector('#vg-modal-ok').onclick     = () => close(true);
        backdrop.querySelector('#vg-modal-cancel').onclick = () => close(false);
        backdrop.addEventListener('click', e => { if (e.target === backdrop) close(false); });
        // Focus OK button
        setTimeout(() => backdrop.querySelector('#vg-modal-ok')?.focus(), 50);
      });
    },

    /**
     * Opens a generic named modal element by id (must already be in the DOM).
     */
    open(id) {
      const el = document.getElementById(id);
      if (el) { el.style.display = 'flex'; el.classList.remove('vg-hide'); }
    },

    close(id) {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    },
  };

  // ── Drawer ────────────────────────────────────────────────────
  const drawer = {
    open(drawerId, backdropId) {
      document.getElementById(drawerId)?.classList.add('open');
      document.getElementById(backdropId)?.classList.add('open');
    },
    close(drawerId, backdropId) {
      document.getElementById(drawerId)?.classList.remove('open');
      document.getElementById(backdropId)?.classList.remove('open');
    },
  };

  // ── Relative time ─────────────────────────────────────────────
  function relTime(iso) {
    if (!iso) return 'Never';
    const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
    if (diff < 5)   return 'Just now';
    if (diff < 60)  return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  }

  // ── Avatar ────────────────────────────────────────────────────
  const AVATAR_COLORS = [
    '#7c3aed','#2563eb','#0891b2','#059669',
    '#d97706','#dc2626','#db2777','#4f46e5',
  ];
  function avatar(name) {
    const parts = (name || '?').trim().split(/\s+/);
    const initials = parts.length >= 2
      ? (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
      : name.slice(0, 2).toUpperCase();
    let hash = 0;
    for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
    const bgColor = AVATAR_COLORS[hash % AVATAR_COLORS.length];
    return { initials, bgColor };
  }

  // ── Sparkline ────────────────────────────────────────────────
  function sparkline(canvas, values, { color = '#dca54c', fillColor = 'rgba(220,165,76,0.1)', height = 60 } = {}) {
    if (!canvas) return;
    const ctx   = canvas.getContext('2d');
    const W     = canvas.offsetWidth || 300;
    const H     = height;
    canvas.width  = W;
    canvas.height = H;
    ctx.clearRect(0, 0, W, H);
    if (!values || values.length === 0) return;
    const max = Math.max(...values, 1);
    const step = W / (values.length - 1 || 1);
    const toX = i => i * step;
    const toY = v => H - 6 - ((v / max) * (H - 12));

    // Fill
    ctx.beginPath();
    ctx.moveTo(toX(0), H);
    values.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
    ctx.lineTo(toX(values.length - 1), H);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();

    // Line
    ctx.beginPath();
    values.forEach((v, i) => i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v)));
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.lineJoin    = 'round';
    ctx.stroke();

    // Dot at last value
    const lx = toX(values.length - 1), ly = toY(values[values.length - 1]);
    ctx.beginPath();
    ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  // ── fetchJSON ────────────────────────────────────────────────
  async function fetchJSON(url, opts = {}) {
    const token = localStorage.getItem('vg_token');
    const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    try {
      const res  = await fetch(url, { ...opts, headers });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = data.detail || `Request failed (${res.status})`;
        toast(msg, 'error');
        throw new Error(msg);
      }
      return data;
    } catch (err) {
      if (!err.message.includes('Request failed')) toast(err.message, 'error');
      throw err;
    }
  }

  // ── Pill helpers ─────────────────────────────────────────────
  function rolePill(role) {
    const map = {
      'Admin':    'vg-pill-admin',
      'ICU Lead': 'vg-pill-iculous',
      'Nurse':    'vg-pill-nurse',
      'Observer': 'vg-pill-observer',
    };
    return `<span class="vg-pill ${map[role] || 'vg-pill-observer'}">${role || 'Nurse'}</span>`;
  }
  function shiftPill(shift) {
    const map = { Morning: 'vg-pill-morning', Evening: 'vg-pill-evening', Night: 'vg-pill-night' };
    return `<span class="vg-pill ${map[shift] || 'vg-pill-morning'}">${shift || 'Morning'}</span>`;
  }
  function severityPill(sev) {
    const s = (sev || 'unknown').toLowerCase();
    return `<span class="vg-pill vg-pill-${s}">${s.toUpperCase()}</span>`;
  }
  function verdictPill(verdict) {
    if (!verdict) return `<span class="vg-pill vg-pill-pending">PENDING</span>`;
    const v = verdict.toLowerCase();
    if (v.includes('confirm'))  return `<span class="vg-pill vg-pill-confirmed">CONFIRMED</span>`;
    if (v.includes('suppress')) return `<span class="vg-pill vg-pill-suppressed">SUPPRESSED</span>`;
    return `<span class="vg-pill vg-pill-pending">PENDING</span>`;
  }
  function statusDot(status) {
    const s = (status || 'off-duty').toLowerCase().replace(' ', '-');
    const label = status || 'Off Duty';
    return `<span class="vg-status-dot ${s}">${label}</span>`;
  }

  // ── Export ───────────────────────────────────────────────────
  window.VG = {
    toast, modal, drawer,
    relTime, avatar, sparkline, fetchJSON,
    rolePill, shiftPill, severityPill, verdictPill, statusDot,
  };
})();
