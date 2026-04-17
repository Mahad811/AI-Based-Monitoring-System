'use strict';
/* ═══════════════════════════════════════════════════════════════
   Admin Panel — Tab controllers
   Tabs: Command Center | Staff Roster | Activity Log
═══════════════════════════════════════════════════════════════ */

// ── Tab routing ───────────────────────────────────────────────
const TABS = ['command', 'roster', 'audit'];

function showTab(id) {
  TABS.forEach(t => {
    document.getElementById(`tab-${t}`)?.classList.toggle('active', t === id);
    document.querySelector(`.sb-tab[data-tab="${t}"]`)?.classList.toggle('active', t === id);
  });
  location.hash = id;
  if (id === 'command') loadCommandCenter();
  if (id === 'roster')  loadRoster();
  if (id === 'audit')   { _auditOffset = 0; loadAudit(0); }
}

document.querySelectorAll('.sb-tab, .sb-tab-link').forEach(btn => {
  btn.addEventListener('click', () => showTab(btn.dataset.tab));
});

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Sidebar profile
  const name    = localStorage.getItem('vg_nurse_name') || 'Admin';
  const role    = localStorage.getItem('vg_role') || 'Admin';
  const av      = VG.avatar(name);
  const sbAv    = document.getElementById('sb-avatar');
  if (sbAv) { sbAv.textContent = av.initials; sbAv.style.background = av.bgColor; }
  const sbName = document.getElementById('sb-nurse-name');
  if (sbName) sbName.textContent = name;
  const sbRole = document.getElementById('sb-nurse-role');
  if (sbRole) sbRole.textContent = role;

  // Start on hashed tab or default to command
  const hash = location.hash.replace('#', '') || 'command';
  showTab(TABS.includes(hash) ? hash : 'command');
});

function doLogout() {
  localStorage.clear();
  window.location.href = '/static/login.html';
}

// ════════════════════════════════════════════════════════════
// TAB 1: COMMAND CENTER
// ════════════════════════════════════════════════════════════
let _healthInterval = null;

async function loadCommandCenter() {
  await Promise.all([loadStats(), refreshHealth()]);
  if (_healthInterval) clearInterval(_healthInterval);
  _healthInterval = setInterval(refreshHealth, 10000);
}

async function loadStats() {
  try {
    const data = await VG.fetchJSON('/api/admin/stats');

    // ── Stat tiles ───────────────────────────────────────────
    const delta = data.alerts_today - (data.alerts_yesterday || 0);
    const deltaHtml = delta > 0
      ? `<span class="vg-stat-delta-up">▲ ${delta} vs yesterday</span>`
      : delta < 0
      ? `<span class="vg-stat-delta-down">▼ ${Math.abs(delta)} vs yesterday</span>`
      : `<span class="vg-stat-delta-neu">Same as yesterday</span>`;

    const verifyTxt = data.avg_verify_time_sec != null
      ? `${data.avg_verify_time_sec}s` : '—';

    _tile('tile-patients', 'PATIENTS ACTIVE',    data.patients_active, 'vg-stat-blue',  `${data.alerts_week} alerts this week`);
    _tile('tile-alerts',   'ALERTS TODAY',        data.alerts_today,    'vg-stat-red',   deltaHtml, true);
    _tile('tile-staff',    'ON-DUTY STAFF',       data.on_duty_count,   'vg-stat-green', 'Currently monitoring');
    _tile('tile-verify',   'AVG VERIFY TIME',     verifyTxt,            'vg-stat-gold',  'Gemini Tier-3 latency');

    // ── Sparkline ─────────────────────────────────────────────
    const hourly = data.hourly_alerts || [];
    const canvas = document.getElementById('cc-sparkline');
    if (canvas) {
      requestAnimationFrame(() => VG.sparkline(canvas, hourly));
    }
    // Hour labels: 0, 6, 12, 18, 23
    const hoursEl = document.getElementById('spark-hours');
    if (hoursEl) {
      hoursEl.innerHTML = ['00:00','06:00','12:00','18:00','23:00']
        .map(h => `<span>${h}</span>`).join('');
    }
    const totalEl = document.getElementById('cc-total-today');
    if (totalEl) totalEl.textContent = `${data.alerts_today} total`;

    // ── Recent Activity ───────────────────────────────────────
    const feed = document.getElementById('cc-activity');
    if (feed) {
      feed.innerHTML = '';
      if (!data.recent_activity?.length) {
        feed.innerHTML = '<div style="color:var(--vg-muted);font-size:0.82rem;padding:12px;">No recent activity.</div>';
      } else {
        data.recent_activity.forEach(a => {
          feed.appendChild(_activityItem(a));
        });
      }
    }
  } catch (e) { /* toast shown by fetchJSON */ }
}

function _tile(id, label, value, colorClass, subHtml, rawSub = false) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('vg-skeleton', 'vg-skeleton-card');
  el.innerHTML = `
    <div class="vg-stat-label">${label}</div>
    <div class="vg-stat-value ${colorClass}">${value ?? '—'}</div>
    <div class="vg-stat-sub">${rawSub ? subHtml : (subHtml || '')}</div>`;
}

async function refreshHealth() {
  const strip = document.getElementById('cc-health-strip');
  if (!strip) return;
  try {
    const h = await VG.fetchJSON('/api/admin/health');
    strip.innerHTML = `
      <div class="vg-health-item">
        <div class="vg-health-dot ${h.postgres}"></div>
        <span>PostgreSQL — <strong>${h.postgres === 'ok' ? 'Connected' : 'Down'}</strong></span>
      </div>
      <div class="vg-health-item">
        <div class="vg-health-dot ${h.kaggle}"></div>
        <span>Kaggle Endpoint — <strong>${_kaggleLabel(h.kaggle)}</strong></span>
      </div>
      <div class="vg-health-item">
        <div class="vg-health-dot ok"></div>
        <span>WebSocket Clients — <strong>${h.ws_clients}</strong></span>
      </div>
      <div class="vg-health-item" style="margin-top:4px;">
        <span class="vg-pill vg-pill-nurse" style="font-size:0.6rem;">${h.inference_mode}</span>
        <span style="font-size:0.72rem;">Inference mode</span>
      </div>`;
  } catch (_) {}
}

function _kaggleLabel(s) {
  return s === 'ok' ? 'Reachable' : s === 'slow' ? 'Slow (>1.5s)' : s === 'disabled' ? 'Disabled' : 'Down';
}

// ════════════════════════════════════════════════════════════
// TAB 2: STAFF ROSTER
// ════════════════════════════════════════════════════════════
let _nurses = [];

async function loadRoster() {
  const grid = document.getElementById('roster-grid');
  if (!grid) return;
  grid.innerHTML = '<div class="vg-skeleton vg-skeleton-card"></div>'.repeat(3);
  try {
    _nurses = await VG.fetchJSON('/api/admin/nurses');
    renderRoster(_nurses);
  } catch (_) {}
}

function renderRoster(nurses) {
  const grid = document.getElementById('roster-grid');
  if (!grid) return;
  grid.innerHTML = '';
  if (!nurses.length) {
    grid.innerHTML = '<p style="color:var(--vg-muted);">No staff profiles found.</p>';
    return;
  }
  nurses.forEach(n => grid.appendChild(_profileCard(n)));
}

function _profileCard(n) {
  const av  = VG.avatar(n.name);
  const div = document.createElement('div');
  div.className = 'vg-profile-card';
  const isAdmin = n.staff_id === 'admin';
  div.innerHTML = `
    <div class="vg-profile-top">
      <div class="vg-avatar" style="background:${av.bgColor};">${av.initials}</div>
      <div style="flex:1;min-width:0;">
        <div class="vg-profile-name">${n.name}</div>
        <div class="vg-profile-id">${n.staff_id}</div>
      </div>
    </div>
    <div class="vg-profile-pills">
      ${VG.rolePill(n.role)}
      ${VG.shiftPill(n.shift)}
      ${n.ward_assignment ? `<span class="vg-pill vg-pill-unknown">${n.ward_assignment}</span>` : ''}
      ${VG.statusDot(n.status)}
    </div>
    <div class="vg-profile-stats">
      <div><span class="vg-profile-stat-val">${n.alerts_handled || 0}</span><br>Alerts handled</div>
      <div><span class="vg-profile-stat-val">${VG.relTime(n.last_login)}</span><br>Last login</div>
      <div><span class="vg-profile-stat-val">${n.join_date ? new Date(n.join_date).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'}) : '—'}</span><br>Joined</div>
    </div>
    <div class="vg-profile-actions">
      <button class="vg-btn vg-btn-sm" onclick="openEditModal(${n.id})">✏ Edit</button>
      <button class="vg-btn vg-btn-sm" style="color:var(--vg-red);border-color:rgba(239,68,68,0.3);"
        onclick="removeNurse(${n.id},'${escHtml(n.name)}')" ${isAdmin ? 'disabled' : ''}>✕ Remove</button>
    </div>`;
  return div;
}

function escHtml(s) { return (s||'').replace(/'/g, "\\'"); }

// ── Add / Edit modal ─────────────────────────────────────────
function openAddModal() {
  document.getElementById('nurse-edit-id').value = '';
  document.getElementById('nurse-modal-title').textContent = 'Add New Staff';
  document.getElementById('nf-staffid').value  = '';
  document.getElementById('nf-staffid').disabled = false;
  document.getElementById('nf-name').value     = '';
  document.getElementById('nf-password').value = '';
  document.getElementById('nf-password').placeholder = 'Required for new staff';
  document.getElementById('nf-role').value     = 'Nurse';
  document.getElementById('nf-shift').value    = 'Morning';
  document.getElementById('nf-ward').value     = '';
  document.getElementById('nf-status').value   = 'off-duty';
  document.getElementById('nurse-modal-backdrop').style.display = 'flex';
}

function openEditModal(id) {
  const n = _nurses.find(x => x.id === id);
  if (!n) return;
  document.getElementById('nurse-edit-id').value = id;
  document.getElementById('nurse-modal-title').textContent = `Edit — ${n.name}`;
  document.getElementById('nf-staffid').value  = n.staff_id;
  document.getElementById('nf-staffid').disabled = true;
  document.getElementById('nf-name').value     = n.name;
  document.getElementById('nf-password').value = '';
  document.getElementById('nf-password').placeholder = 'Leave blank to keep current password';
  document.getElementById('nf-role').value     = n.role || 'Nurse';
  document.getElementById('nf-shift').value    = n.shift || 'Morning';
  document.getElementById('nf-ward').value     = n.ward_assignment || '';
  document.getElementById('nf-status').value   = n.status || 'off-duty';
  document.getElementById('nurse-modal-backdrop').style.display = 'flex';
}

function closeNurseModal() {
  document.getElementById('nurse-modal-backdrop').style.display = 'none';
}

function handleModalBackdropClick(e) {
  if (e.target.id === 'nurse-modal-backdrop') closeNurseModal();
}

async function submitNurseForm() {
  const editId = document.getElementById('nurse-edit-id').value;
  const name   = document.getElementById('nf-name').value.trim();
  const pw     = document.getElementById('nf-password').value;
  const role   = document.getElementById('nf-role').value;
  const shift  = document.getElementById('nf-shift').value;
  const ward   = document.getElementById('nf-ward').value.trim();
  const status = document.getElementById('nf-status').value;

  if (!name) { VG.toast('Full name is required', 'error'); return; }

  try {
    if (editId) {
      // Edit mode
      const body = { name, role, shift, ward_assignment: ward, status };
      if (pw) body.password = pw;
      await VG.fetchJSON(`/api/admin/nurses/${editId}`, {
        method: 'PUT',
        body: JSON.stringify(body),
      });
      VG.toast(`${name} updated successfully`, 'success');
    } else {
      // Add mode
      const staffId = document.getElementById('nf-staffid').value.trim();
      if (!staffId) { VG.toast('Staff ID is required', 'error'); return; }
      if (!pw)      { VG.toast('Password is required for new staff', 'error'); return; }
      await VG.fetchJSON('/api/admin/nurses', {
        method: 'POST',
        body: JSON.stringify({ staff_id: staffId, name, password: pw, role, shift, ward_assignment: ward, status }),
      });
      VG.toast(`${name} added to roster`, 'success');
    }
    closeNurseModal();
    await loadRoster();
  } catch (_) { /* toast already shown */ }
}

async function removeNurse(id, name) {
  const ok = await VG.modal.confirm(
    'Remove Staff Member',
    `Are you sure you want to remove <strong>${name}</strong> from the roster? This cannot be undone.`,
    { confirmLabel: 'Remove', danger: true }
  );
  if (!ok) return;
  try {
    await VG.fetchJSON(`/api/admin/nurses/${id}`, { method: 'DELETE' });
    VG.toast(`${name} removed from roster`, 'success');
    await loadRoster();
  } catch (_) {}
}

// ════════════════════════════════════════════════════════════
// TAB 3: ACTIVITY LOG
// ════════════════════════════════════════════════════════════
let _auditOffset   = 0;
let _auditFilter   = 'all';
const AUDIT_LIMIT  = 50;

function setAuditFilter(btn, action) {
  document.querySelectorAll('.vg-filter-chip').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  _auditFilter = action;
  _auditOffset = 0;
  loadAudit(0);
}

async function loadAudit(offset = 0) {
  _auditOffset = offset;
  const feed = document.getElementById('audit-feed');
  if (!feed) return;
  if (offset === 0) {
    feed.innerHTML = '<div class="vg-skeleton vg-skeleton-row"></div>'.repeat(5);
  }

  const actor = document.getElementById('audit-actor-filter')?.value || '';
  const action = _auditFilter === 'all' ? '' : _auditFilter;

  let url = `/api/admin/audit?limit=${AUDIT_LIMIT}&offset=${offset}`;
  if (action) url += `&action=${action}`;
  if (actor)  url += `&actor_id=${encodeURIComponent(actor)}`;

  try {
    const data = await VG.fetchJSON(url);

    // Populate actor dropdown on first load
    if (offset === 0 && _nurses.length > 0) {
      const sel = document.getElementById('audit-actor-filter');
      if (sel && sel.options.length === 1) {
        _nurses.forEach(n => {
          const opt = new Option(n.name, n.staff_id);
          sel.add(opt);
        });
        const sysOpt = new Option('AI Pipeline', 'system');
        sel.add(sysOpt);
      }
    }

    if (offset === 0) feed.innerHTML = '';
    if (!data.items?.length && offset === 0) {
      feed.innerHTML = '<div style="color:var(--vg-muted);font-size:0.82rem;padding:12px;">No activity matching current filters.</div>';
    } else {
      data.items.forEach(a => feed.appendChild(_activityItem(a)));
    }

    const loadMoreBtn = document.getElementById('audit-loadmore');
    if (loadMoreBtn) {
      loadMoreBtn.style.display = (offset + AUDIT_LIMIT < data.total) ? 'block' : 'none';
    }
  } catch (_) {}
}

function loadMoreAudit() {
  loadAudit(_auditOffset + AUDIT_LIMIT);
}

// ── Activity item renderer ───────────────────────────────────
const ACTION_META = {
  LOGIN:           { icon: '🔑', color: 'rgba(56,189,248,0.15)',  label: 'Login' },
  NURSE_ADDED:     { icon: '➕', color: 'rgba(52,211,153,0.15)',  label: 'Staff Added' },
  NURSE_UPDATED:   { icon: '✏',  color: 'rgba(220,165,76,0.15)', label: 'Staff Updated' },
  NURSE_REMOVED:   { icon: '✕',  color: 'rgba(239,68,68,0.15)',  label: 'Staff Removed' },
  ALERT_CONFIRMED: { icon: '⚡',  color: 'rgba(239,68,68,0.15)',  label: 'Alert Confirmed' },
  ALERT_SUPPRESSED:{ icon: '⚬',  color: 'rgba(139,154,183,0.1)', label: 'Alert Suppressed' },
};

function _activityItem(a) {
  const meta = ACTION_META[a.action] || { icon: '●', color: 'rgba(139,154,183,0.1)', label: a.action };
  const div = document.createElement('div');
  div.className = 'vg-activity-item';
  div.innerHTML = `
    <div class="vg-activity-icon" style="background:${meta.color};">${meta.icon}</div>
    <div class="vg-activity-content">
      <div class="vg-activity-line">
        <strong>${a.actor_name || a.actor_id}</strong>
        — ${meta.label}${a.details ? ` · <span style="color:var(--vg-muted)">${a.details}</span>` : ''}
      </div>
      <div class="vg-activity-time">${VG.relTime(a.timestamp)}</div>
    </div>`;
  return div;
}
