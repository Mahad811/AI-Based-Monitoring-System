const token = localStorage.getItem('vg_token');

async function loadNurses() {
    try {
        const res = await fetch('/api/admin/nurses', {
            headers: { 'Authorization': `Bearer ${token}` } 
        });
        const data = await res.json();
        
        const tbody = document.getElementById('roster-body');
        tbody.innerHTML = '';
        data.forEach(n => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="color:#8b9ab7;">#${n.id}</td>
                <td style="color:#dca54c;">${n.staff_id}</td>
                <td>${n.name}</td>
                <td>
                    <button class="btn-del" onclick="deleteNurse(${n.id})" ${n.staff_id === 'admin' ? 'disabled' : ''}>Remove</button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error('Error loading PostgreSQL roster:', e);
    }
}

async function deleteNurse(id) {
    if (!confirm('Are you absolutely sure you want to remove this staff profile from the database?')) return;
    try {
        const res = await fetch(`/api/admin/nurses/${id}`, { method: 'DELETE' });
        if (res.ok) {
            loadNurses();
        } else {
            alert('Failed to delete.');
        }
    } catch (e) {
        alert(e.message);
    }
}

document.getElementById('add-nurse-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const sf = document.getElementById('add-id').value;
    const nm = document.getElementById('add-name').value;
    const pw = document.getElementById('add-pass').value;
    
    try {
        const res = await fetch('/api/admin/nurses', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ staff_id: sf, name: nm, password: pw })
        });
        
        if (res.ok) {
            document.getElementById('add-nurse-form').reset();
            loadNurses();
        } else {
            const data = await res.json();
            alert(data.detail || 'Failed to add nurse');
        }
    } catch (e) {
        alert(e.message);
    }
});

// Load the table on startup
loadNurses();
