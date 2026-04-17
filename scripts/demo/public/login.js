document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const user    = document.getElementById('username').value;
    const pass    = document.getElementById('password').value;
    const btn     = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const loader  = document.getElementById('btn-loader');
    const errBox  = document.getElementById('error-msg');

    btn.disabled = true;
    loader.classList.remove('hidden');
    errBox.classList.add('hidden');
    btnText.textContent = 'Authenticating…';

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: user, password: pass })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Authentication failed');
        }

        // Persist everything the rest of the app needs
        localStorage.setItem('vg_token',      data.token);
        localStorage.setItem('vg_nurse_name', data.nurse_name);
        localStorage.setItem('vg_staff_id',   data.staff_id);
        localStorage.setItem('vg_role',       data.role || 'Nurse');
        localStorage.setItem('vg_shift',      data.shift || 'Morning');
        localStorage.setItem('vg_ward',       data.ward || '');

        btnText.textContent = '✓ Auth Successful';
        btnText.style.color = '#000';
        btn.disabled = false;
        btn.style.background = '#dca54c';
        loader.classList.add('hidden');

        setTimeout(() => {
            window.location.href = '/static/hub.html';
        }, 650);

    } catch (err) {
        errBox.textContent = err.message;
        errBox.classList.remove('hidden');
        btn.disabled = false;
        loader.classList.add('hidden');
        btnText.textContent = 'Authenticate Context';
        btnText.style.color = '';
    }
});
