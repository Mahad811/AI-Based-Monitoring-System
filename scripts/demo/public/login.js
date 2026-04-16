document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const user = document.getElementById('username').value;
    const pass = document.getElementById('password').value;
    const btn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const loader = document.getElementById('btn-loader');
    const errBox = document.getElementById('error-msg');
    
    // UI Loading state
    btn.disabled = true;
    loader.classList.remove('hidden');
    errBox.classList.add('hidden');
    
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
        
        // Success
        localStorage.setItem('vg_token', data.token);
        localStorage.setItem('vg_nurse_name', data.nurse_name);
        localStorage.setItem('vg_staff_id', data.staff_id);
        
        btnText.textContent = 'Auth Successful';
        btnText.style.color = '#000';
        btn.disabled = false; // re-enable just to show text
        btn.style.background = '#dca54c'; 
        loader.classList.add('hidden');
        
        setTimeout(() => {
            window.location.href = '/static/hub.html';
        }, 600);
        
    } catch (err) {
        errBox.textContent = err.message;
        errBox.classList.remove('hidden');
        btn.disabled = false;
        loader.classList.add('hidden');
        btnText.style.color = '';
    }
});
