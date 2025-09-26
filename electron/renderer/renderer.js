const inputEl = document.getElementById('in');
const pickInBtn = document.getElementById('pick-in');
const runBtn = document.getElementById('run');
const stopBtn = document.getElementById('stop');
const logEl = document.getElementById('log');
const progressEl = document.getElementById('progress');

function appendLog(line) {
    logEl.textContent += line;
    if (!line.endsWith('\n')) logEl.textContent += '\n';
    logEl.scrollTop = logEl.scrollHeight;
}

window.mdpi.onLog((msg) => {
    appendLog(msg);
    if (msg.includes('Pulling') || msg.includes('Downloading')) {
        // naive progress pulse
        const w = Math.min(100, (parseInt(progressEl.style.width || '0') || 0) + 2);
        progressEl.style.width = w + '%';
    }
});

window.mdpi.onCompleted(({ code }) => {
    appendLog(`\nProcess finished with exit code ${code}`);
    progressEl.style.width = '100%';
    setTimeout(() => progressEl.style.width = '0%', 1500);
    alert(code === 0 ? 'Processing completed successfully.' : 'Processing failed. See logs.');
});

pickInBtn.addEventListener('click', async () => {
    const p = await window.mdpi.pickFolder('Select input folder');
    if (p) inputEl.value = p;
});

runBtn.addEventListener('click', async () => {
    logEl.textContent = '';
    progressEl.style.width = '0%';
    const inputPath = inputEl.value.trim();
    if (!inputPath) {
        alert('Please select an input folder.');
        return;
    }
    const res = await window.mdpi.run(inputPath, {});
    if (!res?.ok) {
        appendLog(String(res?.error || 'Failed to start'));
    }
});

stopBtn.addEventListener('click', async () => {
    await window.mdpi.stop();
});


