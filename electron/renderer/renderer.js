const runBtn = document.getElementById('run-btn');
const stopBtn = document.getElementById('stop-btn');
const logEl = document.getElementById('log-container');
const inputRowsContainer = document.getElementById('input-rows');
const addRowBtn = document.getElementById('add-row');

let inputIdCounter = 0;
const inputState = {}; // Store validation status for each input

function appendLog(line) {
    if (line.trim() === '[SEPARATOR]') {
        const separator = document.createElement('hr');
        logEl.appendChild(separator);
        logEl.scrollTop = logEl.scrollHeight;
        return;
    }

    const progressLine = document.getElementById('progress-line');
    const logLine = document.createElement('pre');
    logLine.textContent = line;

    if (progressLine) {
        // If a progress bar is visible, replace it with the new log line.
        // This ensures logs always appear after the last progress update.
        logEl.replaceChild(logLine, progressLine);
    } else {
        logEl.appendChild(logLine);
    }
    logEl.scrollTop = logEl.scrollHeight;
}

function updateRunButtonState() {
    const allValid = Object.values(inputState).length > 0 && Object.values(inputState).every(s => s.isValid);
    runBtn.disabled = !allValid;
}

async function validatePath(id, path) {
    if (!path) {
        inputState[id] = { isValid: false, validationResults: [[false, 'Path cannot be empty.']], metadata: null };
        updateRowValidation(id);
        updateRunButtonState();
        return;
    }

    // Call the validation method, but the primary update will be handled by the onValidate listener
    await window.mdpi.validatePath(path);
}

function createInputRow() {
    const id = `input-${inputIdCounter++}`;

    const container = document.createElement('div');
    container.className = 'input-row-container';
    container.id = `container-${id}`;

    const row = document.createElement('div');
    row.className = 'input-row';

    const pathInput = document.createElement('input');
    pathInput.type = 'text';
    pathInput.placeholder = '/path/to/raw/images';
    pathInput.id = `input-${id}`;

    const browseBtn = document.createElement('button');
    browseBtn.textContent = 'Browse';
    browseBtn.addEventListener('click', async () => {
        const p = await window.mdpi.pickFolder('Select input folder');
        if (p) {
            pathInput.value = p;
            validatePath(id, p);
        }
    });

    const removeBtn = document.createElement('button');
    removeBtn.textContent = 'Remove';
    removeBtn.addEventListener('click', () => {
        delete inputState[id];
        container.remove();
        updateRunButtonState();
    });

    pathInput.addEventListener('input', (e) => {
        validatePath(id, e.target.value);
    });

    const validationDiv = document.createElement('div');
    validationDiv.className = 'validation-result';
    validationDiv.id = `validation-${id}`;

    const metadataDiv = document.createElement('div');
    metadataDiv.className = 'metadata-grid';
    metadataDiv.id = `metadata-${id}`;


    row.appendChild(pathInput);
    row.appendChild(browseBtn);
    row.appendChild(removeBtn);
    container.appendChild(row);
    container.appendChild(validationDiv);
    container.appendChild(metadataDiv);
    inputRowsContainer.appendChild(container);

    // Validate the initial empty path
    validatePath(id, '');
}

function updateRowValidation(id) {
    const state = inputState[id];
    const validationDiv = document.getElementById(`validation-${id}`);
    const metadataDiv = document.getElementById(`metadata-${id}`);

    validationDiv.innerHTML = '';
    metadataDiv.innerHTML = '';

    if (state.validationResults) {
        state.validationResults.forEach(([isValid, message]) => {
            const p = document.createElement('p');
            p.textContent = `${isValid ? '✔' : '❌'} ${message}`;
            p.className = isValid ? 'valid' : 'invalid';
            validationDiv.appendChild(p);
        });
    }

    if (state.isValid && state.metadata) {
        const metadata = state.metadata;
        const keyOrder = ["recording_start", "image_shape", "camera_format"];
        const displayData = {
            "recording_start": `${metadata.recording_start_date} ${metadata.recording_start_time}`,
            "image_shape": `${metadata.image_width_pixels} x ${metadata.image_height_pixels} pixels`,
            "camera_format": metadata.camera_format || 'N/A'
        };

        keyOrder.forEach(key => {
            if (displayData[key]) {
                const keyEl = document.createElement('strong');
                keyEl.textContent = `${key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}:`;
                const valEl = document.createElement('span');
                valEl.textContent = displayData[key];
                metadataDiv.appendChild(keyEl);
                metadataDiv.appendChild(valEl);
            }
        });
    }
}

addRowBtn.addEventListener('click', createInputRow);

runBtn.addEventListener('click', async () => {
    logEl.textContent = '';
    const inputPaths = Object.keys(inputState)
        .map(id => document.getElementById(`input-${id}`).value.trim())
        .filter(Boolean);

    if (inputPaths.length === 0) {
        alert('Please add at least one valid input folder.');
        return;
    }

    const config = {
        capture_rate: parseFloat(document.getElementById('capture-rate').value),
        image_height_cm: parseFloat(document.getElementById('image-height').value),
        image_depth_cm: parseFloat(document.getElementById('image-depth').value),
        image_width_cm: parseFloat(document.getElementById('image-width').value),
    };

    runBtn.disabled = true;
    stopBtn.disabled = false;
    setControlsDisabled(true);

    const res = await window.mdpi.run(inputPaths, config);
    if (!res?.ok) {
        appendLog(String(res?.error || 'Failed to start'));
    }
});

stopBtn.addEventListener('click', async () => {
    await window.mdpi.stop();
});

window.mdpi.onLog((msg) => {
    appendLog(msg);
});

window.mdpi.onValidate((data) => {
    const { results, metadata } = data;
    const isValid = results.every(r => r[0]);

    // Find the input element that corresponds to the validated path
    const inputs = document.querySelectorAll('input[type="text"]');
    let targetId = null;
    for (const _input of inputs) {
        // This assumes the backend returns a path that can be matched, which might need adjustment
        // For now, let's assume we update the last focused or first invalid input.
        // A more robust solution might involve passing the input `id` with the validation request.
        if (Object.keys(inputState).length === 1) {
            targetId = Object.keys(inputState)[0];
            break;
        }
    }

    // Fallback or for single-input scenarios
    if (!targetId && Object.keys(inputState).length > 0) {
        targetId = Object.keys(inputState)[Object.keys(inputState).length - 1];
    }

    if (targetId) {
        inputState[targetId] = { isValid, validationResults: results, metadata };
        updateRowValidation(targetId);
        updateRunButtonState();
    }
});

window.mdpi.onCompleted(({
    code
}) => {
    if (code !== 'stopped') {
        const finalMessage = `\nProcess finished with exit code ${code}`;

        const progressLine = document.getElementById('progress-line');
        if (progressLine) {
            progressLine.removeAttribute('id'); // Convert to a normal line
            // Only append the final message if the progress bar was the last thing shown
            if (progressLine.textContent.includes('%')) {
                appendLog(finalMessage);
            }
        } else {
            appendLog(finalMessage);
        }
    }

    runBtn.disabled = false;
    stopBtn.disabled = true;
    setControlsDisabled(false);
    updateRunButtonState();
});

window.mdpi.onProgressUpdate(({ text }) => {
    let progressLine = document.getElementById('progress-line');
    if (!progressLine) {
        progressLine = document.createElement('pre');
        progressLine.id = 'progress-line';
        logEl.appendChild(progressLine);
    }
    progressLine.textContent = text;
    logEl.scrollTop = logEl.scrollHeight;
});

// Create the first input row on startup
createInputRow();

function setControlsDisabled(disabled) {
    const configCard = document.getElementById('config-card');
    const inputCard = document.getElementById('input-card');

    if (disabled) {
        configCard.classList.add('disabled');
        inputCard.classList.add('disabled');
    } else {
        configCard.classList.remove('disabled');
        inputCard.classList.remove('disabled');
    }

    const elementsToDisable = [
        ...configCard.querySelectorAll('input, button'),
        ...inputCard.querySelectorAll('input, button'),
    ];

    for (const el of elementsToDisable) {
        el.disabled = disabled;
    }
}

// --- Tooltip Dynamic Positioning ---
document.querySelectorAll('.help-tooltip').forEach(tooltipIcon => {
    const tooltipText = tooltipIcon.querySelector('.tooltip-text');

    tooltipIcon.addEventListener('mouseenter', () => {
        // Reset vertical position to default (above)
        tooltipText.classList.remove('tooltip-below');

        // Make it visible to calculate its dimensions
        tooltipText.style.visibility = 'visible';
        tooltipText.style.opacity = '1';

        let tooltipRect = tooltipText.getBoundingClientRect();

        // Check for vertical overflow (clipping at the top)
        if (tooltipRect.top < 0) {
            tooltipText.classList.add('tooltip-below');
            // Recalculate rect after flipping position
            tooltipRect = tooltipText.getBoundingClientRect();
        }

        const viewportWidth = document.documentElement.clientWidth;

        // Reset any previous horizontal adjustments
        tooltipText.style.left = '50%';
        tooltipText.style.marginLeft = `-${tooltipRect.width / 2}px`;

        // Recalculate rect after reset
        const finalRect = tooltipText.getBoundingClientRect();

        // Check for horizontal overflow
        if (finalRect.right > viewportWidth) {
            const overflow = finalRect.right - viewportWidth + 10; // 10px padding
            tooltipText.style.left = `calc(50% - ${overflow}px)`;
        } else if (finalRect.left < 0) {
            const overflow = -finalRect.left + 10; // 10px padding
            tooltipText.style.left = `calc(50% + ${overflow}px)`;
        }
    });

    tooltipIcon.addEventListener('mouseleave', () => {
        tooltipText.style.visibility = 'hidden';
        tooltipText.style.opacity = '0';
    });
});


