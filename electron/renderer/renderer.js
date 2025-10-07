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
    const container = document.getElementById(`container-${id}`);

    if (!path) {
        inputState[id] = { isValid: false, validationResults: [[false, 'Path cannot be empty.']], metadata: null };
        updateRowValidation(id);
        updateRunButtonState();
        return;
    }

    // Show loading state
    container.classList.add('validating');

    // Call the validation method with the input ID, the onValidate listener will handle the response
    await window.mdpi.validatePath(path, id);

    // Loading state will be removed by the onValidate callback
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
        browseBtn.disabled = true;
        browseBtn.textContent = 'Browsing...';

        const p = await window.mdpi.pickFolder('Select input folder');

        browseBtn.disabled = false;
        browseBtn.textContent = 'Browse';

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
        state.validationResults.forEach(([isValid, message, severity = 'error']) => {
            const p = document.createElement('p');
            const icon = isValid ? '✔' : (severity === 'warning' ? '⚠' : '❌');

            p.innerHTML = `<strong>${icon}</strong> ${message}`;
            p.className = isValid ? 'valid' : (severity === 'warning' ? 'warning' : 'invalid');

            // Add action hints for common errors
            if (!isValid) {
                if (message.includes('No CSV file')) {
                    const hint = document.createElement('span');
                    hint.className = 'error-hint';
                    hint.textContent = '→ Add a .csv file with pressure sensor data to the folder';
                    p.appendChild(hint);
                } else if (message.includes('no image')) {
                    const hint = document.createElement('span');
                    hint.className = 'error-hint';
                    hint.textContent = '→ Add images with naming format: [prefix]_YYYYmmdd_HHMMSSfff_[replicate_id].ext (e.g., img_20240315_143022500_001.tif)';
                    p.appendChild(hint);
                } else if (message.includes('Path cannot be empty')) {
                    const hint = document.createElement('span');
                    hint.className = 'error-hint';
                    hint.textContent = '→ Browse or type a folder path containing your images and CSV file';
                    p.appendChild(hint);
                }
            }

            validationDiv.appendChild(p);
        });
    }

    if (state.isValid && state.metadata) {
        const metadata = state.metadata;

        // Build display data from metadata
        const displayData = {};

        // Recording start - handle both date/time format and ISO string format
        if (metadata.recordingStart) {
            // New format from Node.js validation (ISO string)
            const date = new Date(metadata.recordingStart);
            displayData.recording_start = date.toLocaleString('en-US', {
                weekday: 'short',
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                fractionalSecondDigits: 3
            });
        } else if (metadata.recording_start_date && metadata.recording_start_time) {
            // Old format from Python validation
            displayData.recording_start = `${metadata.recording_start_date} ${metadata.recording_start_time}`;
        }

        // Image shape - handle both formats
        if (metadata.imageShape) {
            // New format from Node.js validation
            displayData.image_shape = `${metadata.imageShape.width} x ${metadata.imageShape.height} pixels`;
        } else if (metadata.image_width_pixels && metadata.image_height_pixels) {
            // Old format from Python validation
            displayData.image_shape = `${metadata.image_width_pixels} x ${metadata.image_height_pixels} pixels`;
        }

        // Camera format
        if (metadata.cameraFormat || metadata.camera_format) {
            displayData.camera_format = metadata.cameraFormat || metadata.camera_format;
        }

        // Display metadata in grid
        const keyOrder = ["recording_start", "image_shape", "camera_format"];
        keyOrder.forEach(key => {
            if (displayData[key]) {
                const keyEl = document.createElement('strong');
                keyEl.textContent = `${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:`;
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

    // Show loading feedback - will remain until pipeline completes
    runBtn.classList.add('loading');
    runBtn.disabled = true;
    stopBtn.disabled = false;
    setControlsDisabled(true);

    const res = await window.mdpi.run(inputPaths, config);

    if (!res?.ok) {
        appendLog(String(res?.error || 'Failed to start'));
        // Remove loading state on error
        runBtn.classList.remove('loading');
    }
    // Note: loading state is removed in onCompleted callback when pipeline finishes
});

stopBtn.addEventListener('click', async () => {
    stopBtn.disabled = true;
    await window.mdpi.stop();
    // Note: stopBtn will be re-enabled in onCompleted callback
});

window.mdpi.onLog((msg) => {
    appendLog(msg);
});

window.mdpi.onValidate((data) => {
    const { id, results, metadata } = data;
    const isValid = results.every(r => r[0]);

    if (id && inputState.hasOwnProperty(id)) {
        // Remove loading state
        const container = document.getElementById(`container-${id}`);
        if (container) {
            container.classList.remove('validating');
        }

        // Update validation state
        inputState[id] = { isValid, validationResults: results, metadata };
        updateRowValidation(id);
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

    // Remove loading spinner from Run button
    runBtn.classList.remove('loading');
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

    const showTooltip = () => {
        // Reset vertical position to default (above)
        tooltipText.classList.remove('tooltip-below');

        // Make it visible to calculate its dimensions
        tooltipText.style.visibility = 'visible';
        tooltipText.style.opacity = '1';
        tooltipText.setAttribute('aria-hidden', 'false');
        tooltipIcon.setAttribute('aria-expanded', 'true');

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
    };

    const hideTooltip = () => {
        tooltipText.style.visibility = 'hidden';
        tooltipText.style.opacity = '0';
        tooltipText.setAttribute('aria-hidden', 'true');
        tooltipIcon.setAttribute('aria-expanded', 'false');
    };

    // Mouse events
    tooltipIcon.addEventListener('mouseenter', showTooltip);
    tooltipIcon.addEventListener('mouseleave', hideTooltip);

    // Keyboard events
    tooltipIcon.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            const isExpanded = tooltipIcon.getAttribute('aria-expanded') === 'true';
            if (isExpanded) {
                hideTooltip();
            } else {
                showTooltip();
            }
        } else if (e.key === 'Escape') {
            hideTooltip();
        }
    });
});


