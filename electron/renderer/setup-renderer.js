window.api.on('status-update', (message) => {
    const statusContainer = document.getElementById('status-messages');
    const logElement = statusContainer?.querySelector('pre code');

    if (logElement) {
        if (logElement.textContent === 'Initializing...') {
            logElement.textContent = '';
        }

        // Sanitize message to remove ANSI escape codes which can mess up rendering
        const sanitizedMessage = message.replace(/[\u001b\u009b][[()#;?]*.{0,2}m/g, '');

        logElement.textContent += sanitizedMessage;
        statusContainer.scrollTop = statusContainer.scrollHeight;
    }
});

window.api.on('setup-error', (errorMessage) => {
    const statusContainer = document.getElementById('status-container');
    const errorContainer = document.getElementById('error-container');
    const errorMessageElem = document.getElementById('error-message');

    if (statusContainer) statusContainer.style.display = 'none';
    if (errorContainer) errorContainer.style.display = 'block';
    if (errorMessageElem) errorMessageElem.textContent = errorMessage;
});

document.addEventListener('DOMContentLoaded', () => {
    const retryButton = document.getElementById('retry-button');
    if (retryButton) {
        retryButton.addEventListener('click', () => {
            window.api.send('retry-setup');
        });
    }

    const closeButton = document.getElementById('close-btn');
    if (closeButton) {
        closeButton.addEventListener('click', () => {
            window.api.send('cancel-setup');
        });
    }

    const toggleLogsBtn = document.getElementById('toggle-logs-btn');
    const statusContainer = document.getElementById('status-container');

    if (toggleLogsBtn && statusContainer) {
        toggleLogsBtn.addEventListener('click', () => {
            const isExpanded = statusContainer.classList.toggle('expanded');
            toggleLogsBtn.classList.toggle('expanded', isExpanded);

            const buttonText = toggleLogsBtn.querySelector('span:first-child');
            if (buttonText) {
                buttonText.textContent = isExpanded ? 'Hide Details' : 'Show Details';
            }

            const newHeight = isExpanded ? 400 : 210;
            window.api.send('resize-setup-window', { height: newHeight });
        });
    }
});
