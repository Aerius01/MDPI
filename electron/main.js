import { app, BrowserWindow, dialog, ipcMain } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { spawn, exec } from 'node:child_process';
import fetch from 'node-fetch';
import os from 'node:os';


// Resolve Docker image dynamically for development:
// 1) If MDPI_DOCKER_IMAGE is set, use it
// 2) Else if MDPI_GIT_BRANCH is set, use ghcr.io/<owner>/<image>:<branch>
// 3) Else if a git repo is present, read current branch from .git/HEAD
// 4) Else fall back to :latest
const GHCR_OWNER = (process.env.MDPI_GHCR_OWNER || 'aerius01').toLowerCase();
const GHCR_IMAGE = 'mdpi-pipeline';
const PULL_POLICY = (process.env.MDPI_PULL_POLICY || 'always').toLowerCase(); // always | if-not-present | never

function readGitBranch() {
    try {
        const headPath = path.join(REPO_ROOT, '.git', 'HEAD');
        if (!fs.existsSync(headPath)) return null;
        const head = fs.readFileSync(headPath, 'utf8').trim();
        // Typical format: "ref: refs/heads/<branch>"
        const match = head.match(/^ref:\s+refs\/heads\/(.+)$/);
        return match ? match[1] : null;
    } catch (_e) {
        return null;
    }
}

function sanitizeForDockerTag(value) {
    // Docker tag rules are restrictive; lowercase and replace invalid chars with '-'
    return String(value).toLowerCase().replace(/[^a-z0-9._-]/g, '-');
}

function computeDefaultImage() {
    if (process.env.MDPI_DOCKER_IMAGE) {
        return process.env.MDPI_DOCKER_IMAGE;
    }
    const envBranch = process.env.MDPI_GIT_BRANCH;
    const gitBranch = envBranch || readGitBranch();
    if (gitBranch) {
        const tag = sanitizeForDockerTag(gitBranch);
        return `ghcr.io/${GHCR_OWNER}/${GHCR_IMAGE}:${tag}`;
    }
    return `ghcr.io/${GHCR_OWNER}/${GHCR_IMAGE}:latest`;
}

const DOCKER_IMAGE = computeDefaultImage();
const CONTAINER_NAME = 'mdpi-backend-container';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.join(__dirname, '..');
const PROJECTS_ROOT = path.join(REPO_ROOT, '..');
const USER_HOME = os.homedir();
const SERVER_URL = 'http://localhost:5001';

let mainWindow = null;
let setupWindow = null; // Window for setup progress
let backendContainerId = null;
let logStreamProcess = null;
let currentModule = 'PIPELINE';

function createSetupWindow() {
    setupWindow = new BrowserWindow({
        width: 500,
        height: 210,
        title: 'MDPI Backend Setup',
        frame: false,
        resizable: false,
        transparent: true,
        backgroundColor: '#00000000',
        webPreferences: {
            preload: path.join(__dirname, 'setup-preload.cjs'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });
    setupWindow.loadFile(path.join(__dirname, 'renderer', 'setup.html'));
    setupWindow.center();
    setupWindow.on('closed', () => {
        setupWindow = null;
    });
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1100,
        height: 750,
        webPreferences: {
            // Use CJS preload for maximal compatibility with contextIsolation
            preload: path.join(__dirname, 'preload.cjs'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });

    mainWindow.setMenu(null);
    mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});

// On quit, ensure the container is stopped
app.on('quit', async () => {
    if (backendContainerId) {
        await stopBackendContainer();
    }
});


app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// Disable hardware acceleration
app.disableHardwareAcceleration();

app.whenReady().then(async () => {
    // Preflight: ensure Docker is installed and the daemon is running
    const ok = await preflightDocker();
    if (!ok) return; // preflight will show a message and quit

    createSetupWindow();
    // Automatically start the backend on launch
    await manageBackendProcess();
});

ipcMain.on('resize-setup-window', (_event, { height }) => {
    if (setupWindow) {
        const [width] = setupWindow.getSize();
        setupWindow.setResizable(true);
        setupWindow.setSize(width, height, true);
        setupWindow.setResizable(false);
    }
});

ipcMain.on('cancel-setup', () => {
    log('User cancelled setup. Shutting down.');
    app.quit();
});


function log(message) {
    console.log(message);
    const targetWindow = setupWindow || mainWindow;
    if (targetWindow) {
        if (targetWindow === setupWindow) {
            targetWindow.webContents.send('status-update', message);
        } else {
            targetWindow.webContents.send('log', message);
        }
    }
}

function logCommandOutput(message) {
    // Only log command output during setup
    if (setupWindow) {
        log(message);
    } else {
        // Also log to the main console for debugging, but not to the UI
        console.log(message);
    }
}

function runCommand(cmd, args) {
    return new Promise((resolve, reject) => {
        const child = spawn(cmd, args, { shell: process.platform === 'win32' });
        let stdout = '';
        let stderr = '';
        child.stdout.on('data', (d) => {
            stdout += d.toString();
            logCommandOutput(d.toString());
        });
        child.stderr.on('data', (d) => {
            stderr += d.toString();
            logCommandOutput(d.toString());
        });
        child.on('close', (code) => {
            if (code === 0) {
                resolve(stdout.trim());
            } else {
                reject(new Error(stderr));
            }
        });
        child.on('error', (err) => reject(err));
    });
}

async function showOkAndQuit(title, message) {
    try {
        await dialog.showMessageBox({
            type: 'error',
            title: title || 'MDPI',
            message: message || 'An unexpected error occurred.',
            buttons: ['OK'],
            defaultId: 0,
            noLink: true
        });
    } finally {
        app.quit();
    }
}

async function preflightDocker() {
    // Check Docker CLI presence
    try {
        await runCommand('docker', ['--version']);
    } catch (_e) {
        await showOkAndQuit(
            'Docker not found',
            'Docker is required but was not found.\nPlease install Docker and ensure it is on your PATH, then relaunch the app.'
        );
        return false;
    }

    // Check Docker daemon connectivity
    try {
        await runCommand('docker', ['info']);
    } catch (_e) {
        await showOkAndQuit(
            'Docker is not running',
            'Docker is installed but not running.\nStart Docker, then relaunch the app.'
        );
        return false;
    }
    return true;
}

async function stopLogStream() {
    if (logStreamProcess) {
        log('Stopping log stream...');
        logStreamProcess.kill();
        logStreamProcess = null;
    }
}

async function startLogStream(containerId) {
    await stopLogStream(); // Ensure no old stream is running

    log('Starting log stream...');
    logStreamProcess = spawn('docker', ['logs', '-f', containerId], { shell: process.platform === 'win32' });

    const handleData = (data) => {
        const text = data.toString();
        const lines = text.split(/(\r\n|\n|\r)/);
        const progressRegex = /\[PROGRESS\]\s+(\d+)\/(\d+)/;
        const moduleRegex = /\[([A-Z_]+)\]/;
        const completionRegex = /\[PIPELINE\]: All steps completed successfully!/;
        const errorRegex = /\[PIPELENE\]: Error:/;

        for (const line of lines) {
            let trimmedLine = line.trim();
            if (!trimmedLine) continue;

            // Replace container-relative paths with host-relative paths
            trimmedLine = trimmedLine.replace(/\/projects\/MDPI/g, REPO_ROOT);
            trimmedLine = trimmedLine.replace(/\/host_home/g, USER_HOME);

            // Suppress spurious MLIR and ABSL messages
            if (trimmedLine.includes('MLIR V1 optimization pass is not enabled') ||
                trimmedLine.includes('All log messages before absl::InitializeLog() is called are written to STDERR')) {
                continue;
            }

            // Suppress Flask development server warnings
            // eslint-disable-next-line no-control-regex
            const ansiRegex = /[\u001b\u009b][[()#;?]*.{0,2}(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g;
            const plainLine = trimmedLine.replace(ansiRegex, '');
            const flaskMessages = [
                '* Serving Flask app',
                '* Debug mode: off',
                'WARNING: This is a development server.',
                '* Running on all addresses',
                '* Running on http://',
                'Press CTRL+C to quit'
            ];

            if (flaskMessages.some(msg => plainLine.includes(msg))) {
                continue;
            }

            const moduleMatch = trimmedLine.match(moduleRegex);
            if (moduleMatch && !trimmedLine.startsWith('[PROGRESS]')) {
                const newModule = moduleMatch[1];
                if (newModule !== currentModule) {
                    currentModule = newModule;
                }
            }

            const progressMatch = trimmedLine.match(progressRegex);

            if (progressMatch) {
                const step = parseInt(progressMatch[1], 10);
                const total = parseInt(progressMatch[2], 10);
                const percent = Math.round((step / total) * 100);

                const barWidth = 40;
                const filledLen = Math.round(barWidth * percent / 100);
                const emptyLen = barWidth - filledLen;
                const percentStr = String(percent).padStart(3, ' ');
                const stepStr = String(step).padStart(String(total).length, ' ');

                const bar = `[${'#'.repeat(filledLen)}${'-'.repeat(emptyLen)}]`;
                const progressBar = `[${currentModule}]: ${bar} ${percentStr}% | ${stepStr}/${total}`;

                mainWindow?.webContents.send('progress-update', { text: progressBar });
            } else if (completionRegex.test(trimmedLine)) {
                log(trimmedLine);
                mainWindow?.webContents.send('completed', { code: 0 });
            } else if (errorRegex.test(trimmedLine)) {
                log(trimmedLine);
                mainWindow?.webContents.send('completed', { code: 1 });
            } else if (!/HTTP\/\d\.\d" \d{3}/.test(plainLine)) {
                log(trimmedLine);
            }
        }
    };

    logStreamProcess.stdout.on('data', handleData);
    logStreamProcess.stderr.on('data', handleData);

    logStreamProcess.on('close', (code) => {
        if (code !== 0) {
            log(`Log stream process exited with code ${code}`);
        }
        logStreamProcess = null;
    });

    logStreamProcess.on('error', (err) => {
        log(`Log stream error: ${err.message}`);
        logStreamProcess = null;
    });
}

async function isContainerRunning(name) {
    return new Promise((resolve) => {
        exec(`docker ps -q -f name=${name}`, (err, stdout) => {
            if (err || !stdout) resolve(null);
            else resolve(stdout.trim());
        });
    });
}

async function stopBackendContainer() {
    log('Stopping backend container...');
    await stopLogStream();
    try {
        await runCommand('docker', ['stop', CONTAINER_NAME]);
        log('Container stopped.');
        log('[SEPARATOR]');
    } catch (e) {
        log(`Could not stop container (may not be running): ${e.message}`);
    } finally {
        backendContainerId = null;
    }
}

async function ensureImage() {
    log(`Checking for Docker image: ${DOCKER_IMAGE} (pullPolicy=${PULL_POLICY})...`);

    // Development image is built locally
    if (DOCKER_IMAGE === 'mdpi-local:dev') {
        log(`Image ${DOCKER_IMAGE} not found. Building...`);
        try {
            const dockerfilePath = path.join(REPO_ROOT, 'docker', 'Dockerfile');
            await runCommand('docker', ['build', '-f', dockerfilePath, '-t', DOCKER_IMAGE, REPO_ROOT]);
            log('Image built successfully.');
        } catch (buildError) {
            log(`Failed to build image ${DOCKER_IMAGE}. Error: ${buildError.message}`);
            throw new Error(`Could not build development image: ${DOCKER_IMAGE}`);
        }
        return;
    }

    // For registry images, decide when to pull
    const imageExistsLocally = async () => {
        try {
            await runCommand('docker', ['image', 'inspect', DOCKER_IMAGE]);
            return true;
        } catch (_e) {
            return false;
        }
    };

    const tryPull = async () => {
        log(`Attempting to pull ${DOCKER_IMAGE} from registry...`);
        try {
            await runCommand('docker', ['pull', DOCKER_IMAGE]);
            log(`Successfully pulled image ${DOCKER_IMAGE}.`);
            return true;
        } catch (pullError) {
            log(`Warning: pull failed for ${DOCKER_IMAGE}: ${pullError.message}`);
            return false;
        }
    };

    if (PULL_POLICY === 'always') {
        const pulled = await tryPull();
        if (!pulled) {
            if (await imageExistsLocally()) {
                log(`Using existing local image ${DOCKER_IMAGE} (offline or network issue).`);
                return;
            }
            throw new Error(`Could not pull production image and no local copy found: ${DOCKER_IMAGE}`);
        }
        return;
    }

    if (PULL_POLICY === 'never') {
        if (await imageExistsLocally()) {
            log(`Image ${DOCKER_IMAGE} found locally (pull policy: never).`);
            return;
        }
        throw new Error(`Image not present locally and pulls are disabled (MDPI_PULL_POLICY=never): ${DOCKER_IMAGE}`);
    }

    // if-not-present
    if (await imageExistsLocally()) {
        log(`Image ${DOCKER_IMAGE} found locally (pull policy: if-not-present).`);
        return;
    }
    const pulled = await tryPull();
    if (!pulled) {
        throw new Error(`Could not pull production image: ${DOCKER_IMAGE}`);
    }
}

async function startBackendContainer() {
    log('Checking for backend container...');
    let containerId = await isContainerRunning(CONTAINER_NAME);

    if (containerId) {
        log(`Container '${CONTAINER_NAME}' is already running.`);
    } else {
        log(`Starting backend container...`);
        await ensureImage(); // Build or pull image

        // Stop container if it exists but is not running
        await runCommand('docker', ['rm', CONTAINER_NAME]).catch(() => { });

        const uid = typeof process.getuid === 'function' ? process.getuid() : undefined;
        const gid = typeof process.getgid === 'function' ? process.getgid() : undefined;
        const userArgs = (uid !== undefined && gid !== undefined) ? ['--user', `${uid}:${gid}`] : [];

        log('Running container...');
        containerId = await runCommand('docker', [
            'run', '-d', '--rm',
            '--name', CONTAINER_NAME,
            '-p', '5001:5001',
            // Mount the user's home directory to allow access to any file.
            '-v', `${USER_HOME}:/host_home`,
            '-e', `HOST_HOME_DIR=${USER_HOME}`,
            // Set a writable home directory for the non-root user
            '-e', 'HOME=/tmp',
            '-e', 'XDG_CACHE_HOME=/tmp/.cache',
            ...userArgs,
            DOCKER_IMAGE
        ]);
        log(`Backend container started with ID: ${containerId}`);
    }

    if (containerId) {
        // Wait for the server to be healthy
        log('Waiting for backend server to become healthy...');
        let retries = 30; // Increased retries
        let healthy = false;
        while (retries > 0) {
            try {
                const res = await fetch(`${SERVER_URL}/health`);
                if (res.ok) {
                    log('Backend server is healthy.');
                    healthy = true;
                    break;
                }
            } catch (e) {
                // Ignore fetch errors while waiting
            }
            await new Promise(resolve => setTimeout(resolve, 1000));
            retries--;
        }

        if (!healthy) {
            log('Backend server did not become healthy in time.');
            await stopBackendContainer();
            throw new Error('Server readiness check failed.');
        }

        return containerId;
    } else {
        throw new Error('Backend container ID not found after start attempt.');
    }
}

async function manageBackendProcess() {
    try {
        backendContainerId = await startBackendContainer();

        if (backendContainerId) {
            // Backend is ready, switch to main window
            if (setupWindow) {
                setupWindow.close();
            }
            createWindow();
            await startLogStream(backendContainerId);

        } else {
            throw new Error('Backend container ID not found after start attempt.');
        }
    } catch (e) {
        log(`Failed to start backend: ${e.message}`);
        setupWindow?.webContents.send('setup-error', e.message);
        backendContainerId = null;
    }
}

ipcMain.on('retry-setup', () => {
    log('Retrying setup...');
    if (setupWindow) {
        // Reset UI
        setupWindow.webContents.send('status-update', 'Retrying...');
        const errorContainer = `document.getElementById('error-container').style.display = 'none';`;
        const statusContainer = `document.getElementById('status-container').style.display = 'block';`;
        setupWindow.webContents.executeJavaScript(`${errorContainer}; ${statusContainer}`);
    }
    manageBackendProcess();
});

ipcMain.handle('pick-folder', async (_evt, { title }) => {
    const res = await dialog.showOpenDialog({
        properties: ['openDirectory'],
        title: title || 'Select folder'
    });
    if (res.canceled || res.filePaths.length === 0) {
        return null;
    }
    // With 'openDirectory', the dialog will only return directory paths.
    return res.filePaths[0];
});

ipcMain.handle('validate-path', async (evt, { path: filePath }) => {
    if (!backendContainerId) {
        return { results: [[false, 'Backend is not running.']], metadata: null };
    }
    try {
        // We need to pass the path as the host sees it.
        // The server will adapt it for the container.
        const response = await fetch(`${SERVER_URL}/validate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: filePath })
        });

        const data = await response.json();
        evt.sender.send('validate-path-result', data);
        return data;

    } catch (e) {
        const errorResult = { results: [[false, `Error validating path: ${e.message}`]], metadata: null };
        evt.sender.send('validate-path-result', errorResult);
        return errorResult;
    }
});


ipcMain.handle('run-pipeline', async (_evt, { inputPaths, config }) => {
    if (!backendContainerId) {
        return { ok: false, error: 'Backend is not running.' };
    }
    log('Initializing pipeline');
    try {
        const response = await fetch(`${SERVER_URL}/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_paths: inputPaths, config })
        });
        const resJson = await response.json();
        if (!response.ok) {
            log(`Pipeline start failed: ${resJson.error}`);
            return { ok: false, error: resJson.error };
        }

        return { ok: true };
    } catch (e) {
        log(`Error starting pipeline: ${e.message}`);
        return { ok: false, error: e.message };
    }
});

ipcMain.handle('stop-pipeline', async () => {
    if (!backendContainerId) {
        return { ok: false, error: 'Backend is not running.' };
    }
    try {
        log('Sending stop signal to backend...');
        const response = await fetch(`${SERVER_URL}/stop`, {
            method: 'POST'
        });

        if (response.ok) {
            log('Stop signal acknowledged by backend. Pipeline stopped.');
            mainWindow?.webContents.send('completed', { code: 'stopped' });
            return { ok: true };
        } else {
            const error = await response.json();
            log(`Backend failed to stop gracefully: ${error.error}`);
            log('Falling back to a container restart to ensure a clean state.');
            await stopBackendContainer();
            mainWindow?.webContents.send('log', 'Restarting container...');
            backendContainerId = await startBackendContainer();
            await startLogStream(backendContainerId);
            mainWindow?.webContents.send('log', 'Backend container restarted.');
            mainWindow?.webContents.send('completed', { code: 'stopped' });
            return { ok: true };
        }
    } catch (e) {
        log(`Error sending stop signal: ${e.message}`);
        // If the fetch fails, the container might be down. Try to restart it.
        try {
            log('The backend is unresponsive. Attempting to recover by restarting the container.');
            await stopBackendContainer(); // Ensure it's really stopped first
            backendContainerId = await startBackendContainer();
            await startLogStream(backendContainerId);
            mainWindow?.webContents.send('log', 'Backend container restarted.');
            mainWindow?.webContents.send('completed', { code: 'stopped' });
            return { ok: true };
        } catch (restartError) {
            log(`FATAL: Could not restart backend: ${restartError.message}`);
            mainWindow?.webContents.send('log', `FATAL: Could not restart backend: ${restartError.message}`);
            mainWindow?.webContents.send('completed', { code: 'error' });
            return { ok: false, error: restartError.message };
        }
    }
});


