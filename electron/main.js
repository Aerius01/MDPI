import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import os from 'node:os';

// Docker image to use; publish your backend to this tag
const DOCKER_IMAGE = process.env.MDPI_DOCKER_IMAGE || 'mdpi-local:dev';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.join(__dirname, '..');

let mainWindow = null;
let currentProcess = null;

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

    mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// Disable hardware acceleration to avoid VSync/GL errors on some Linux setups
app.disableHardwareAcceleration();

app.whenReady().then(createWindow);

function runCommand(cmd, args, onData, onError, onClose) {
    const child = spawn(cmd, args, { shell: process.platform === 'win32' });
    child.stdout.on('data', (d) => onData?.(d.toString()))
    child.stderr.on('data', (d) => onError?.(d.toString()))
    child.on('close', (code) => onClose?.(code));
    child.on('error', (err) => onError?.(String(err)));
    return child;
}

function checkDocker() {
    return new Promise((resolve, reject) => {
        runCommand('docker', ['--version'],
            () => { },
            (err) => reject(new Error(`Docker not available: ${err}`)),
            (code) => code === 0 ? resolve(true) : reject(new Error('Docker not available'))
        );
    });
}

function ensureImage(image) {
    return new Promise((resolve, reject) => {
        // First check local images
        const images = runCommand('docker', ['images', '--format', '{{.Repository}}:{{.Tag}}']);
        let buffer = '';
        images.stdout.on('data', (d) => buffer += d.toString());
        images.on('close', (code) => {
            if (code !== 0) {
                reject(new Error('Failed to list docker images'));
                return;
            }
            const hasImage = buffer.split(/\r?\n/).some((l) => l.trim() === image);
            if (hasImage) {
                resolve(true);
                return;
            }
            // Local dev: build image from project Dockerfile
            mainWindow?.webContents.send('log', `Building local image ${image} from docker/Dockerfile...`);
            const dockerfilePath = path.join(REPO_ROOT, 'docker', 'Dockerfile');
            const build = runCommand('docker', ['build', '-f', dockerfilePath, '-t', image, REPO_ROOT],
                (out) => mainWindow?.webContents.send('log', out),
                (err) => mainWindow?.webContents.send('log', err),
                (code) => code === 0 ? resolve(true) : reject(new Error('Failed to build local image'))
            );
        });
    });
}

function runContainer({ inputPath, extraEnv }) {
    // Normalize host paths to absolute
    const inPath = path.resolve(inputPath);

    const volumeArgs = ['-v', `${inPath}:/app/input`];
    const defaultEnv = {
        HOME: '/tmp',
        XDG_CACHE_HOME: '/tmp/.cache'
    };
    const mergedEnv = { ...defaultEnv, ...(extraEnv || {}) };
    const envArgs = Object.entries(mergedEnv).flatMap(([k, v]) => ['-e', `${k}=${v}`]);
    const uid = typeof process.getuid === 'function' ? process.getuid() : undefined;
    const gid = typeof process.getgid === 'function' ? process.getgid() : undefined;
    const userArgs = (uid !== undefined && gid !== undefined) ? ['--user', `${uid}:${gid}`] : [];
    const args = ['run', '--rm', '--workdir', '/app', ...userArgs, ...envArgs, ...volumeArgs, DOCKER_IMAGE];

    currentProcess = runCommand('docker', args,
        (out) => mainWindow?.webContents.send('log', out),
        (err) => mainWindow?.webContents.send('log', err),
        (code) => {
            mainWindow?.webContents.send('completed', { code });
            currentProcess = null;
        }
    );
}

ipcMain.handle('pick-folder', async (_evt, { title }) => {
    const res = await dialog.showOpenDialog(mainWindow, { properties: ['openDirectory'], title: title || 'Select folder' });
    if (res.canceled || res.filePaths.length === 0) return null;
    return res.filePaths[0];
});

ipcMain.handle('run-pipeline', async (_evt, { inputPath, env }) => {
    mainWindow?.webContents.send('log', `Checking Docker...`);
    try {
        await checkDocker();
    } catch (e) {
        mainWindow?.webContents.send('log', String(e));
        return { ok: false, error: 'Docker is not available. Please install Docker Desktop.' };
    }

    mainWindow?.webContents.send('log', `Ensuring local image ${DOCKER_IMAGE}...`);
    try {
        await ensureImage(DOCKER_IMAGE);
    } catch (e) {
        mainWindow?.webContents.send('log', `Failed to ensure image: ${e}`);
        return { ok: false, error: 'Failed to build local Docker image' };
    }

    mainWindow?.webContents.send('log', `Running container...`);
    runContainer({ inputPath, extraEnv: env });
    return { ok: true };
});

ipcMain.handle('stop-pipeline', async () => {
    if (currentProcess && currentProcess.pid) {
        try {
            process.platform === 'win32' ? spawn('taskkill', ['/pid', String(currentProcess.pid), '/T', '/F']) : currentProcess.kill('SIGTERM');
            mainWindow?.webContents.send('log', 'Stopping container...');
            currentProcess = null;
            return { ok: true };
        } catch (e) {
            return { ok: false, error: String(e) };
        }
    }
    return { ok: true };
});


