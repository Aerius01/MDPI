import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('mdpi', {
    pickFolder: (title) => ipcRenderer.invoke('pick-folder', { title }),
    run: (inputPath, env) => ipcRenderer.invoke('run-pipeline', { inputPath, env }),
    stop: () => ipcRenderer.invoke('stop-pipeline'),
    onLog: (cb) => {
        ipcRenderer.on('log', (_e, msg) => cb(msg));
    },
    onCompleted: (cb) => {
        ipcRenderer.on('completed', (_e, payload) => cb(payload));
    }
});


