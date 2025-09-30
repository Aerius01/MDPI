const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('mdpi', {
    pickFolder: (title) => ipcRenderer.invoke('pick-folder', { title }),
    validatePath: (path) => ipcRenderer.invoke('validate-path', { path }),
    run: (inputPaths, config) => ipcRenderer.invoke('run-pipeline', { inputPaths, config }),
    stop: () => ipcRenderer.invoke('stop-pipeline'),
    onLog: (cb) => ipcRenderer.on('log', (_e, msg) => cb(msg)),
    onProgressUpdate: (callback) => ipcRenderer.on('progress-update', (_evt, value) => callback(value)),
    onCompleted: (callback) => ipcRenderer.on('completed', (_evt, value) => callback(value)),
    onValidate: (callback) => ipcRenderer.on('validate-path-result', (_evt, value) => callback(value)),
});


