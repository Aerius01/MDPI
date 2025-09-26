## MDPI Electron App

Electron frontend for running the Dockerized MDPI pipeline.

### Prerequisites
- Docker Desktop or Docker Engine installed and running
- Prebuilt image published to a registry and accessible by the app
  - Set the image via env var `MDPI_DOCKER_IMAGE` or edit `DOCKER_IMAGE` in `main.js`.

### Development
```bash
cd electron
npm install
npm run dev
```

### Packaging
```bash
cd electron
npm run build
```
Artifacts will be created per OS target (dmg, nsis, AppImage) using electron-builder.

### How it works
- Renderer lets users select input/output folders and start/stop
- Main process validates Docker, pulls image if missing, then runs:
  - `docker run --rm -v <input>:/app/input -v <output>:/app/output <IMAGE>`
- stdout/stderr are streamed to renderer via IPC
- Completion notifies renderer with exit code

### Integrating with your backend
- Inside your Docker container, ensure your entrypoint reads from `/app/input` and writes to `/app/output`.
- Print logs to stdout/stderr so they appear in the app.

### Notes
- For first run, the app will automatically pull the image if not present and display pull output in the log panel.
- You can pass additional environment variables by editing the `run` call in `renderer.js` or wiring UI controls.


