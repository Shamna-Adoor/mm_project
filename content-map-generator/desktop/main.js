const { app, BrowserWindow, shell } = require('electron')
const { spawn } = require('child_process')
const path = require('path')
const http = require('http')

const PORT    = 8000
const ROOT    = path.join(__dirname, '..')
const PYTHON  = path.join(ROOT, '.venv', 'bin', 'python')

let win    = null
let server = null

// ── Start the FastAPI server ──────────────────────────────────────────────────
function startServer() {
  server = spawn(PYTHON, ['-m', 'uvicorn', 'server:app', `--port=${PORT}`, '--host=127.0.0.1'], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  server.stderr.on('data', d => process.stderr.write(d))
  server.on('exit', code => { if (code !== null) console.log(`Server exited (${code})`) })
}

// ── Poll until the server is up, then load the window ────────────────────────
function waitForServer(cb, attempts = 40) {
  http.get(`http://127.0.0.1:${PORT}/api/health`, res => {
    if (res.statusCode === 200) { cb(); return }
    retry()
  }).on('error', retry)

  function retry() {
    if (attempts > 0) setTimeout(() => waitForServer(cb, attempts - 1), 500)
    else cb() // try anyway
  }
}

// ── Create window ─────────────────────────────────────────────────────────────
function createWindow() {
  win = new BrowserWindow({
    width:  1400,
    height: 900,
    title:  'Content Map Player',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  win.loadURL(`http://127.0.0.1:${PORT}`)
  win.on('closed', () => { win = null })

  // Open external links in the system browser
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })
}

// ── App lifecycle ─────────────────────────────────────────────────────────────
app.whenReady().then(() => {
  startServer()

  // Show a loading window while the server warms up
  win = new BrowserWindow({ width: 480, height: 280, frame: false, resizable: false })
  win.loadURL(`data:text/html,
    <body style="margin:0;background:#111827;display:flex;align-items:center;justify-content:center;height:100vh;font-family:system-ui">
      <div style="text-align:center;color:#fff">
        <div style="font-size:2rem;margin-bottom:12px">▶</div>
        <div style="font-size:1rem;font-weight:600">Content Map Player</div>
        <div style="font-size:.8rem;color:#9ca3af;margin-top:8px">Starting analysis server…</div>
      </div>
    </body>`)

  waitForServer(() => {
    if (win) win.close()
    createWindow()
  })
})

app.on('window-all-closed', () => {
  if (server) server.kill()
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (win === null) createWindow()
})

app.on('before-quit', () => {
  if (server) server.kill()
})
