const { app, BrowserWindow, dialog, Menu } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

let backendProcess = null;
let mainWindow = null;
let backendStarted = false;
let stdoutBuffer = "";

app.setAppUserModelId("jp.modestudio.app");

function getBackendExePath() {
  if (app.isPackaged) {
    return path.join(
      process.resourcesPath,
      "modestudio-backend",
      "modestudio-backend.exe"
    );
  }

  return path.join(
    __dirname,
    "..",
    "python_backend",
    "dist",
    "modestudio-backend",
    "modestudio-backend.exe"
  );
}

function getIconPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "icon.ico");
  }

  return path.join(__dirname, "build", "icon.ico");
}

function createMainWindow(url) {
  mainWindow = new BrowserWindow({
    width: 1500,
    height: 950,
    minWidth: 1100,
    minHeight: 720,
    show: false,
    icon: getIconPath(),
    backgroundColor: "#f5f7fb",
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  mainWindow.loadURL(url);

  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function startBackend() {
  const exePath = getBackendExePath();

  backendProcess = spawn(exePath, [], {
    windowsHide: true
  });

  const startupTimer = setTimeout(() => {
    if (!backendStarted) {
      dialog.showErrorBox(
        "ModeStudio backend did not start",
        "The backend process started, but it did not report a local URL."
      );
      app.quit();
    }
  }, 60000);

  backendProcess.stdout.on("data", (data) => {
    const text = data.toString();
    process.stdout.write(text);

    stdoutBuffer += text;
    const match = stdoutBuffer.match(/MODESTUDIO_URL=(http:\/\/127\.0\.0\.1:\d+)/);

    if (match && !backendStarted) {
      backendStarted = true;
      clearTimeout(startupTimer);
      createMainWindow(match[1]);
    }

    if (stdoutBuffer.length > 10000) {
      stdoutBuffer = stdoutBuffer.slice(-5000);
    }
  });

  backendProcess.stderr.on("data", (data) => {
    process.stderr.write(data.toString());
  });

  backendProcess.on("error", (err) => {
    clearTimeout(startupTimer);
    dialog.showErrorBox(
      "ModeStudio backend failed to start",
      String(err && err.message ? err.message : err)
    );
    app.quit();
  });

  backendProcess.on("exit", (code) => {
    clearTimeout(startupTimer);
    backendProcess = null;

    if (!app.isQuitting && mainWindow) {
      dialog.showErrorBox(
        "ModeStudio backend stopped",
        `The backend process exited with code ${code}.`
      );
      app.quit();
    }
  });
}

function stopBackend() {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
}

const gotLock = app.requestSingleInstanceLock();

if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) {
        mainWindow.restore();
      }
      mainWindow.focus();
    }
  });

  app.whenReady().then(() => {
    Menu.setApplicationMenu(null);
    startBackend();
  });

  app.on("before-quit", () => {
    app.isQuitting = true;
    stopBackend();
  });

  app.on("window-all-closed", () => {
    app.quit();
  });
}