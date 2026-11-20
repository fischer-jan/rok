#!/usr/bin/env bash
# Build Rok.app, a macOS shortcut that starts Rok from this repo.
# It needs the venv .venv-rok with rok_native built into it.
# Usage: launcher/make_rok_app.sh [target dir]   (default: ~/Applications)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-$HOME/Applications}"
APP="$TARGET/Rok.app"
PY="$REPO/.venv-rok/bin/python"
OPEN_PY="$REPO/rok/__main__.py"

"$PY" -c "import rok_native" 2>/dev/null || {
  echo "No rok_native in $PY. Build it first:" >&2
  echo "  uvx maturin develop --release -m native/Cargo.toml  (inside .venv-rok)" >&2
  exit 1
}

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Rok</string>
  <key>CFBundleDisplayName</key><string>Rok</string>
  <key>CFBundleIdentifier</key><string>app.rok.Rok</string>
  <key>CFBundleVersion</key><string>0.0.1</string>
  <key>CFBundleShortVersionString</key><string>0.0.1</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>rok</string>
  <key>CFBundleIconFile</key><string>Rok</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# The executable: a small native binary that embeds the venv's Python and runs
# rok/__main__.py. Spotlight lists only apps with a native executable. It
# embeds Python instead of starting the venv's python, because
# Homebrew's python restarts itself inside its own Python.app, and the Dock and
# the menu bar then say "Python" instead of "Rok".
FW="$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('PYTHONFRAMEWORKPREFIX'))")"
INC="$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['include'])")"
SRC="$(mktemp -t rok_launch).c"
cat > "$SRC" <<'CSRC'
#include <Python.h>
#include <stdio.h>
#include <string.h>
/* Rok shortcut: run rok/__main__.py in an embedded Python that uses the venv.
   Every argument but a Finder -psn one goes to Rok. */
int main(int argc, char **argv) {
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    /* The venv's python as the executable: Python then finds pyvenv.cfg and
       the venv's site-packages. */
    PyStatus st = PyConfig_SetBytesString(&config, &config.executable, ROK_PY);
    if (!PyStatus_Exception(st))
        st = PyConfig_SetBytesString(&config, &config.run_filename, ROK_OPEN);
    char **args = calloc((size_t)argc + 1, sizeof(char *));
    int n = 0;
    args[n++] = ROK_OPEN;
    for (int i = 1; i < argc; i++)
        if (strncmp(argv[i], "-psn", 4) != 0)
            args[n++] = argv[i];
    if (!PyStatus_Exception(st))
        st = PyConfig_SetBytesArgv(&config, n, args);
    if (!PyStatus_Exception(st))
        st = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(st))
        Py_ExitStatusException(st);
    return Py_RunMain();
}
CSRC
xcrun clang -O1 -Wall -I"$INC" -F"$FW" -framework Python -Wl,-rpath,"$FW" \
  -DROK_PY="\"$PY\"" -DROK_OPEN="\"$OPEN_PY\"" -o "$APP/Contents/MacOS/rok" "$SRC"
rm -f "$SRC"

cp "$REPO/assets/icon/Rok.icns" "$APP/Contents/Resources/Rok.icns"

touch "$APP"
# An ad-hoc signature, then a fresh LaunchServices entry: Spotlight and the
# Dock leave an unsigned or stale bundle out.
codesign --force --deep -s - "$APP" 2>/dev/null || true
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
[ -x "$LSREGISTER" ] && "$LSREGISTER" -f "$APP" || true
echo "Built $APP"
echo "Open it from Finder, or: open \"$APP\" [--args FILE.vox]. Drag it to the Dock to keep it."
