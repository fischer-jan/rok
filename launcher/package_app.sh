#!/usr/bin/env bash
# Build dist/Rok.app, a self-contained Rok that runs without this repo,
# without Homebrew and without a venv.
#
# The bundle holds:
#   Contents/MacOS/Rok          a small native launcher that embeds Python
#   Contents/Resources/python/     a relocatable CPython (python-build-standalone,
#                                  from uv) with numpy and rok_native installed
#   Contents/Resources/app/        rok/ (with rok/shaders) and assets/
#   Contents/Resources/LICENSE, CREDITS.md, THIRD-PARTY-LICENSES.txt
#
# It needs uv, Rust (cargo) and the Xcode command line tools.
# Usage: launcher/package_app.sh            (the Python version: ROK_PY_VERSION, default 3.14)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYVER="${ROK_PY_VERSION:-3.14}"
DIST="$REPO/dist"
APP="$DIST/Rok.app"
RES="$APP/Contents/Resources"
PYHOME="$RES/python"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$REPO/pyproject.toml" | head -1)"

# 1. A relocatable CPython. uv installs python-build-standalone builds; Homebrew's
#    framework build links to Homebrew libraries, so it cannot go into the bundle.
uv python install "$PYVER" >/dev/null
SRC_PY="$(uv python find --managed-python "$PYVER")"
SRC_HOME="$(cd "$(dirname "$SRC_PY")/.." && pwd -P)"
case "$SRC_HOME" in
  "$(uv python dir)"*) ;;
  *) echo "Not a uv-managed Python: $SRC_HOME" >&2; exit 1 ;;
esac
echo "Python: $SRC_HOME"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RES/app"
cp -R "$SRC_HOME" "$PYHOME"

# Leave out what Rok does not need: Tcl/Tk and tkinter, IDLE, the test suite,
# pip and ensurepip, the docs data, the static library and the headers.
STD="$PYHOME/lib/python$PYVER"
rm -rf "$PYHOME/include" "$PYHOME/share" "$PYHOME/lib/pkgconfig" \
  "$PYHOME"/lib/libtcl* "$PYHOME"/lib/tcl* "$PYHOME"/lib/tk* \
  "$PYHOME"/lib/itcl* "$PYHOME"/lib/thread* \
  "$STD/test" "$STD/idlelib" "$STD/tkinter" "$STD/turtledemo" "$STD/turtle.py" \
  "$STD/ensurepip" "$STD/pydoc_data" "$STD/lib2to3" \
  "$STD"/config-"$PYVER"-* "$STD/site-packages"/pip* \
  "$STD"/lib-dynload/_tkinter* "$STD/lib-dynload"/*_test* "$STD/lib-dynload"/_xxtestfuzz*
( cd "$PYHOME/bin" && ls | grep -vx "python$PYVER" | xargs rm -f && ln -s "python$PYVER" python3 )

# 2. numpy, installed with uv into the bundled Python's site-packages: the
#    version uv.lock pins, and only a wheel whose hash uv.lock lists.
SITE="$STD/site-packages"
REQS="$DIST/requirements.txt"
mkdir -p "$DIST"
uv export --project "$REPO" --frozen --no-dev --no-emit-project --no-header --format requirements-txt > "$REQS"
uv pip install --quiet --python "$PYHOME/bin/python$PYVER" --target "$SITE" --require-hashes -r "$REQS"
rm -f "$REQS"
# numpy's own tests and its console scripts (f2py) stay out.
find "$SITE/numpy" -name tests -type d -prune -exec rm -rf {} +
rm -rf "$SITE/bin"

# 3. rok_native: a release wheel built for the bundled Python. The module is
#    not abi3, so the wheel must match this Python's version.
rm -f "$REPO"/native/target/wheels/rok_native-*.whl
uvx maturin build --release -m "$REPO/native/Cargo.toml" -i "$SRC_PY"
WHEEL="$(ls "$REPO"/native/target/wheels/rok_native-*.whl | head -1)"
uv pip install --quiet --python "$PYHOME/bin/python$PYVER" --target "$SITE" "$WHEEL"

# 3b. The licences: Rok's own, the credits, and every bundled library's text.
cp "$REPO/LICENSE" "$REPO/CREDITS.md" "$RES/"
python3 -I "$REPO/launcher/third_party_licences.py" "$SITE" "$RES/THIRD-PARTY-LICENSES.txt"

# 4. Rok's code and assets, without caches.
cp -R "$REPO/rok" "$RES/app/rok"
cp -R "$REPO/assets" "$RES/app/assets"
find "$RES" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$RES" \( -name '*.pyc' -o -name '.DS_Store' \) -delete

# 5. The icon.
cp "$REPO/assets/icon/Rok.icns" "$RES/Rok.icns"

# 6. The launcher: a native binary that embeds the bundled libpython and runs
#    rok/__main__.py. A native executable keeps "Rok" in the Dock and the
#    menu bar, and Spotlight lists the app. It finds every path from its own
#    location, so the bundle can live anywhere.
LIBPY="$PYHOME/lib/libpython$PYVER.dylib"
install_name_tool -id "@rpath/libpython$PYVER.dylib" "$LIBPY"
LSRC="$(mktemp -t rok_launch).c"
cat > "$LSRC" <<'CSRC'
#include <Python.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Rok.app: run Contents/Resources/app/rok/__main__.py in the bundled Python.
   Every argument but a Finder -psn one goes to Rok. */
int main(int argc, char **argv) {
    char exe[PATH_MAX], real[PATH_MAX], home[PATH_MAX], prog[PATH_MAX], script[PATH_MAX];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) != 0 || !realpath(exe, real)) {
        fprintf(stderr, "Rok: cannot find its own path\n");
        return 1;
    }
    char *slash = strrchr(real, '/');          /* .../Contents/MacOS/Rok */
    *slash = 0;
    snprintf(home, sizeof home, "%s/../Resources/python", real);
    snprintf(prog, sizeof prog, "%s/bin/python" ROK_PYVER, home);
    snprintf(script, sizeof script, "%s/../Resources/app/rok/__main__.py", real);

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.use_environment = 0;       /* ignore PYTHONPATH and PYTHONHOME */
    config.user_site_directory = 0;   /* no ~/.local site-packages */
    config.write_bytecode = 0;        /* keep the signed bundle unchanged */
    config.parse_argv = 0;            /* the arguments are Rok's, not Python's */
    PyStatus st = PyConfig_SetBytesString(&config, &config.home, home);
    if (!PyStatus_Exception(st))
        st = PyConfig_SetBytesString(&config, &config.executable, prog);
    if (!PyStatus_Exception(st))
        st = PyConfig_SetBytesString(&config, &config.run_filename, script);
    char **args = calloc((size_t)argc + 1, sizeof(char *));
    int n = 0;
    args[n++] = script;
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
xcrun clang -O1 -Wall -I"$SRC_HOME/include/python$PYVER" -DROK_PYVER="\"$PYVER\"" \
  -L"$PYHOME/lib" -lpython"$PYVER" \
  -Wl,-rpath,@executable_path/../Resources/python/lib \
  -o "$APP/Contents/MacOS/Rok" "$LSRC"
rm -f "$LSRC"

# The standalone python binary is a static 18 MB copy of libpython. Replace it
# with a small one that uses the bundled libpython, for scripts and tests:
#   Contents/Resources/python/bin/python3 -m rok run SCRIPT.py
PSRC="$(mktemp -t rok_python).c"
printf '#include <Python.h>\nint main(int argc, char **argv) { return Py_BytesMain(argc, argv); }\n' > "$PSRC"
rm -f "$PYHOME/bin/python$PYVER"
xcrun clang -O1 -Wall -I"$SRC_HOME/include/python$PYVER" -L"$PYHOME/lib" -lpython"$PYVER" \
  -Wl,-rpath,@executable_path/../lib -o "$PYHOME/bin/python$PYVER" "$PSRC"
rm -f "$PSRC"

# 7. Info.plist, with .vox and .rok as document types.
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Rok</string>
  <key>CFBundleDisplayName</key><string>Rok</string>
  <key>CFBundleIdentifier</key><string>app.rok.Rok</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>Rok</string>
  <key>CFBundleIconFile</key><string>Rok</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key><string>Voxel Model</string>
      <key>CFBundleTypeRole</key><string>Editor</string>
      <key>CFBundleTypeExtensions</key><array><string>vox</string></array>
      <key>CFBundleTypeIconFile</key><string>Rok</string>
      <key>LSHandlerRank</key><string>Alternate</string>
    </dict>
    <dict>
      <key>CFBundleTypeName</key><string>Rok Scene</string>
      <key>CFBundleTypeRole</key><string>Editor</string>
      <key>CFBundleTypeExtensions</key><array><string>rok</string></array>
      <key>CFBundleTypeIconFile</key><string>Rok</string>
      <key>LSHandlerRank</key><string>Owner</string>
    </dict>
  </array>
</dict>
</plist>
PLIST

# 8. An ad-hoc signature, so macOS starts the app on this Mac.
touch "$APP"
codesign --force --deep -s - "$APP"
codesign --verify --deep --strict "$APP"

# 9. Sizes.
ZIP="$DIST/Rok.zip"
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"
echo "Built $APP"
echo "App size: $(du -sh "$APP" | cut -f1)"
echo "Zip size: $(du -sh "$ZIP" | cut -f1)  ($ZIP)"
