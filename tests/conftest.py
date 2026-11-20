"""The tests get a config folder of their own.

Rok keeps the look, the render preset, Voxel grid and more in its config
folder (rok/app/recent.py, ROK_CONFIG). The tests must not read the user's: a
setting changed in the app (Voxel grid on) once broke the picture goldens.
This runs before any test module imports Rok.
"""

import os
import tempfile

os.environ["ROK_CONFIG"] = tempfile.mkdtemp(prefix="rok-tests-")
