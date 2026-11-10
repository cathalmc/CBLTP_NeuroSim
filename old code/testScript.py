import os
import sys
import numpy as np
build_dir = "build"
configType = "Debug"

moduleDir = os.path.join(build_dir, configType)
if moduleDir not in sys.path:
    sys.path.append(moduleDir)

import cathalTest

cathalTest.paritySum(20)