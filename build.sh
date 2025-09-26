#!/bin/bash
SO_FILE="/home/david-james/miniconda3/envs/test-env/lib/python3.11/site-packages/cv2/python-3.11/cv2.cpython-311-x86_64-linux-gnu.so"

python -m nuitka \
  --standalone \
  --python-flag=-u \
  --enable-plugin=pyside6 \
  --remove-output \
  --user-package-configuration-file=mdpi.nuitka-package.config.yml \
  --include-package=cv2 \
  --include-data-files="${SO_FILE}=cv2/cv2.so" \
  --include-data-files="${SO_FILE}=cv2/python-3.11/" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libopencv_*.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libopenvino.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libavcodec.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libavformat.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libavutil.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libswscale.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libswresample.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libdav1d.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libSvtAv1Enc.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libavfilter.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libpostproc.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libaom.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libvpx.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libx264.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libx265.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libopenh264.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libtbb.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libOpenEXR*.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libva.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libva-*.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libdrm.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libIlmThread*.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libIex*.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libImath*.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libjasper.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libcblas.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/liblapack.so*=./" \
  --include-data-files="/home/david-james/miniconda3/envs/test-env/lib/libpugixml.so*=./" \
  --include-data-dir="/home/david-james/Desktop/04-MDPI/MDPI/pyqt_app/styles=pyqt_app/styles" \
  --include-data-dir="/home/david-james/Desktop/04-MDPI/MDPI/model=model" \
  --output-dir=./build \
  --output-filename=MDPI_Pipeline \
  ./pyqt_app/main.py
