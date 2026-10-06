# CPU-only research environment for this benchmark: learned (KalmanNet family)
# vs. classical (IMM) filtering for maneuvering target tracking. CPU-only by
# design -- the eventual deployment target (Jetson Orin gimbal tracker)
# reserves the GPU for the vision pipeline, so training and inference both run
# on CPU here, matching deployment reality end to end.
#
# Build and run via ./docker_run.sh, which wires the mount and GIT_COMMIT.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
# Entrypoints live at the repo root and import `data.` / `filters.` packages;
# running one as `python data/make_dataset.py` otherwise puts only its own dir on sys.path.
ENV PYTHONPATH=/work
ENV OMP_NUM_THREADS=1

# ffmpeg: matplotlib needs it to write mp4 animations (make_animation.py). The
# pillow writer (already bundled) covers GIF; ffmpeg is only for the mp4 path.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

WORKDIR /work
CMD ["bash"]
