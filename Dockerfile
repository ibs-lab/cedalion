###############################################################################
# cedalion development environment
#
#   docker build -t cedalion .
#   docker run -it --rm cedalion                              # shell in the env
#   docker run -it --rm -p 8888:8888 cedalion \
#       jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
#
# Optional photon transport solver (see install_nirfaster.sh):
#   docker build -t cedalion --build-arg INSTALL_NIRFASTER=CPU .
###############################################################################

# environment_dev.yml declares conda-forge as its only channel, so miniforge is
# the matching base image. It is also the conda distribution used by CI
# (.github/workflows/run_tests.yml).
ARG CONDA_IMAGE=condaforge/miniforge3:26.3.2-3
FROM ${CONDA_IMAGE}

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git          - hatch-vcs derives cedalion's version from it during the install
# libgl*/libx* - runtime libraries of VTK and pyvista
# xvfb, xauth  - offscreen rendering. xvfb-run needs xauth, which the xvfb
#                package only recommends and which --no-install-recommends
#                therefore skips.
# curl, unzip  - used by install_nirfaster.sh
# hadolint ignore=DL3008
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libgl1-mesa-dri \
        libglib2.0-0 \
        libglx-mesa0 \
        libsm6 \
        libxext6 \
        libxkbcommon0 \
        libxrender1 \
        openssh-client \
        procps \
        unzip \
        xauth \
        xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /cedalion

# The conda environment is created before the source tree is copied, so that
# editing cedalion does not invalidate this expensive layer.
#
# environment_dev.yml ends with an editable install of ./workflows, resolved
# relative to the working directory, so that directory is needed here as well.
# conda in miniforge resolves with the libmamba solver by default.
COPY environment_dev.yml ./
COPY workflows ./workflows
RUN conda env create -n cedalion -f environment_dev.yml && \
    conda clean -afy && \
    find /opt/conda/ -follow -type f -name '*.a' -delete

# Put the environment first on PATH, so that python/pip/pytest also resolve in
# non-interactive calls such as `docker exec`. Interactive shells additionally
# get a full `conda activate` from .bashrc.
ENV PATH=/opt/conda/envs/cedalion/bin:${PATH}
RUN echo "conda activate cedalion" >> /root/.bashrc

# The source tree is copied last. .git is deliberately included (see
# .dockerignore): hatch-vcs needs it to write src/cedalion/_version.py.
COPY . .
RUN pip install --no-cache-dir -e . --no-deps && \
    python -c "import cedalion; print('cedalion', cedalion.__version__)"

# Optional: NIRFASTer, the photon propagator used by cedalion.dot.forward_model
# that also runs on the CPU.
ARG INSTALL_NIRFASTER=""
RUN if [ -n "${INSTALL_NIRFASTER}" ]; then \
        bash install_nirfaster.sh "${INSTALL_NIRFASTER}"; \
    fi

# pyvista renders offscreen against the Xvfb server that the entrypoint starts
# on this display.
ENV DISPLAY=:99 \
    PYVISTA_OFF_SCREEN=true

EXPOSE 8888

# Start an X virtual framebuffer, which pyvista's 3D plotting needs in a
# headless container, and then hand the command PID 1 with `exec`, so that it
# receives the TTY and signals such as `docker stop`. cedalion.def does the
# equivalent for Apptainer.
#
# Xvfb is started directly rather than through xvfb-run: as an entrypoint, that
# is PID 1, xvfb-run hung waiting for the SIGUSR1 readiness handshake with Xvfb,
# and its default error file of /dev/null hid the reason. Errors are kept in
# /tmp/xvfb.log here. Bypass the entrypoint with `--entrypoint ""`.
ENTRYPOINT ["/bin/sh", "-c", "Xvfb \"$DISPLAY\" -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 & sleep 1; exec \"$@\"", "--"]
CMD ["bash"]
