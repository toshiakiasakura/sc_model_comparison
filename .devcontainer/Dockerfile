# Multi-arch tag: resolves to the native image (amd64 or arm64) of the host.
# Do not pin the x86_64- tag: on Apple Silicon it runs under Rosetta, where
# multi-threaded Julia (JULIA_NUM_THREADS > 1) segfaults during GC.
FROM quay.io/jupyter/datascience-notebook:julia-1.11.1

WORKDIR /workdir
EXPOSE 8888

ENV JULIA_PROJECT=/workdir
ENV DEVCONTAINER=TRUE

# TODO: Activate checkpoint and install packages.
RUN R -e "install.packages(c('socialmixr', 'jsonlite', 'here', 'checkpoint', 'cmoceean'), repos = 'https://cran.ism.ac.jp')"
RUN R -e "install.packages(c('tableone', 'broom'), repos = 'https://cran.ism.ac.jp')"

# Install Node.js 20 and Claude Code CLI
ARG CLAUDE_CODE_VERSION=latest
USER root
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} && \
    mkdir -p /home/jovyan/.julia /home/jovyan/.claude && \
    chown jovyan:users /home/jovyan/.julia /home/jovyan/.claude
USER jovyan