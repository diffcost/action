# The base image is pinned by digest, so a re-pointed tag can't change what a release runs.
# Update the digest when upgrading Python: docker pull python:3.14-slim && docker inspect --format='{{index .RepoDigests 0}}' python:3.14-slim
FROM python:3.14-slim@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f

# Install git (needed for diff operations)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
# requirements.txt is a lock (every dependency pinned with hashes, generated from
# requirements.in), so each build installs exactly the released versions
RUN pip install --no-cache-dir --require-hashes -r /requirements.txt

COPY src/ /src/

# GitHub runs Docker actions as root and mounts the checkout (owned by the runner's
# uid) at /github/workspace. Do not add a USER instruction: a non-root user cannot
# read the workspace or write the GITHUB_OUTPUT file. Mark the mounted repo as safe
# so git does not refuse it with "detected dubious ownership".
RUN git config --system --add safe.directory '*'

ENTRYPOINT ["python", "/src/main.py"]
