#!/bin/bash
# Log in to Delta. The connection stays open for 10 minutes after you exit, so
# sync_from_delta.sh can reuse it without another password and Duo prompt.
exec ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%C -o ControlPersist=10m \
  mconway@login.delta.ncsa.illinois.edu "$@"
