#!/usr/bin/env bash
#
#

set -eo pipefail
# Automatically export
set -a

# shellcheck source=/dev/null
source .env
set +a

alias dgd='dg dev --host 0.0.0.0'
