#!/usr/bin/env bash
# One-time setup: create an EMR Serverless Spark application sized for S1G.
# Prints the application id so you can export it as EMR_APPLICATION_ID before
# calling submit_s1g.sh.
#
# Required env vars:
#   AWS_REGION                  e.g. us-east-1
#
# Optional env vars:
#   APP_NAME                    default: syntactic-clustering
#   RELEASE_LABEL               default: emr-7.2.0  (Spark 3.5)
#   MAX_CPU                     pre-init max vCPU      (default: 32)
#   MAX_MEM_GB                  pre-init max memory GB (default: 128)

set -euo pipefail

: "${AWS_REGION:?AWS_REGION must be set}"
APP_NAME="${APP_NAME:-syntactic-clustering}"
RELEASE_LABEL="${RELEASE_LABEL:-emr-7.2.0}"
MAX_CPU="${MAX_CPU:-32}"
MAX_MEM_GB="${MAX_MEM_GB:-128}"

APP_ID=$(aws emr-serverless create-application \
  --region "${AWS_REGION}" \
  --name "${APP_NAME}" \
  --type "SPARK" \
  --release-label "${RELEASE_LABEL}" \
  --maximum-capacity "cpu=${MAX_CPU}vCPU,memory=${MAX_MEM_GB}GB" \
  --auto-stop-configuration "enabled=true,idleTimeoutMinutes=15" \
  --query 'applicationId' --output text)

echo "Created application: ${APP_ID}"
aws emr-serverless start-application --region "${AWS_REGION}" --application-id "${APP_ID}"
echo "Started.  export EMR_APPLICATION_ID=${APP_ID}"
