#!/usr/bin/env bash
# Run S1G through the syntactic clustering pipeline on EMR Serverless,
# with pair-based precision / recall / F1 metrics against truthABCgoodDQ.txt.
#
# Required env vars:
#   AWS_REGION                  e.g. us-east-1
#   S3_BUCKET                   bucket to stage code, data, output, logs
#   EMR_APPLICATION_ID          existing EMR Serverless Spark application id
#   EMR_EXECUTION_ROLE_ARN      IAM role assumed by the job
#
# Optional env vars:
#   S3_PREFIX                   key prefix under S3_BUCKET (default: s1g-emr-serverless)
#   JOB_NAME                    EMR job run name        (default: s1g-arcs-metrics)
#   CLUSTERING_METHOD           dbscan | arcs           (default: arcs)
#   DRIVER_CORES / DRIVER_MEM   Spark driver sizing     (default: 4 / 16g)
#   EXECUTOR_CORES / EXEC_MEM   Spark executor sizing   (default: 4 / 16g)
#   EXECUTOR_COUNT              dynamic alloc max execs (default: 8)
#   SHUFFLE_PARTITIONS                                  (default: 400)
#
# Exits non-zero if the StartJobRun call fails. Prints the JobRunId on success.

set -euo pipefail

: "${AWS_REGION:?AWS_REGION must be set}"
: "${S3_BUCKET:?S3_BUCKET must be set}"
: "${EMR_APPLICATION_ID:?EMR_APPLICATION_ID must be set}"
: "${EMR_EXECUTION_ROLE_ARN:?EMR_EXECUTION_ROLE_ARN must be set}"

S3_PREFIX="${S3_PREFIX:-s1g-emr-serverless}"
JOB_NAME="${JOB_NAME:-s1g-arcs-metrics}"
CLUSTERING_METHOD="${CLUSTERING_METHOD:-arcs}"
DRIVER_CORES="${DRIVER_CORES:-4}"
DRIVER_MEM="${DRIVER_MEM:-16g}"
EXECUTOR_CORES="${EXECUTOR_CORES:-4}"
EXEC_MEM="${EXEC_MEM:-16g}"
EXECUTOR_COUNT="${EXECUTOR_COUNT:-8}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-400}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/emr_serverless"
mkdir -p "${BUILD_DIR}"

S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX}"
S3_CODE="${S3_BASE}/code"
S3_DATA="${S3_BASE}/data"
S3_OUTPUT="${S3_BASE}/output/$(date -u +%Y%m%dT%H%M%SZ)"
S3_LOGS="${S3_BASE}/logs"

echo "Packaging pipeline source..."
(
  cd "${REPO_ROOT}"
  rm -f "${BUILD_DIR}/pipeline.zip"
  zip -qr "${BUILD_DIR}/pipeline.zip" pipeline -x "pipeline/__pycache__/*" "pipeline/*.pyc"
)

echo "Uploading code + data to ${S3_BASE}/ ..."
aws s3 cp "${BUILD_DIR}/pipeline.zip"            "${S3_CODE}/pipeline.zip"            --region "${AWS_REGION}"
aws s3 cp "${REPO_ROOT}/run_pipeline.py"         "${S3_CODE}/run_pipeline.py"         --region "${AWS_REGION}"
aws s3 cp "${REPO_ROOT}/S1G.txt"                 "${S3_DATA}/S1G.txt"                 --region "${AWS_REGION}"
aws s3 cp "${REPO_ROOT}/truthABCgoodDQ.txt"      "${S3_DATA}/truthABCgoodDQ.txt"      --region "${AWS_REGION}"

INPUT_URI="${S3_DATA}/S1G.txt"
TRUTH_URI="${S3_DATA}/truthABCgoodDQ.txt"

# Spark submit parameters. --py-files puts pipeline.zip on every worker's
# PYTHONPATH so `from pipeline.* import ...` resolves inside run_pipeline.py.
SPARK_SUBMIT_PARAMS=$(cat <<EOF
--conf spark.executor.cores=${EXECUTOR_CORES}
--conf spark.executor.memory=${EXEC_MEM}
--conf spark.driver.cores=${DRIVER_CORES}
--conf spark.driver.memory=${DRIVER_MEM}
--conf spark.dynamicAllocation.enabled=true
--conf spark.dynamicAllocation.minExecutors=1
--conf spark.dynamicAllocation.maxExecutors=${EXECUTOR_COUNT}
--conf spark.sql.shuffle.partitions=${SHUFFLE_PARTITIONS}
--conf spark.metrics.namespace=s1g
--conf spark.ui.prometheus.enabled=true
--conf spark.executor.processTreeMetrics.enabled=true
--conf spark.eventLog.enabled=true
--py-files ${S3_CODE}/pipeline.zip
EOF
)
# Collapse newlines so EMR accepts it as a single CLI string.
SPARK_SUBMIT_PARAMS=$(echo "${SPARK_SUBMIT_PARAMS}" | tr '\n' ' ')

# Pipeline CLI args. S1G.txt is CSV with a header; column 0 (RecID) is the
# join key for the truth file and is excluded from the clustering payload.
JOB_ARGS=$(cat <<EOF
[
  "--input",                "${INPUT_URI}",
  "--output",               "${S3_OUTPUT}",
  "--format",               "csv",
  "--has-header",
  "--skip-leading-columns", "1",
  "--ref-id-column",        "0",
  "--clustering-method",    "${CLUSTERING_METHOD}",
  "--shuffle-partitions",   "${SHUFFLE_PARTITIONS}",
  "--truth-file",           "${TRUTH_URI}"
]
EOF
)

JOB_DRIVER=$(cat <<EOF
{
  "sparkSubmit": {
    "entryPoint": "${S3_CODE}/run_pipeline.py",
    "entryPointArguments": ${JOB_ARGS},
    "sparkSubmitParameters": "${SPARK_SUBMIT_PARAMS}"
  }
}
EOF
)

# Send driver/executor logs to both S3 (durable, full Spark event log for
# History Server replay) and CloudWatch (live tail + metric filters).
MONITORING_CONFIG=$(cat <<EOF
{
  "managedPersistenceMonitoringConfiguration": { "enabled": true },
  "s3MonitoringConfiguration": {
    "logUri": "${S3_LOGS}/"
  },
  "cloudWatchLoggingConfiguration": {
    "enabled": true,
    "logGroupName": "/aws/emr-serverless/s1g",
    "logStreamNamePrefix": "${JOB_NAME}"
  }
}
EOF
)

echo "Submitting EMR Serverless job '${JOB_NAME}'..."
echo "  output : ${S3_OUTPUT}"
echo "  metrics: ${S3_OUTPUT}/_metrics/metrics.{json,log}"
echo "  logs   : ${S3_LOGS}/  +  CloudWatch /aws/emr-serverless/s1g"

RUN_ID=$(aws emr-serverless start-job-run \
  --region "${AWS_REGION}" \
  --application-id "${EMR_APPLICATION_ID}" \
  --execution-role-arn "${EMR_EXECUTION_ROLE_ARN}" \
  --name "${JOB_NAME}" \
  --job-driver "${JOB_DRIVER}" \
  --configuration-overrides "{\"monitoringConfiguration\": ${MONITORING_CONFIG}}" \
  --query 'jobRunId' --output text)

echo "JobRunId: ${RUN_ID}"
echo
echo "Tail status with:"
echo "  aws emr-serverless get-job-run \\"
echo "    --region ${AWS_REGION} --application-id ${EMR_APPLICATION_ID} \\"
echo "    --job-run-id ${RUN_ID} --query 'jobRun.state' --output text"
echo
echo "Once SUCCESS, fetch metrics with:"
echo "  aws s3 cp ${S3_OUTPUT}/_metrics/metrics.log -"
echo "  aws s3 cp ${S3_OUTPUT}/_metrics/metrics.json -"
