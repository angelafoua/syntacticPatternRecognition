# Running S1G on EMR Serverless with metrics

End-to-end recipe for running `S1G.txt` through the syntactic clustering
pipeline on AWS EMR Serverless, scored against `truthABCgoodDQ.txt` for
pair-based precision / recall / F1.

## Prerequisites

- AWS CLI configured with credentials that can create EMR Serverless
  applications, start job runs, and read/write the staging S3 bucket.
- An execution-role ARN that EMR Serverless can assume. The role needs:
  - `s3:GetObject` / `s3:PutObject` on the staging bucket
  - `logs:CreateLogStream` / `logs:PutLogEvents` on
    `/aws/emr-serverless/s1g`
- An S3 bucket in the same region as the EMR Serverless application.

## 1. Create the application (one time)

```bash
export AWS_REGION=us-east-1
./scripts/emr_serverless/create_application.sh
# prints: export EMR_APPLICATION_ID=00f...
```

## 2. Submit the S1G job

```bash
export AWS_REGION=us-east-1
export S3_BUCKET=my-er-bucket
export EMR_APPLICATION_ID=00f...
export EMR_EXECUTION_ROLE_ARN=arn:aws:iam::123456789012:role/EMRServerlessJobRole

./scripts/emr_serverless/submit_s1g.sh
```

The script:
1. Zips the local `pipeline/` package.
2. Uploads `pipeline.zip`, `run_pipeline.py`, `S1G.txt`, and
   `truthABCgoodDQ.txt` under `s3://$S3_BUCKET/$S3_PREFIX/`.
3. Calls `StartJobRun` with the run arguments below.
4. Prints the `JobRunId` and the S3 path where metrics will land.

### Run arguments

| Flag | Value | Why |
|---|---|---|
| `--input`                | `s3://.../S1G.txt`            | staged input |
| `--format`               | `csv`                         | S1G is comma-delimited |
| `--has-header`           | (set)                         | first row is `RecID,fname,...` |
| `--skip-leading-columns` | `1`                           | drop `RecID` from clustering payload |
| `--ref-id-column`        | `0`                           | still capture `RecID` for metrics |
| `--truth-file`           | `s3://.../truthABCgoodDQ.txt` | enables pair metrics |
| `--clustering-method`    | `arcs` (override via env)     | ARCS path; switch to `dbscan` if desired |

## 3. Read the metrics

When the job completes the pipeline writes two files alongside the parquet
output:

```
s3://$S3_BUCKET/$S3_PREFIX/output/<timestamp>/_metrics/metrics.json
s3://$S3_BUCKET/$S3_PREFIX/output/<timestamp>/_metrics/metrics.log
```

`metrics.json` includes precision, recall, F1, TP / FP / FN, linked /
expected pair counts, predicted-cluster size distribution, truth-cluster
size distribution, and per-stage timings. `metrics.log` is a human-readable
rendering of the same data.

Tail-and-grep example:

```bash
aws s3 cp "$(aws s3 ls s3://$S3_BUCKET/$S3_PREFIX/output/ \
  | sort | tail -1 | awk '{print $2}')metrics.log" -
```

## 4. Infrastructure metrics

- **CloudWatch**: driver + executor logs stream to log group
  `/aws/emr-serverless/s1g`.
- **S3 logs**: full Spark event log lands under
  `s3://$S3_BUCKET/$S3_PREFIX/logs/` and can be replayed in the EMR
  Serverless-managed Spark History Server (Application -> "Spark UI").
- **Per-executor metrics**: the submitter enables
  `spark.ui.prometheus.enabled` and `processTreeMetrics`, so CPU /
  memory / GC are visible in the History Server and in
  `AWS/EMRServerless` CloudWatch metrics.
