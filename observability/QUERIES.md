# BIBLOS v2 - Observability Queries

This document provides example queries for debugging and analyzing BIBLOS v2
performance using Jaeger, Tempo, and PromQL (Prometheus/Grafana).

## Table of Contents

1. [Jaeger Trace Queries](#jaeger-trace-queries)
2. [Tempo TraceQL Queries](#tempo-traceql-queries)
3. [PromQL Metrics Queries](#promql-metrics-queries)
4. [Common Debugging Scenarios](#common-debugging-scenarios)
5. [Dashboard Panels](#dashboard-panels)

---

## Jaeger Trace Queries

### Find All Pipeline Executions
```
service=biblos-v2 operation="pipeline.execute"
```

### Find Slow Pipeline Executions (>5s)
```
service=biblos-v2 operation="pipeline.execute" minDuration=5s
```

### Find Failed Pipeline Executions
```
service=biblos-v2 operation="pipeline.execute" error=true
```

### Find Executions for Specific Verse
```
service=biblos-v2 tags={"verse.id":"GEN.1.1"}
```

### Find All Agent Extractions
```
service=biblos-v2 operation=~"agent\..*\.process"
```

### Find Slow Agents (>2s)
```
service=biblos-v2 operation=~"agent\..*\.extract" minDuration=2s
```

### Find ML Inference Operations
```
service=biblos-v2 operation=~"ml\.inference\..*"
```

### Find Database Operations
```
service=biblos-v2 operation=~"db\..*"
```

---

## Tempo TraceQL Queries

### Find All Pipeline Traces
```traceql
{span.service.name="biblos-v2" && name="pipeline.execute"}
```

### Find Slow Pipelines with Duration
```traceql
{span.service.name="biblos-v2" && name="pipeline.execute" && duration > 5s}
```

### Find Failed Operations
```traceql
{span.service.name="biblos-v2" && status=error}
```

### Find Specific Verse Processing
```traceql
{span.service.name="biblos-v2" && span.verse.id="GEN.1.1"}
```

### Find Low Confidence Results
```traceql
{span.service.name="biblos-v2" && name=~"agent.*" && span.result.confidence < 0.5}
```

### Find Embedding Generation
```traceql
{span.service.name="biblos-v2" && name="ml.inference.embed"}
```

### Find Cross-Reference Discovery
```traceql
{span.service.name="biblos-v2" && name="ml.inference.infer" && span.result.candidate_count > 0}
```

### Find Slow Database Queries
```traceql
{span.service.name="biblos-v2" && name=~"db.*" && duration > 100ms}
```

### Find Cache Misses
```traceql
{span.service.name="biblos-v2" && span.cache.hit=false}
```

### Find API Requests by Endpoint
```traceql
{span.service.name="biblos-api" && name=~"/api/v1/.*"}
```

---

## PromQL Metrics Queries

### Pipeline Metrics

#### Pipeline Duration (p99)
```promql
histogram_quantile(0.99, rate(biblos_pipeline_duration_seconds_bucket[5m]))
```

#### Pipeline Success Rate
```promql
sum(rate(biblos_verses_processed_total{status="completed"}[5m])) /
sum(rate(biblos_verses_processed_total[5m]))
```

#### Pipeline Throughput (verses/minute)
```promql
sum(rate(biblos_verses_processed_total[1m])) * 60
```

### Phase Metrics

#### Phase Duration by Phase
```promql
histogram_quantile(0.95,
  rate(biblos_phase_duration_seconds_bucket[5m])
) by (phase)
```

#### Slowest Phases
```promql
topk(5,
  avg(rate(biblos_phase_duration_seconds_sum[5m]) /
      rate(biblos_phase_duration_seconds_count[5m])) by (phase)
)
```

### Agent Metrics

#### Agent Duration Distribution
```promql
histogram_quantile(0.95,
  rate(biblos_agent_duration_seconds_bucket[5m])
) by (agent)
```

#### Agent Confidence Distribution
```promql
histogram_quantile(0.50,
  rate(biblos_agent_confidence_bucket[5m])
) by (agent)
```

#### Agent Error Rate
```promql
sum(rate(biblos_extraction_errors_total[5m])) by (agent) /
sum(rate(biblos_verses_processed_total[5m])) by (agent)
```

### ML Metrics

#### ML Inference Latency (p99)
```promql
histogram_quantile(0.99,
  rate(biblos_ml_inference_duration_seconds_bucket[5m])
) by (operation)
```

#### Embedding Generation Time
```promql
histogram_quantile(0.95,
  rate(biblos_embedding_generation_duration_seconds_bucket[5m])
)
```

#### Cross-Reference Discovery Rate
```promql
sum(rate(biblos_cross_references_discovered_total[5m])) by (connection_type)
```

### Cache Metrics

#### Cache Hit Rate
```promql
sum(rate(biblos_cache_hits_total[5m])) /
(sum(rate(biblos_cache_hits_total[5m])) + sum(rate(biblos_cache_misses_total[5m])))
```

#### Cache Hit Rate by Type
```promql
sum(rate(biblos_cache_hits_total[5m])) by (cache_type) /
(sum(rate(biblos_cache_hits_total[5m])) by (cache_type) +
 sum(rate(biblos_cache_misses_total[5m])) by (cache_type))
```

### Database Metrics

#### Database Query Latency
```promql
histogram_quantile(0.95,
  rate(biblos_db_query_duration_seconds_bucket[5m])
) by (database, operation)
```

#### Active Database Connections
```promql
biblos_db_connections_active
```

### API Metrics

#### API Request Duration (p95)
```promql
histogram_quantile(0.95,
  rate(biblos_api_request_duration_seconds_bucket[5m])
) by (endpoint)
```

#### API Request Rate by Endpoint
```promql
sum(rate(biblos_api_requests_total[5m])) by (endpoint, method)
```

#### API Error Rate
```promql
sum(rate(biblos_api_requests_total{status_code=~"5.."}[5m])) /
sum(rate(biblos_api_requests_total[5m]))
```

---

## Common Debugging Scenarios

### Scenario 1: Slow Pipeline Execution

**Symptoms:** Pipeline takes >10s for a single verse

**Investigation Steps:**
1. Find the trace:
   ```traceql
   {span.service.name="biblos-v2" && name="pipeline.execute" && duration > 10s}
   ```

2. Check which phase is slow:
   ```promql
   topk(1, rate(biblos_phase_duration_seconds_sum[5m]) by (phase))
   ```

3. Drill into the specific phase span to see agent breakdown

4. Check for database bottlenecks:
   ```traceql
   {span.service.name="biblos-v2" && name=~"db.*" && duration > 1s}
   ```

### Scenario 2: High Error Rate

**Symptoms:** Many failed extractions

**Investigation Steps:**
1. Find error distribution:
   ```promql
   sum(rate(biblos_extraction_errors_total[5m])) by (agent)
   ```

2. Find sample error traces:
   ```traceql
   {span.service.name="biblos-v2" && status=error}
   ```

3. Check error messages in span attributes

4. Correlate with logs using trace_id

### Scenario 3: Low Confidence Results

**Symptoms:** Many results with confidence <0.5

**Investigation Steps:**
1. Check confidence distribution:
   ```promql
   histogram_quantile(0.25, rate(biblos_agent_confidence_bucket[5m])) by (agent)
   ```

2. Find low-confidence extractions:
   ```traceql
   {span.service.name="biblos-v2" && span.result.confidence < 0.5}
   ```

3. Review input data for problematic verses

### Scenario 4: API Latency Spikes

**Symptoms:** API response times increase suddenly

**Investigation Steps:**
1. Check overall latency trend:
   ```promql
   histogram_quantile(0.99, rate(biblos_api_request_duration_seconds_bucket[5m]))
   ```

2. Identify slow endpoints:
   ```promql
   topk(3, histogram_quantile(0.95, rate(biblos_api_request_duration_seconds_bucket[5m])) by (endpoint))
   ```

3. Find slow API traces:
   ```traceql
   {span.service.name="biblos-api" && duration > 5s}
   ```

4. Check downstream dependencies (ML, DB)

### Scenario 5: Memory/Cache Issues

**Symptoms:** Cache hit rate drops

**Investigation Steps:**
1. Monitor cache hit rate:
   ```promql
   sum(rate(biblos_cache_hits_total[5m])) /
   (sum(rate(biblos_cache_hits_total[5m])) + sum(rate(biblos_cache_misses_total[5m])))
   ```

2. Find cache miss patterns:
   ```traceql
   {span.service.name="biblos-v2" && span.cache.hit=false}
   ```

3. Check embedding cache specifically:
   ```promql
   biblos_cache_hits_total{cache_type="embedding"} /
   (biblos_cache_hits_total{cache_type="embedding"} + biblos_cache_misses_total{cache_type="embedding"})
   ```

---

## Dashboard Panels

### Pipeline Overview Dashboard

```json
{
  "panels": [
    {
      "title": "Pipeline Throughput",
      "type": "graph",
      "query": "sum(rate(biblos_verses_processed_total[1m])) * 60"
    },
    {
      "title": "Pipeline Success Rate",
      "type": "gauge",
      "query": "sum(rate(biblos_verses_processed_total{status=\"completed\"}[5m])) / sum(rate(biblos_verses_processed_total[5m]))"
    },
    {
      "title": "Pipeline Duration (p95)",
      "type": "graph",
      "query": "histogram_quantile(0.95, rate(biblos_pipeline_duration_seconds_bucket[5m]))"
    },
    {
      "title": "Phase Duration Breakdown",
      "type": "barchart",
      "query": "avg(rate(biblos_phase_duration_seconds_sum[5m]) / rate(biblos_phase_duration_seconds_count[5m])) by (phase)"
    }
  ]
}
```

### Agent Performance Dashboard

```json
{
  "panels": [
    {
      "title": "Agent Duration by Type",
      "type": "heatmap",
      "query": "rate(biblos_agent_duration_seconds_bucket[5m])"
    },
    {
      "title": "Agent Confidence Distribution",
      "type": "histogram",
      "query": "rate(biblos_agent_confidence_bucket[5m])"
    },
    {
      "title": "Agent Error Rate",
      "type": "graph",
      "query": "sum(rate(biblos_extraction_errors_total[5m])) by (agent)"
    }
  ]
}
```

### ML Inference Dashboard

```json
{
  "panels": [
    {
      "title": "Embedding Generation Latency",
      "type": "graph",
      "query": "histogram_quantile(0.95, rate(biblos_embedding_generation_duration_seconds_bucket[5m]))"
    },
    {
      "title": "Cross-References Discovered",
      "type": "counter",
      "query": "sum(increase(biblos_cross_references_discovered_total[1h]))"
    },
    {
      "title": "Connection Type Distribution",
      "type": "piechart",
      "query": "sum(rate(biblos_cross_references_discovered_total[5m])) by (connection_type)"
    }
  ]
}
```

---

## Trace ID Correlation

### Finding Logs by Trace ID

When you have a trace ID from Jaeger/Tempo, you can find corresponding logs:

```bash
# In your log aggregation system (Loki, Elasticsearch)
{job="biblos-v2"} |= "trace_id=<YOUR_TRACE_ID>"
```

### Adding Trace ID to API Responses

All API responses include an `X-Trace-ID` header. Use this to correlate:

```bash
curl -v http://localhost:8000/api/v1/extract -d '...' 2>&1 | grep X-Trace-ID
```

### Linking Traces to Metrics

Use the trace exemplars feature in Grafana to link metrics to traces:

```promql
histogram_quantile(0.99, rate(biblos_pipeline_duration_seconds_bucket[5m]))
# Click on data point to see linked traces
```

---

## Alerting Rules

### Critical Alerts

```yaml
# Pipeline failure rate > 10%
- alert: BiblosPipelineHighErrorRate
  expr: |
    sum(rate(biblos_verses_processed_total{status!="completed"}[5m])) /
    sum(rate(biblos_verses_processed_total[5m])) > 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "BIBLOS pipeline error rate is above 10%"

# Pipeline latency > 30s (p95)
- alert: BiblosPipelineHighLatency
  expr: |
    histogram_quantile(0.95, rate(biblos_pipeline_duration_seconds_bucket[5m])) > 30
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "BIBLOS pipeline latency is high"

# Database query latency > 5s
- alert: BiblosDatabaseSlowQueries
  expr: |
    histogram_quantile(0.95, rate(biblos_db_query_duration_seconds_bucket[5m])) > 5
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Database queries are slow"
```

---

## Quick Reference

| Metric | Description | Unit |
|--------|-------------|------|
| `biblos_pipeline_duration_seconds` | Total pipeline execution time | seconds |
| `biblos_phase_duration_seconds` | Phase execution time | seconds |
| `biblos_agent_duration_seconds` | Agent extraction time | seconds |
| `biblos_verses_processed_total` | Counter of processed verses | count |
| `biblos_cross_references_discovered_total` | Counter of discovered cross-refs | count |
| `biblos_ml_inference_duration_seconds` | ML inference latency | seconds |
| `biblos_cache_hits_total` | Cache hit counter | count |
| `biblos_cache_misses_total` | Cache miss counter | count |
| `biblos_db_query_duration_seconds` | Database query latency | seconds |
| `biblos_api_request_duration_seconds` | API request latency | seconds |

| Span Attribute | Description |
|----------------|-------------|
| `verse.id` | Canonical verse identifier |
| `phase.name` | Pipeline phase name |
| `agent.name` | Agent name |
| `result.confidence` | Extraction confidence score |
| `processing_time_ms` | Processing time in milliseconds |
| `cache.hit` | Whether cache was hit |
| `db.system` | Database system (postgres, neo4j, redis) |
| `error.message` | Error message if failed |
