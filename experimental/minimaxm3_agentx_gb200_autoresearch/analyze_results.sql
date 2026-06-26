-- Sanitized, read-only analysis for the GB200 MiniMax M3 AgentX sweep and
-- the aggregate single-node B200 baseline. Connect with a caller-supplied
-- PostgreSQL URI; this file intentionally contains no credentials.
--
-- Workflow IDs:
--   GB200 MiniMax M3 FP8 Dynamo-vLLM: 28203718340 (expected 13 points)
--   B200 DSV4 FP4 vLLM baseline:      28154234456
--
-- Units and conventions:
--   *_tput_*: tokens/second
--   p90_e2el, p90_tpot, p90_ttft: seconds
--   required p90-tail interactivity: 1 / p90_tpot, tokens/second/user
--   stored p90_intvty is diagnostic because reciprocal percentile order differs.
--   effective_gpu_count is total_tput_tps / tput_per_gpu.
--
-- The baseline uses a different model, precision, and serving topology, so
-- comparisons are directional capacity checks, not model-normalized claims.

BEGIN READ ONLY;

SELECT wr.github_run_id,
       wr.run_attempt,
       wr.conclusion,
       wr.head_sha,
       count(br.id) AS ingested_points,
       count(br.trace_replay_id) AS points_with_trace_replay
FROM workflow_runs wr
LEFT JOIN benchmark_results br ON br.workflow_run_id = wr.id
WHERE wr.github_run_id IN (28203718340, 28154234456)
GROUP BY wr.github_run_id, wr.run_attempt, wr.conclusion, wr.head_sha
ORDER BY wr.github_run_id DESC;

WITH selected AS (
  SELECT wr.github_run_id,
         br.id AS result_id,
         br.conc,
         br.offload_mode,
         br.error,
         br.trace_replay_id,
         c.hardware,
         c.framework,
         c.model,
         c.precision,
         c.is_multinode,
         c.disagg,
         br.metrics
  FROM benchmark_results br
  JOIN workflow_runs wr ON wr.id = br.workflow_run_id
  JOIN configs c ON c.id = br.config_id
  WHERE wr.github_run_id IN (28203718340, 28154234456)
)
SELECT github_run_id,
       result_id,
       hardware,
       framework,
       model,
       precision,
       is_multinode,
       disagg,
       conc,
       offload_mode,
       round((metrics->>'duration_seconds')::numeric, 2) AS duration_s,
       round((metrics->>'total_tput_tps')::numeric, 2) AS total_tput_tps,
       round((metrics->>'tput_per_gpu')::numeric, 2) AS tput_per_gpu,
       round((metrics->>'total_tput_tps')::numeric /
             nullif((metrics->>'tput_per_gpu')::numeric, 0), 0) AS effective_gpu_count,
       round((metrics->>'input_tput_tps')::numeric, 2) AS input_tput_tps,
       round((metrics->>'output_tput_tps')::numeric, 2) AS output_tput_tps,
       round((metrics->>'output_tput_per_gpu')::numeric, 2) AS output_tput_per_gpu,
       round((metrics->>'p90_e2el')::numeric, 3) AS p90_e2e_s,
       round((metrics->>'p90_tpot')::numeric, 6) AS p90_tpot_s,
       round(1 / nullif((metrics->>'p90_tpot')::numeric, 0), 2) AS p90_tail_interactivity_tps,
       round((metrics->>'p90_intvty')::numeric, 2) AS stored_p90_intvty,
       round((metrics->>'p90_ttft')::numeric, 3) AS p90_ttft_s,
       (metrics->>'num_requests_successful')::integer AS successful_requests,
       (metrics->>'num_requests_total')::integer AS total_requests,
       (metrics->>'total_requests_completed')::integer AS completed_requests,
       round((metrics->>'response_cache_hit_rate')::numeric, 4) AS response_cache_hit_rate,
       round((metrics->>'server_external_cache_hit_rate')::numeric, 4) AS external_cache_hit_rate,
       round((metrics->>'server_gpu_cache_hit_rate')::numeric, 4) AS gpu_cache_hit_rate,
       round((metrics->>'gpu_kv_cache_usage_pct')::numeric, 2) AS gpu_kv_cache_usage_pct,
       round((metrics->>'cpu_kv_cache_usage_pct')::numeric, 2) AS cpu_kv_cache_usage_pct,
       error,
       trace_replay_id
FROM selected
ORDER BY github_run_id DESC, result_id;

SELECT br.id AS result_id,
       br.conc,
       br.offload_mode,
       round((br.metrics->>'duration_seconds')::numeric, 2) AS duration_s,
       br.trace_replay_id
FROM benchmark_results br
JOIN workflow_runs wr ON wr.id = br.workflow_run_id
WHERE wr.github_run_id = 28203718340
  AND (br.metrics->>'duration_seconds')::numeric < 3500
ORDER BY br.id;

COMMIT;
