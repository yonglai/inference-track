#!/bin/sh
# Send one tiny completion per second and print: time, HTTP status (000 = no connection), seconds.
# Usage: ./client_loop.sh http://<service-cluster-ip> [seconds-between-requests]
URL="${1:?usage: client_loop.sh http://<service-cluster-ip> [gap-seconds]}"
GAP="${2:-1}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
while true; do
  printf '%s ' "$(date -u +%T)"
  curl -s -o /dev/null --max-time 5 -w '%{http_code} %{time_total}s\n' \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Hello\",\"max_tokens\":4,\"temperature\":0.7,\"top_p\":0.9,\"top_k\":40}" \
    "$URL/v1/completions"
  sleep "$GAP"
done
