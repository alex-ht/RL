```
docker buildx build \
  --progress plain \
  --network host \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --tag alexht/nemo-rl \
  --build-arg SKIP_VLLM_BUILD=1 \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg MAX_JOBS=1 \
  --build-arg NVTE_BUILD_THREADS_PER_JOB=1 \
  .
```
