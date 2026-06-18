Benchmarking LLM Inference: A10 vs RTX 4090 vs L40S

We ran Qwen3.5-9B on three GPUs using vLLM (batch=1, 256/1024/2048 input tokens, 256 output tokens). The results reveal a clean split between prefill and decode — and the right GPU depends on which phase dominates your workload.

The Prefill vs Decode Tradeoff

Prefill (computing the first token) is compute-bound. The L40S delivered TTFT of 181ms vs 233ms for the 4090 on 2K inputs — a 22% advantage driven by its 362 TFLOPS (4× the 4090's 90). But here's the twist: that advantage is much smaller than the FLOPS gap would suggest, because even prefill has to load all 9B weights from HBM.

Decode (generating subsequent tokens) is bandwidth-bound. The 4090's 1,008 GB/s memory bandwidth gives it ~18ms ITL, vs 23ms for the L40S (864 GB/s) and 35ms for the A10 (600 GB/s). The correlation between bandwidth and decode latency is r = −0.997 — nearly perfect.

Why this matters for GPU selection

Low-latency chatbot (single user)? The L40S wins — first-token time is the UX metric that matters. High-throughput API serving? The 4090 delivers 55 tok/s vs 43 for the L40S, and at consumer pricing (~$1,600 vs $10,000+), it's not even close on $/token.

The surprise winner

The RTX 4090 punches far above its weight. It delivers 0.61 tokens/s per TFLOP (5× the L40S) and the best bandwidth utilization of any card tested. For most production workloads where decode dominates total latency, it's the pragmatic choice.

Bottom line: Know your bottleneck. Prefill needs compute. Decode needs bandwidth. Pick your GPU accordingly.
