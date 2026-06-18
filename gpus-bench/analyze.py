import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DARK_BG = "#0d1117"
CARD_BG = "#161b22"
GRID_COLOR = "#30363d"
TEXT_COLOR = "#e6edf3"
AXIS_COLOR = "#8b949e"

GPU_SPECS = {
    "A10":    {"bw": 600,  "flops": 62,  "vram": 24},
    "RTX 4090": {"bw": 1008, "flops": 90,  "vram": 24},
    "L40S":   {"bw": 864,  "flops": 362, "vram": 48},
}

RESULTS = {
    "RTX 4090": {
        256:  {"ttft": 60.50,  "itl": 17.77, "tpot": 17.77, "out_tok": 55.74, "p50_ttft": 60.33,  "p99_ttft": 64.59,  "p50_itl": 17.77, "p99_itl": 18.56},
        1024: {"ttft": 121.77, "itl": 17.92, "tpot": 17.92, "out_tok": 54.55, "p50_ttft": 122.54, "p99_ttft": 127.85, "p50_itl": 17.93, "p99_itl": 18.83},
        2048: {"ttft": 232.99, "itl": 18.02, "tpot": 18.02, "out_tok": 53.02, "p50_ttft": 234.70, "p99_ttft": 238.75, "p50_itl": 18.04, "p99_itl": 18.75},
    },
    "L40S": {
        256:  {"ttft": 56.99,  "itl": 22.64, "tpot": 22.64, "out_tok": 43.91, "p50_ttft": 56.55,  "p99_ttft": 59.93,  "p50_itl": 22.64, "p99_itl": 22.97},
        1024: {"ttft": 101.33, "itl": 22.83, "tpot": 22.83, "out_tok": 43.21, "p50_ttft": 101.49, "p99_ttft": 105.82, "p50_itl": 22.84, "p99_itl": 23.21},
        2048: {"ttft": 180.85, "itl": 22.99, "tpot": 22.99, "out_tok": 42.36, "p50_ttft": 180.54, "p99_ttft": 184.57, "p50_itl": 22.99, "p99_itl": 23.32},
    },
    "A10": {
        256:  {"ttft": 81.81,  "itl": 35.05, "tpot": 35.05, "out_tok": 28.38, "p50_ttft": 81.91,  "p99_ttft": 84.05,  "p50_itl": 35.05, "p99_itl": 35.46},
        1024: {"ttft": 255.63, "itl": 35.10, "tpot": 35.10, "out_tok": 27.80, "p50_ttft": 255.96, "p99_ttft": 260.52, "p50_itl": 35.11, "p99_itl": 35.42},
        2048: {"ttft": 500.82, "itl": 35.23, "tpot": 35.23, "out_tok": 26.99, "p50_ttft": 506.09, "p99_ttft": 510.31, "p50_itl": 35.24, "p99_itl": 35.64},
    },
}

INPUT_LENS = [256, 1024, 2048]
GPU_ORDER = ["A10", "L40S", "RTX 4090"]
GPU_COLORS = {"A10": "#ff6b35", "L40S": "#00d4ff", "RTX 4090": "#ff2e7d"}

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "xtick.color": AXIS_COLOR,
    "ytick.color": AXIS_COLOR,
    "text.color": TEXT_COLOR,
    "legend.facecolor": CARD_BG,
    "legend.edgecolor": GRID_COLOR,
    "legend.labelcolor": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,
    "font.size": 12,
    "axes.titlesize": 15,
})


def grouped_bar(ax, metrics_dict, ylabel, title, y_formatter=None):
    x = np.arange(len(INPUT_LENS))
    n = len(GPU_ORDER)
    w = 0.22
    for i, gpu in enumerate(GPU_ORDER):
        vals = [metrics_dict[gpu][l] for l in INPUT_LENS]
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=gpu, color=GPU_COLORS[gpu], edgecolor="white", linewidth=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l} tokens" for l in INPUT_LENS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=10)
    if y_formatter:
        ax.yaxis.set_major_formatter(y_formatter)


def plot_all():
    # 1. TTFT Bar
    fig, ax = plt.subplots(figsize=(8, 5))
    ttft_data = {gpu: {l: RESULTS[gpu][l]["ttft"] for l in INPUT_LENS} for gpu in GPU_ORDER}
    grouped_bar(ax, ttft_data, "TTFT (ms)", "Mean Time to First Token per Input Length")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_ttft_bar.png"), dpi=150, facecolor=DARK_BG)
    plt.close(fig)

    # 2. ITL Bar
    fig, ax = plt.subplots(figsize=(8, 5))
    itl_data = {gpu: {l: RESULTS[gpu][l]["itl"] for l in INPUT_LENS} for gpu in GPU_ORDER}
    grouped_bar(ax, itl_data, "ITL (ms)", "Mean Inter-Token Latency per Input Length")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_itl_bar.png"), dpi=150, facecolor=DARK_BG)
    plt.close(fig)

    # 3. Throughput Bar
    fig, ax = plt.subplots(figsize=(8, 5))
    tp_data = {gpu: {l: RESULTS[gpu][l]["out_tok"] for l in INPUT_LENS} for gpu in GPU_ORDER}
    grouped_bar(ax, tp_data, "Output Tokens / s", "Output Token Throughput per Input Length")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_throughput_bar.png"), dpi=150, facecolor=DARK_BG)
    plt.close(fig)

    # 4. Bandwidth vs ITL scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    mean_itl = {gpu: np.mean([RESULTS[gpu][l]["itl"] for l in INPUT_LENS]) for gpu in GPU_ORDER}
    bws = [GPU_SPECS[gpu]["bw"] for gpu in GPU_ORDER]
    itls = [mean_itl[gpu] for gpu in GPU_ORDER]
    ax.scatter(bws, itls, c=[GPU_COLORS[gpu] for gpu in GPU_ORDER], s=200, edgecolors="white", linewidth=1.2, zorder=5)
    for gpu, bw, itl in zip(GPU_ORDER, bws, itls):
        ax.annotate(f"{gpu}\n({bw} GB/s, {itl:.2f} ms)", (bw, itl),
                    textcoords="offset points", xytext=(15, 10), fontsize=10, color=TEXT_COLOR,
                    arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=0.8))
    # simple trend
    z = np.polyfit(bws, itls, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(bws) - 50, max(bws) + 50, 100)
    ax.plot(x_line, p(x_line), "--", color=AXIS_COLOR, alpha=0.5, label=f"Trend (R²≈{np.corrcoef(bws, itls)[0,1]**2:.3f})")
    ax.set_xlabel("Memory Bandwidth (GB/s)")
    ax.set_ylabel("Mean ITL (ms)")
    ax.set_title("Bandwidth vs Inter-Token Latency")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_bw_vs_itl.png"), dpi=150, facecolor=DARK_BG)
    plt.close(fig)

    # 5. FLOPS vs TTFT scatter (at 2048 input)
    fig, ax = plt.subplots(figsize=(7, 5))
    flops_vals = [GPU_SPECS[gpu]["flops"] for gpu in GPU_ORDER]
    ttft_2048 = [RESULTS[gpu][2048]["ttft"] for gpu in GPU_ORDER]
    ax.scatter(flops_vals, ttft_2048, c=[GPU_COLORS[gpu] for gpu in GPU_ORDER], s=200, edgecolors="white", linewidth=1.2, zorder=5)
    offsets = {"A10": (0, -22), "L40S": (-15, 10), "RTX 4090": (10, 10)}
    for gpu, flops, ttft in zip(GPU_ORDER, flops_vals, ttft_2048):
        ax.annotate(f"{gpu}\n({flops} TFLOPS, {ttft:.1f} ms)", (flops, ttft),
                    textcoords="offset points", xytext=offsets[gpu], fontsize=10, color=TEXT_COLOR,
                    arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=0.8))
    ax.set_xlabel("BF16 TFLOPS")
    ax.set_ylabel("TTFT @ 2048 input (ms)")
    ax.set_title("Compute (FLOPS) vs Time to First Token")
    ax.set_xscale("log")
    ax.text(0.95, 0.95, "Prefill is BW-bound too:\nL40S has 4× FLOPS but\nonly ~22% lower TTFT",
            transform=ax.transAxes, va="top", ha="right", fontsize=9, color=AXIS_COLOR, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD_BG, edgecolor=GRID_COLOR, alpha=0.9))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_flops_vs_ttft.png"), dpi=150, facecolor=DARK_BG)
    plt.close(fig)

    # 6. P50 / P99 TTFT distribution per GPU (averaged across input lengths)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(GPU_ORDER))
    w = 0.30
    avg_p50 = [np.mean([RESULTS[gpu][l]["p50_ttft"] for l in INPUT_LENS]) for gpu in GPU_ORDER]
    avg_p99 = [np.mean([RESULTS[gpu][l]["p99_ttft"] for l in INPUT_LENS]) for gpu in GPU_ORDER]
    b1 = ax.bar(x - w / 2, avg_p50, w, label="P50 (Median)", color="#58a6ff", edgecolor="white", linewidth=0.3)
    b2 = ax.bar(x + w / 2, avg_p99, w, label="P99", color="#f85149", edgecolor="white", linewidth=0.3)
    for bars, vals in [(b1, avg_p50), (b2, avg_p99)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(max(avg_p50), max(avg_p99)) * 0.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(GPU_ORDER)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Latency Distribution (avg over 256/1024/2048 input lengths)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_latency_dist.png"), dpi=150, facecolor=DARK_BG)
    plt.close(fig)


def compute_efficiency():
    print("=" * 72)
    print("EFFICIENCY METRICS")
    print("=" * 72)
    for gpu in GPU_ORDER:
        bw = GPU_SPECS[gpu]["bw"]
        flops = GPU_SPECS[gpu]["flops"]
        avg_out = np.mean([RESULTS[gpu][l]["out_tok"] for l in INPUT_LENS])
        tok_per_tflop = avg_out / flops
        tok_per_gbs = avg_out / bw
        print(f"\n{gpu}:")
        print(f"  Avg Output Tok/s:       {avg_out:.2f}")
        print(f"  Tok/s per TFLOP:        {tok_per_tflop:.4f}")
        print(f"  Tok/s per GB/s BW:       {tok_per_gbs:.5f}")


def analyze_tradeoffs():
    print("\n" + "=" * 72)
    print("COMPUTE vs BANDWIDTH TRADEOFF ANALYSIS")
    print("=" * 72)

    bw_vals = np.array([GPU_SPECS[gpu]["bw"] for gpu in GPU_ORDER])
    flops_vals = np.array([GPU_SPECS[gpu]["flops"] for gpu in GPU_ORDER])

    avg_itl = np.array([np.mean([RESULTS[gpu][l]["itl"] for l in INPUT_LENS]) for gpu in GPU_ORDER])
    avg_ttft_2048 = np.array([RESULTS[gpu][2048]["ttft"] for gpu in GPU_ORDER])

    # ITL vs Bandwidth
    print("\n--- Does ITL scale proportionally with bandwidth? ---")
    ref_bw = bw_vals[-1]  # A10 is lowest
    ref_itl = avg_itl[-1]
    for i, gpu in enumerate(GPU_ORDER):
        bw_ratio = bw_vals[i] / ref_bw
        itl_ratio = ref_itl / avg_itl[i]
        print(f"  {gpu:12s}: BW={bw_vals[i]:4d} GB/s ({bw_ratio:.2f}x A10), "
              f"ITL={avg_itl[i]:.2f} ms ({itl_ratio:.2f}x faster than A10)")

    corr_bw_itl = np.corrcoef(bw_vals, avg_itl)[0, 1]
    print(f"  Correlation (BW vs ITL): r = {corr_bw_itl:.3f} — "
          f"{'Strong' if abs(corr_bw_itl) > 0.95 else 'Moderate'} "
          f"{'negative' if corr_bw_itl < 0 else 'positive'} correlation")

    # TTFT vs FLOPS
    print("\n--- Does TTFT scale with FLOPS? ---")
    ref_flops = flops_vals[-1]
    ref_ttft = avg_ttft_2048[-1]
    for i, gpu in enumerate(GPU_ORDER):
        flops_ratio = flops_vals[i] / ref_flops
        ttft_ratio = ref_ttft / avg_ttft_2048[i]
        print(f"  {gpu:12s}: FLOPS={flops_vals[i]:3d} TFLOPS ({flops_ratio:.1f}x A10), "
              f"TTFT={avg_ttft_2048[i]:.1f} ms ({ttft_ratio:.2f}x faster than A10)")

    print("\n  Key insight: L40S has 4.0× the FLOPS of 4090 but only 1.29× lower TTFT.")
    print("  Prefill is strongly bandwidth-constrained at batch=1 —")
    print("  both weights AND KV cache must be loaded from HBM for each prefill step.")

    # Where 4090 wins
    print("\n--- Where RTX 4090 wins (despite lower FLOPS) ---")
    print("  ITL / TPOT / Throughput: All bandwidth-bound metrics favor the 4090")
    print("  because it has the highest memory bandwidth (1008 GB/s).")
    print("  At batch=1, decoding is dominated by weight loading from HBM.")

    # Where L40S wins
    print("\n--- Where L40S wins (despite lower bandwidth than 4090) ---")
    print("  TTFT (Prefill): The L40S has 4× the BF16 FLOPS (362 vs 90),")
    print("  which helps accelerate the compute-heavy prefill phase.")
    print("  It still underperforms its FLOPS advantage due to the BW bottleneck.")


def recommendations():
    print("\n" + "=" * 72)
    print("GPU RECOMMENDATIONS")
    print("=" * 72)

    recs = [
        ("Low-latency single-user chatbot (batch=1)", "L40S",
         "Lowest TTFT across all input lengths (181ms vs 233ms for 4090 @ 2K input). "
         "First-token latency is the primary UX metric for interactive chatbots."),
        ("High-throughput API serving", "RTX 4090",
         "Highest output token throughput (53-56 tok/s) and lowest ITL (~18ms), "
         "meaning more tokens delivered per second to each user. "
         "Also benefits from highest bandwidth for decode-heavy multi-user serving."),
        ("Budget-constrained deployment", "RTX 4090",
         "Best efficiency metrics: ~0.61 tok/s/TFLOP (5× L40S) and ~0.055 tok/s/GB/s (10% above L40S). "
         "Consumer pricing (~$1,600 vs ~$10,000+ for L40S) yields dramatically better $/token."),
    ]
    for use_case, winner, rationale in recs:
        print(f"\n  {use_case}")
        print(f"  Winner: {winner}")
        print(f"  Why:    {rationale}")


def summary():
    print("\n" + "=" * 72)
    print("SUMMARY (LinkedIn / HF Community Post)")
    print("=" * 72)
    print("""
We benchmarked a Qwen3.5-9B model on three GPUs — A10, RTX 4090, and L40S — 
using vLLM at batch size 1 across 256, 1024, and 2048 input lengths with 256 output tokens.

The key finding: LLM serving performance splits cleanly along the compute vs bandwidth axis.
For decode operations (ITL, TPOT, throughput), memory bandwidth dominates — the RTX 4090 
leads with ~18ms ITL and ~55 tok/s thanks to its 1008 GB/s HBM, beating even the data-center 
L40S (864 GB/s, ~23ms ITL). For prefill (TTFT), compute matters more — the L40S delivers the 
fastest first token at 181ms vs 233ms for the 4090 on 2048-token inputs, leveraging its 4× 
BF16 FLOPS advantage.

The surprise winner is the RTX 4090: despite being a consumer card, it delivers the best 
token throughput per TFLOP (0.61 vs 0.12 for L40S) and the best throughput per GB/s of 
bandwidth. For most production serving scenarios where decode dominates total latency, 
the 4090 punches well above its weight class.

Efficiency analysis shows the 4090 converts both compute and bandwidth into output tokens 
more effectively than the A10 or L40S, making it the strongest choice for budget-constrained 
or high-throughput deployments. The L40S remains the pick for latency-sensitive applications 
where first-token response time is the critical metric.

Bottom line: for LLM inference at batch=1, bandwidth is the bottleneck for decode, 
compute is the bottleneck for prefill — and the 4090 exploits the bandwidth bottleneck 
better than any other card in this comparison.
    """.strip())


if __name__ == "__main__":
    plot_all()
    print("Plots saved to ./results/\n")
    compute_efficiency()
    analyze_tradeoffs()
    recommendations()
    summary()
