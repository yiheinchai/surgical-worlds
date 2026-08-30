# Native resolution ladder (CRCD)

What training data looks like at each **native model output** resolution.
Costs are estimates for RTX 3090, scaling from measured 128² training run.

| Native res | Patches | Compute vs 128² | Infer/step | Full train | VRAM | Fits 24GB |
|------------|---------|-----------------|------------|------------|------|-----------|
| 64² | 256 | 0.2× | ~230 ms | ~5 h | ~0.2 GB | ✓ |
| 128² | 1,024 | 1.0× | ~319 ms | ~7 h | ~0.6 GB | ✓ |
| 192² | 2,304 | 2.2× | ~1113 ms | ~24 h | ~1.4 GB | ✓ |
| 256² | 4,096 | 4.0× | ~2995 ms | ~66 h | ~2.4 GB | ✓ |
| 384² | 9,216 | 9.0× | OOM | OOM | ~5.5 GB | ✗ |
| 512² | 16,384 | 16.0× | OOM | OOM | ~9.8 GB | ✗ |

**Recommendation:**
- **128²** — current model; cheapest (~7h train, ~320ms/step)
- **256²** — best quality/cost tradeoff (~9× infer vs 128², ~65h train); fits 24GB
- **384²+** — dynamics forward OOM on RTX 3090 (24GB); needs A100 40GB+

See `native_resolution_ladder.png` (true pixel size) and `native_resolution_same_display.png` (all blown up to 512px for fair comparison).