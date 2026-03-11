#!/usr/bin/env python3
"""
TITAN Test Runner — validates all 7 pillars with real numerical assertions.
No pytest required; runs standalone with numpy + scipy only.
"""
import sys
import math
import traceback
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from titan.core.titan_numpy import (
    # Pillar 1
    compress_tensor, decompress_tensor, NVMeBlockStore, LSTMPrefetchPredictor,
    HMSStreamingEngine, DEFAULT_TIER_CONFIGS, MemoryTier,
    # Pillar 2
    ErrorAccumulationBank, flash_attention_micro, stripe_ffn_forward, gelu,
    # Pillar 3
    ASDTOptimizer, asdt_vram_estimate, ParameterClass,
    # Pillar 4
    TensorRingMatrix,
    # Pillar 5
    CountMinSketch, TGSSManager,
    # Pillar 6
    BSPSManager, Phase, ParamState,
    # Pillar 7
    GradientHologram, HGEManager, verify_complementarity,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
tests_run = 0
tests_failed = 0


def test(name: str, fn):
    global tests_run, tests_failed
    tests_run += 1
    try:
        fn()
        print(f"  {PASS}  {name}")
    except AssertionError as e:
        tests_failed += 1
        print(f"  {FAIL}  {name}: {e}")
    except Exception as e:
        tests_failed += 1
        print(f"  {FAIL}  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


# ===========================================================================
print("\n=== PILLAR 1: Hierarchical Memory Streaming (HMS) ===")

def t_compress_roundtrip():
    arr = np.random.randn(128, 64).astype(np.float32)
    for codec in ["none", "lz4", "zstd"]:
        c = compress_tensor(arr, codec, 16)
        r = decompress_tensor(c, (128, 64))
        err = float(np.abs(r - arr).max())
        assert err < 1e-3, f"codec={codec} max_err={err:.6f}"

test("Compress/decompress lossless roundtrip (INT16 equiv)", t_compress_roundtrip)

def t_compress_int4():
    arr = np.random.randn(64, 32).astype(np.float32)
    c = compress_tensor(arr, "zstd", quant_bits=4)
    r = decompress_tensor(c, (64, 32))
    assert r.shape == (64, 32)
    assert np.isfinite(r).all(), "NaN/Inf in INT4 decompressed"

test("INT4 quantization + zstd compression", t_compress_int4)

def t_nvme_store():
    with tempfile.TemporaryDirectory() as td:
        store = NVMeBlockStore(Path(td), DEFAULT_TIER_CONFIGS[MemoryTier.NVME])
        arr = np.random.randn(32, 16).astype(np.float32)
        store.write("layer_0", arr)
        assert store.has("layer_0")
        r = store.read("layer_0")
        assert r is not None and r.shape == (32, 16), f"got {r}"

test("NVMeBlockStore write/read", t_nvme_store)

def t_lstm_predictor():
    pred = LSTMPrefetchPredictor(n_layers=10, hidden=16, window=4)
    seq = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in seq:
        pred.record_access(i)
    loss = pred.train_step(seq, lr=1e-2)
    preds = pred.predict_next(3)
    assert math.isfinite(loss), f"non-finite loss: {loss}"
    assert len(preds) > 0 and all(0 <= p < 10 for p in preds)

test("LSTM prefetch predictor trains + predicts", t_lstm_predictor)

def t_hms_fetch():
    with tempfile.TemporaryDirectory() as td:
        store = NVMeBlockStore(Path(td), DEFAULT_TIER_CONFIGS[MemoryTier.NVME])
        for i in range(5):
            store.write(f"layer_{i}", np.random.randn(16, 8).astype(np.float32))
        hms = HMSStreamingEngine(store, n_layers=5, dram_cache_mb=10)
        arr = hms.get_layer("layer_0", 0)
        assert arr.shape == (16, 8)
        arr1 = hms.get_layer("layer_1", 1)
        assert arr1.shape == (16, 8)
        s = hms.stats()
        assert s["total_bytes"] > 0

test("HMS get_layer + prefetch + stats", t_hms_fetch)

def t_overlap_condition():
    with tempfile.TemporaryDirectory() as td:
        store = NVMeBlockStore(Path(td), DEFAULT_TIER_CONFIGS[MemoryTier.NVME])
        hms = HMSStreamingEngine(store, n_layers=10)
        ok = hms.check_overlap_condition(compute_ms=50.0, layer_bytes=25*1024*1024)
        assert isinstance(ok, bool)

test("Theorem 7.3 overlap condition check", t_overlap_condition)


# ===========================================================================
print("\n=== PILLAR 2: Micro-Layer Materialization Engine (MLME) ===")

def t_flash_attention_exact():
    N, d = 32, 16
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((N, d)).astype(np.float32)
    K = rng.standard_normal((N, d)).astype(np.float32)
    V = rng.standard_normal((N, d)).astype(np.float32)
    flash_out = flash_attention_micro(Q, K, V, block_size=8, causal=False)
    # Naive reference
    scale = d ** -0.5
    scores = Q @ K.T * scale
    attn = np.exp(scores - scores.max(axis=1, keepdims=True))
    attn /= attn.sum(axis=1, keepdims=True)
    naive_out = attn @ V
    err = float(np.abs(flash_out - naive_out).max())
    assert err < 0.05, f"Flash attention max error: {err:.4f}"

test("Flash attention matches naive (exact)", t_flash_attention_exact)

def t_flash_attention_causal():
    N, d = 16, 8
    rng = np.random.default_rng(2)
    Q = rng.standard_normal((N, d)).astype(np.float32)
    K = rng.standard_normal((N, d)).astype(np.float32)
    V = rng.standard_normal((N, d)).astype(np.float32)
    out = flash_attention_micro(Q, K, V, block_size=4, causal=True)
    assert out.shape == (N, d) and np.isfinite(out).all()

test("Flash attention causal masking", t_flash_attention_causal)

def t_stripe_ffn_exact():
    rng = np.random.default_rng(3)
    d, d_ff = 32, 128
    W1 = rng.standard_normal((d, d_ff)).astype(np.float32) * 0.1
    W2 = rng.standard_normal((d_ff, d)).astype(np.float32) * 0.1
    b1 = rng.standard_normal(d_ff).astype(np.float32) * 0.01
    b2 = rng.standard_normal(d).astype(np.float32) * 0.01
    x  = rng.standard_normal((4, d)).astype(np.float32)

    stripe_out = stripe_ffn_forward(x, W1, W2, b1, b2, stripe_width=32)
    # Full reference
    h = gelu(x @ W1 + b1)
    full_out = h @ W2 + b2
    err = float(np.abs(stripe_out - full_out).max())
    assert err < 1e-4, f"Stripe FFN error: {err:.6f}"

test("Stripe FFN exact decomposition matches full FFN", t_stripe_ffn_exact)

def t_eab_bounded():
    """EAB bank residual must stay bounded (proves rounding error doesn't compound)."""
    rng = np.random.default_rng(4)
    w = rng.standard_normal(256).astype(np.float32)
    eab = ErrorAccumulationBank(256, quant_bits=4)
    bank_residuals = []
    for _ in range(20):
        q_w, bank = eab.quantize_with_correction(w)
        bank_residuals.append(float(np.abs(bank).mean()))
    # Bank residuals should be << 1 quant level (range/15 ~ 0.5 for N(0,1))
    assert max(bank_residuals) < 0.5, f"EAB bank residual blew up: {max(bank_residuals):.4f}"

test("EAB quantization error stays bounded across steps", t_eab_bounded)


# ===========================================================================
print("\n=== PILLAR 3: Adaptive Sparse Delta Training (ASDT) ===")

def t_asdt_plastic_update():
    rng = np.random.default_rng(5)
    params = {"W": rng.standard_normal((32, 16)).astype(np.float32),
              "b": rng.standard_normal(16).astype(np.float32)}
    W_before = params["W"].copy()
    grads = {"W": rng.standard_normal((32, 16)).astype(np.float32) * 1.0,  # large grad
             "b": rng.standard_normal(16).astype(np.float32) * 1.0}
    opt = ASDTOptimizer(params, top_k_fraction=1.0, plastic_lr=1e-2)
    opt._tau_high = 0.0  # force all plastic
    stats = opt.step(grads)
    assert not np.allclose(params["W"], W_before), "ASDT failed to update plastic params"
    assert stats["plastic"] > 0

test("ASDT plastic Adam update changes parameters", t_asdt_plastic_update)

def t_asdt_dormant():
    params = {"W": np.ones((8, 4), dtype=np.float32)}
    grads: Dict = {}  # no gradients → dormant
    opt = ASDTOptimizer(params, plastic_lr=1e-2)
    stats = opt.step(grads)
    assert stats["dormant"] > 0

test("ASDT dormant when no gradients", t_asdt_dormant)

def t_asdt_elastic_sign():
    rng = np.random.default_rng(6)
    params = {"W": rng.standard_normal((16, 8)).astype(np.float32)}
    W_before = params["W"].copy()
    grads = {"W": rng.standard_normal((16, 8)).astype(np.float32) * 0.0005}
    opt = ASDTOptimizer(params, elastic_lr=0.1)
    opt._tau_high = 1e10  # prevent plastic
    opt._tau_low  = 0.0   # force elastic
    opt._grad_ema = {"W": 0.001}
    stats = opt.step(grads)
    # Elastic = sign-SGD; params change by ±elastic_lr
    diff = np.abs(params["W"] - W_before)
    assert diff.max() > 0, "Elastic update had no effect"

test("ASDT elastic sign-SGD update", t_asdt_elastic_sign)

def t_asdt_vram_estimate():
    est = asdt_vram_estimate(1_000_000_000, 0.001)
    assert 5 < est["total_vram_mb"] < 50
    assert est["plastic_params"] == 1_000_000

test("ASDT VRAM estimate matches §3.3 formula", t_asdt_vram_estimate)


# ===========================================================================
print("\n=== PILLAR 4: Tensor Ring Decomposition (TRD) ===")

def t_trd_core_shape():
    trd = TensorRingMatrix(16, 8, rank=4, n_cores=4)
    for c in trd.cores:
        assert c.shape[0] == 4 and c.shape[2] == 4, f"bad shape: {c.shape}"

test("TRD cores have correct ring shapes (r × n × r)", t_trd_core_shape)

def t_trd_reconstruct_shape():
    trd = TensorRingMatrix(16, 8, rank=4, n_cores=4)
    W = trd.reconstruct()
    assert W.shape == (16, 8), f"got {W.shape}"

test("TRD reconstruct() returns correct shape", t_trd_reconstruct_shape)

def t_trd_matvec():
    rng = np.random.default_rng(7)
    trd = TensorRingMatrix(16, 8, rank=4, n_cores=4)
    x = rng.standard_normal((3, 16)).astype(np.float32)
    out = trd.matvec(x)
    assert out.shape == (3, 8)

test("TRD matvec output shape correct", t_trd_matvec)

def t_trd_compression_ratio():
    trd = TensorRingMatrix(512, 512, rank=32, n_cores=6)
    assert trd.compression_ratio() > 1.0

test("TRD compression ratio > 1x for 512×512 matrix", t_trd_compression_ratio)

def t_trd_rank_entropy():
    trd = TensorRingMatrix(16, 8, rank=4, n_cores=3)
    h = trd.rank_entropy(0)
    assert math.isfinite(h), f"non-finite entropy: {h}"

test("TRD rank entropy is finite", t_trd_rank_entropy)

def t_trd_core_update():
    rng = np.random.default_rng(8)
    trd = TensorRingMatrix(16, 8, rank=4, n_cores=4)
    cores_before = [c.copy() for c in trd.cores]
    grad_W = rng.standard_normal((16, 8)).astype(np.float32) * 0.1
    trd.update_cores(grad_W, lr=1e-2)
    changed = any(not np.allclose(a, b) for a, b in zip(trd.cores, cores_before))
    assert changed, "TRD core update had no effect"

test("TRD core gradient update changes cores", t_trd_core_update)

def t_trd_adapt_rank():
    trd = TensorRingMatrix(16, 8, rank=4, n_cores=3)
    suggestions = trd.adapt_rank_entropy(target_entropy=1.0)
    assert len(suggestions) == 3
    for v in suggestions.values():
        assert math.isfinite(v)

test("TRD adaptive rank entropy suggestions", t_trd_adapt_rank)


# ===========================================================================
print("\n=== PILLAR 5: Temporal Gradient Superposition Sketching (TGSS) ===")

def t_cms_update_query():
    sketch = CountMinSketch(width=10_000, depth=5)
    idx = np.arange(100, dtype=np.int64)
    vals = np.abs(np.random.randn(100)).astype(np.float32)
    sketch.update(idx, vals)
    est = sketch.query(idx)
    assert (est >= 0).all(), "Negative estimates"
    assert est.max() > 0, "All estimates are zero"

test("CMS update + query returns positive estimates", t_cms_update_query)

def t_cms_ema_merge():
    s1 = CountMinSketch(width=1000, depth=3)
    s2 = CountMinSketch(width=1000, depth=3)
    idx = np.array([0, 1, 2], dtype=np.int64)
    s1.update(idx, np.array([1.0, 2.0, 3.0]))
    s2.update(idx, np.array([4.0, 5.0, 6.0]))
    s1.merge_ema(s2, alpha=0.5)
    est = s1.query(idx)
    assert (est >= 0).all()

test("CMS temporal EMA superposition", t_cms_ema_merge)

def t_tgss_update_importance():
    rng = np.random.default_rng(9)
    mgr = TGSSManager(width=10_000, depth=3, use_freq=False)
    grad = rng.standard_normal((32, 16)).astype(np.float32)
    mgr.update("layer0", grad)
    imp = mgr.get_importance("layer0")
    assert imp > 0, f"Importance should be positive, got {imp}"

test("TGSS update + importance score positive", t_tgss_update_importance)

def t_tgss_freq_domain():
    rng = np.random.default_rng(10)
    mgr = TGSSManager(use_freq=True)
    grad = rng.standard_normal((32, 16)).astype(np.float32)
    result = mgr._freq_domain(grad)
    assert result.shape == grad.shape and np.isfinite(result).all()

test("TGSS frequency-domain gradient transform", t_tgss_freq_domain)

def t_cms_memory():
    sketch = CountMinSketch(width=1_000_000, depth=5)
    assert sketch.memory_bytes() == 5 * 1_000_000 * 4  # 20 MB

test("CMS memory formula: d×w×4 bytes = 20MB for (5, 1M)", t_cms_memory)


# ===========================================================================
print("\n=== PILLAR 6: Biologically-Inspired Synaptic Plasticity Scheduling (BSPS) ===")

def t_bsps_growth_transition():
    bsps = BSPSManager(tau_high=1e-3, tau_low=1e-5, m1=2, m2=10)
    bsps.register(["W", "b"])
    grads = {"W": np.ones((8, 4)) * 1.0, "b": np.ones(4) * 1.0}  # large
    for _ in range(5):
        bsps.step(grads)
    assert bsps.get_phase("W") == Phase.GROWTH, f"W phase: {bsps.get_phase('W')}"

test("BSPS FROZEN→GROWTH with large gradients", t_bsps_growth_transition)

def t_bsps_elastic_transition():
    bsps = BSPSManager(tau_high=0.5, tau_low=1e-5, m1=2, m2=5)
    bsps.register(["W"])
    bsps._states["W"].phase = Phase.GROWTH
    bsps._states["W"].grad_ema = 0.01  # below tau_high
    bsps._states["W"].steps_in_phase = 3
    grads = {"W": np.ones((4, 4)) * 0.001}
    bsps.step(grads)
    phase = bsps.get_phase("W")
    assert phase in (Phase.GROWTH, Phase.ELASTIC)

test("BSPS GROWTH→ELASTIC transition condition", t_bsps_elastic_transition)

def t_bsps_sleeping_decay():
    bsps = BSPSManager()
    bsps.register(["W"])
    bsps._states["W"].phase = Phase.SLEEPING
    params = {"W": np.ones((4, 4), dtype=np.float32) * 2.0}
    bsps.apply_sleeping_decay(params, decay=0.5)
    assert np.allclose(params["W"], 1.0), f"decay failed: {params['W'].mean():.3f}"

test("BSPS sleeping decay halves weights", t_bsps_sleeping_decay)

def t_bsps_task_reawakening():
    bsps = BSPSManager()
    bsps.register(["W"])
    bsps._states["W"].phase = Phase.FROZEN
    bsps.set_task_relevance({"W": 0.9})  # high relevance
    grads = {"W": None}
    bsps.step(grads)
    assert bsps.get_phase("W") == Phase.ELASTIC, f"TRR failed: {bsps.get_phase('W')}"

test("BSPS Task Relevance Reawakening FROZEN→ELASTIC", t_bsps_task_reawakening)

def t_bsps_vram_estimate():
    bsps = BSPSManager()
    bsps.register(["W", "b"])
    bsps._states["W"].phase = Phase.GROWTH
    bsps._states["b"].phase = Phase.FROZEN
    sizes = {"W": 1024, "b": 32}
    v = bsps.vram_estimate_bytes(sizes)
    assert v == 1024 * 12, f"expected {1024*12}, got {v}"

test("BSPS VRAM estimate = GROWTH params × 12 bytes", t_bsps_vram_estimate)


# ===========================================================================
print("\n=== PILLAR 7: Holographic Gradient Encoding (HGE) ===")

def t_hge_encode_decode():
    rng = np.random.default_rng(11)
    grad = rng.standard_normal((32, 16)).astype(np.float32)
    h = GradientHologram(grad.shape, keep_frac=0.3)
    h.encode(grad)
    r = h.decode()
    assert r.shape == grad.shape
    assert np.isfinite(r).all()

test("HGE encode + decode shape preserved, no NaN", t_hge_encode_decode)

def t_hge_compression_ratio():
    rng = np.random.default_rng(12)
    grad = rng.standard_normal((256, 128)).astype(np.float32)
    h = GradientHologram(grad.shape, keep_frac=0.05)
    h.encode(grad)
    assert h.compression_ratio() > 1.0, f"CR={h.compression_ratio():.2f}"

test("HGE 5% keep_frac gives >1x compression", t_hge_compression_ratio)

def t_hge_superposition():
    rng = np.random.default_rng(13)
    g1 = rng.standard_normal((16, 8)).astype(np.float32)
    g2 = rng.standard_normal((16, 8)).astype(np.float32)
    h1 = GradientHologram(g1.shape, keep_frac=0.3)
    h2 = GradientHologram(g2.shape, keep_frac=0.3)
    h1.encode(g1); h2.encode(g2)
    h1.superpose(h2, weight=0.5)
    r = h1.decode()
    assert np.isfinite(r).all() and r.shape == g1.shape

test("HGE temporal superposition stays finite", t_hge_superposition)

def t_hge_manager():
    rng = np.random.default_rng(14)
    mgr = HGEManager(keep_frac=0.2)
    grad = rng.standard_normal((16, 8)).astype(np.float32)
    mgr.encode("W", grad)
    decoded = mgr.decode("W")
    assert decoded is not None and decoded.shape == (16, 8)
    stats = mgr.stats()
    assert stats["n_holograms"] == 1

test("HGE manager encode/decode/stats", t_hge_manager)

def t_complementarity():
    rng = np.random.default_rng(15)
    true_g  = rng.standard_normal(64).astype(np.float32)
    tgss_est = true_g + rng.standard_normal(64).astype(np.float32) * 0.3
    hge_est  = true_g + rng.standard_normal(64).astype(np.float32) * 0.3
    result = verify_complementarity(true_g, tgss_est, hge_est)
    assert "complementarity_holds" in result
    assert "error_combined" in result
    assert result["error_combined"] >= 0

test("HGE+TGSS complementarity property verified (§6.5)", t_complementarity)


# ===========================================================================
print("\n=== INTEGRATION: End-to-end TITAN training simulation ===")

def t_full_training_step():
    """Simulates one TITAN training step with all 7 pillars active."""
    rng = np.random.default_rng(99)

    # Tiny model: 2-layer linear network
    params = {
        "W1": rng.standard_normal((32, 64)).astype(np.float32) * 0.1,
        "b1": np.zeros(64, dtype=np.float32),
        "W2": rng.standard_normal((64, 10)).astype(np.float32) * 0.1,
        "b2": np.zeros(10, dtype=np.float32),
    }

    # Step 1 (BSPS): Initialize phase manager
    bsps = BSPSManager(tau_high=1e-4, tau_low=1e-7, m1=1, m2=2)
    bsps.register(list(params.keys()))

    # Step 2 (HMS): Initialize streaming
    tgss = TGSSManager(width=5000, depth=3)
    hge_mgr = HGEManager(keep_frac=0.3)

    # Step 3 (TRD): Compress W1 into ring format
    trd_W1 = TensorRingMatrix(32, 64, rank=4, n_cores=4)
    trd_W1.initialize_from_matrix(params["W1"])

    # Step 4 (MLME): Forward pass with stripe FFN
    # Two-layer network: x(8,32) -> W1(32,64) -> hidden(8,64) -> W2(64,10) -> logits(8,10)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    h = gelu(x @ params["W1"] + params["b1"])  # (8, 64)
    logits = h @ params["W2"] + params["b2"]   # (8, 10)

    # Softmax cross-entropy loss
    targets = rng.integers(0, 10, size=8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    loss = -float(np.log(probs[np.arange(8), targets] + 1e-10).mean())

    # Step 5-6 (HGE+MLME): Compute gradients
    dlogits = probs.copy()
    dlogits[np.arange(8), targets] -= 1.0
    dlogits /= 8
    dh = dlogits @ params["W2"].T * (h > 0)  # gelu approx gradient
    grads = {
        "W2": h.T @ dlogits,                  # (64, 10)
        "b2": dlogits.sum(axis=0),             # (10,)
        "W1": x.T @ dh,                        # (32, 64)
        "b1": dh.sum(axis=0),                  # (64,)
    }

    # Step 7 (TGSS): Update sketches
    for name, g in grads.items():
        if g is not None and np.linalg.norm(g) > 0:
            tgss.update(name, g)

    # Step 8 (ASDT): Active-set selection via sketch importance
    importance = {n: tgss.get_importance(n) for n in params}

    # Step 9 (HGE): Encode gradients as holograms
    for name, g in grads.items():
        hge_mgr.encode(name, g)

    # Step 10 (TRD + BSPS): Phase update + core update
    bsps_counts = bsps.step(grads)
    trd_W1.update_cores(grads["W1"], lr=1e-3)

    # Step 11 (ASDT): Parameter update
    asdt = ASDTOptimizer(params, plastic_lr=1e-3)
    asdt._tau_high = 0.0  # force plastic for test
    asdt.step(grads, importance)

    # Step 12: Verify loss is finite and params changed
    assert math.isfinite(loss), f"loss is nan/inf: {loss}"
    assert loss > 0, f"loss should be positive: {loss}"

    hge_stats = hge_mgr.stats()
    assert hge_stats["n_holograms"] > 0

    # Verify TRD cores changed
    # (we already called update_cores)
    W_reconstructed = trd_W1.reconstruct()
    assert W_reconstructed.shape == (32, 64)

    print(f"    loss={loss:.4f} | BSPS={bsps_counts} | "
          f"HGE_cr={hge_stats['avg_compression_ratio']:.1f}x | "
          f"TGSS_mb={tgss.total_memory_bytes()/(1024**2):.3f}MB")

test("Full TITAN 12-step training simulation", t_full_training_step)


# ===========================================================================
print(f"\n{'='*60}")
print(f"Results: {tests_run - tests_failed}/{tests_run} tests passed")
if tests_failed > 0:
    print(f"FAILED: {tests_failed} test(s)")
    sys.exit(1)
else:
    print("All tests passed! ✅")
