import pytest

from llmplatform.capacity import GIB, MODELS, kv_bytes_per_token, plan, replicas_needed


def test_kv_cache_per_token_llama_8b():
    # 2 (K,V) x 32 layers x 8 KV heads x 128 dims x 2 bytes = 128 KiB
    assert kv_bytes_per_token(MODELS["llama-3.1-8b"]) == 128 * 1024
    assert kv_bytes_per_token(MODELS["llama-3.1-8b"], "fp8") == 64 * 1024


def test_weights_and_tensor_parallel_split():
    p1 = plan("llama-3.1-8b", "H100-SXM", tensor_parallel=1)
    p2 = plan("llama-3.1-8b", "H100-SXM", tensor_parallel=2)
    assert p1.weights_gib_per_gpu == pytest.approx(8.03e9 * 2 / GIB, rel=1e-3)
    assert p2.weights_gib_per_gpu == pytest.approx(p1.weights_gib_per_gpu / 2, rel=1e-3)
    assert p2.max_concurrent_sequences > 2 * p1.max_concurrent_sequences  # freed memory


def test_70b_does_not_fit_on_one_80gb_gpu_in_bf16_but_fits_with_tp4():
    assert not plan("llama-3.1-70b", "H100-SXM", 1).fits
    assert plan("llama-3.1-70b", "H100-SXM", 4).fits


def test_quantisation_frees_kv_capacity():
    bf16 = plan("llama-3.1-8b", "L40S", weight_dtype="bf16")
    fp8 = plan("llama-3.1-8b", "L40S", weight_dtype="fp8", kv_dtype="fp8")
    assert fp8.max_concurrent_sequences > 2 * bf16.max_concurrent_sequences
    assert fp8.single_stream_tokens_s == pytest.approx(2 * bf16.single_stream_tokens_s)


def test_fp8_warning_on_gpu_without_fp8():
    assert "FP8" in plan("llama-3.1-8b", "A100-80GB", weight_dtype="fp8").note


def test_replicas_follow_littles_law():
    p = plan("llama-3.1-8b", "H100-SXM")
    low = replicas_needed(p, requests_per_s=0.5, output_tokens=300)
    high = replicas_needed(p, requests_per_s=20, output_tokens=300)
    assert low == 1 and high > low
    slow = plan("llama-3.1-70b", "L40S", 4)
    with pytest.raises(ValueError):
        replicas_needed(slow, 1, 300, target_tokens_s_per_user=20)
