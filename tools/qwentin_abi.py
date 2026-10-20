"""The C ABI of libforward_qwen.so as ctypes signatures, generated from the engine source by
tools/gen_qwentin_h.py -- do not edit. bind(lib) sets restype / argtypes of every qwn_* entry point the library
exports; pointer parameters are c_void_p (pass ctypes pointers, arrays, c_void_p(addr) or None)."""

import ctypes as C

SIGNATURES = {
    "qwn_quant_bfrag_check": (C.c_int, [C.c_int, C.c_int, C.c_void_p, C.c_void_p]),
    "qwn_quant_bfrag_bench": (C.c_int, [C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]),
    "qwn_free": (None, []),
    "qwn_tqf_blake2b256": (C.c_int, [C.c_void_p, C.c_size_t, C.c_void_p]),
    "qwn_gpu_pack_e2m3_test": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p]),
    "qwn_probe": (C.c_int, [C.c_char_p]),
    "qwn_init": (C.c_int, [C.c_char_p]),
    "qwn_hidden_size": (C.c_int, []),
    "qwn_intermediate_size": (C.c_int, []),
    "qwn_vocab_size": (C.c_int, []),
    "qwn_num_layers": (C.c_int, []),
    "qwn_num_attention_heads": (C.c_int, []),
    "qwn_num_key_value_heads": (C.c_int, []),
    "qwn_head_dim": (C.c_int, []),
    "qwn_max_seq": (C.c_int, []),
    "qwn_layer_type": (C.c_int, [C.c_int]),
    "qwn_device_bytes": (C.c_size_t, []),
    "qwn_non_quant_count": (C.c_int, []),
    "qwn_qmma_count": (C.c_int, []),
    "qwn_f32_count": (C.c_int, []),
    "qwn_reset_state": (C.c_int, []),
    "qwn_wide_gemm_check": (C.c_int, [C.c_int, C.c_int, C.c_void_p]),
    "qwn_wide_gemm_bench": (C.c_int, [C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]),
    "qwn_decode": (C.c_int, [C.c_int, C.c_int]),
    "qwn_deltanet_chunk_check": (C.c_int, [C.c_int, C.c_int, C.c_void_p]),
    "qwn_deltanet_chunk_check_ck": (C.c_int, [C.c_int, C.c_int, C.c_int, C.c_void_p]),
    "qwn_deltanet_chunk_check_hs": (C.c_int, [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p]),
    "qwn_deltanet_chunk_bench": (C.c_int, [C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]),
    "qwn_deltanet_chunk_bench_hs": (C.c_int, [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]),
    "qwn_wide_linear_test": (C.c_int, [C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]),
    "qwn_wide_linear_bench": (C.c_int, [C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]),
    "qwn_paged_attn_st": (C.c_int, []),
    "qwn_paged_t4_test_encode": (C.c_int, [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p]),
    "qwn_paged_init": (C.c_int, [C.c_int, C.c_int, C.c_int]),
    "qwn_paged_free": (C.c_int, []),
    "qwn_paged_reset_slot": (C.c_int, [C.c_int]),
    "qwn_paged_load_client": (C.c_int, [C.c_int, C.c_int]),
    "qwn_paged_slot_bytes": (C.c_size_t, [C.c_int]),
    "qwn_paged_token_bytes": (C.c_size_t, []),
    "qwn_paged_dstate": (C.c_int, []),
    "qwn_paged_state_bytes": (C.c_size_t, []),
    "qwn_park_ring_set": (C.c_int, [C.c_int]),
    "qwn_paged_slot_save": (C.c_int, [C.c_int, C.c_int, C.c_void_p, C.c_size_t]),
    "qwn_paged_slot_restore": (C.c_int, [C.c_int, C.c_int, C.c_void_p, C.c_size_t]),
    "qwn_graph_stats": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]),
    "qwn_paged_decode_step": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]),
    "qwn_paged_stats": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]),
    "qwn_paged_logits_sink": (C.c_int, [C.c_void_p, C.c_int, C.c_int]),
    "qwn_gdn_prefill_ab": (C.c_int, [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]),
    "qwn_paged_prefill_batch": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p]),
    "qwn_paged_verify": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p]),
    "qwn_paged_commit": (C.c_int, [C.c_void_p, C.c_void_p, C.c_int]),
    "qwn_paged_verify_hidden": (C.c_void_p, [C.c_void_p]),
    "qwn_paged_mtp_init": (C.c_int, [C.c_int]),
    "qwn_paged_mtp_draft": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p]),
}


def bind(lib, names=None):
    for name in names or SIGNATURES:
        fn = getattr(lib, name, None)
        if fn is not None:
            fn.restype, fn.argtypes = SIGNATURES[name]
    return lib
