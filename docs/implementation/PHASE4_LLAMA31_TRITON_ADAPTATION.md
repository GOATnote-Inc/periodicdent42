# PHASE 4: LLaMA 3.1 8B Production Validation (Triton)

**Priority**: 🎯 VALIDATION (Proves production readiness)  
**Effort**: 20-25 hours  
**Implementation**: **Python Integration** (uses Triton kernels from Phases 1-3)  
**Complexity**: Medium (integration + benchmarking)

---

## 🎯 **OBJECTIVE**

Validate FlashCore Triton kernels against real-world LLM inference:

1. **Integration**: Replace HuggingFace attention with FlashCore
2. **Correctness**: Generate identical text to reference
3. **Performance**: Competitive with PyTorch SDPA
4. **Deployment**: Document production usage

**Success Criterion**: Generate 100 coherent tokens from LLaMA 3.1 8B with FlashCore, matching HuggingFace reference output.

---

## 📐 **LLAMA 3.1 8B CONFIGURATION**

### **Model Architecture**
```python
config = {
    'model_name': 'meta-llama/Llama-3.1-8B',
    'vocab_size': 128256,
    'hidden_size': 4096,
    'intermediate_size': 14336,
    'num_hidden_layers': 32,
    'num_attention_heads': 32,      # H_q
    'num_key_value_heads': 8,       # H_kv (GQA with 4:1 ratio)
    'head_dim': 128,                # D
    'max_position_embeddings': 131072,  # 128K context
    'rms_norm_eps': 1e-5,
    'rope_theta': 500000.0,
}
```

### **FlashCore Requirements Met**
- ✅ GQA (32:8 ratio) - Phase 2
- ✅ KV Cache - Phase 1  
- ✅ Causal Masking - Phase 3
- ✅ RoPE - Handled in QK projections (not in attention kernel)

**Attention Layer Only Needs**: GQA + KV Cache + Causal (All Triton!)

---

## 🔧 **INTEGRATION STRATEGY**

### **Approach: Monkey-Patch HuggingFace**

Replace `LlamaAttention.forward()` with FlashCore Triton implementation:

```python
# integration/llama_flashcore.py

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from flashcore.fast.attention_production import attention_with_kv_cache  # Our Triton kernel

class LlamaFlashCoreAttention(LlamaAttention):
    """Drop-in replacement for LlamaAttention using FlashCore Triton kernels."""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # QKV projections (same as original)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE (same as original - done BEFORE attention)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Convert cache format if needed
        if past_key_value is not None:
            # HuggingFace DynamicCache format -> our tuple format
            if hasattr(past_key_value, 'key_cache'):
                # New format (transformers 4.36+)
                past_kv_tuple = (
                    past_key_value.key_cache[self.layer_idx],
                    past_key_value.value_cache[self.layer_idx]
                )
            else:
                # Old format (tuple)
                past_kv_tuple = past_key_value
        else:
            past_kv_tuple = None
        
        # FLASHCORE TRITON ATTENTION (replaces original)
        attn_output, updated_cache = attention_with_kv_cache(
            query=query_states,                    # [B, H_q=32, S, D=128]
            key=key_states,                        # [B, H_kv=8, S, D=128]
            value=value_states,                    # [B, H_kv=8, S, D=128]
            past_key_value=past_kv_tuple,
            is_causal=True,                        # LLaMA is autoregressive
            update_cache=use_cache,
            num_query_heads=self.num_heads,        # 32
            num_kv_heads=self.num_key_value_heads  # 8
        )
        
        # Reshape output (same as original)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # Output projection (same as original)
        attn_output = self.o_proj(attn_output)
        
        # Handle cache format for return
        if use_cache:
            # Convert back to HuggingFace format if needed
            if hasattr(past_key_value, 'update'):
                # New DynamicCache format
                past_key_value.update(updated_cache[0], updated_cache[1], self.layer_idx)
                cache_to_return = past_key_value
            else:
                # Old tuple format
                cache_to_return = updated_cache
        else:
            cache_to_return = None
        
        return attn_output, None, cache_to_return


def replace_llama_attention_with_flashcore(model):
    """
    Replace all LlamaAttention modules with FlashCore Triton version.
    
    Usage:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
        replace_llama_attention_with_flashcore(model)
        # Now model uses FlashCore Triton kernels for attention
    """
    import gc
    
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            # Get parent module and attribute name
            *parent_path, attr_name = name.split('.')
            parent_name = '.'.join(parent_path)
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # Create FlashCore version (preserves all weights)
            flashcore_attn = LlamaFlashCoreAttention(module.config, module.layer_idx)
            flashcore_attn.load_state_dict(module.state_dict())
            
            # Replace
            setattr(parent, attr_name, flashcore_attn)
            replaced_count += 1
    
    print(f"✅ Replaced {replaced_count} attention layers with FlashCore Triton kernels")
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache()
    
    return model
```

---

## 🐍 **TEST SCRIPTS**

### **Test 1: Correctness Validation**

```python
# tests/test_llama31_correctness.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from integration.llama_flashcore import replace_llama_attention_with_flashcore

def test_llama31_generation_correctness():
    """
    Test that FlashCore generates identical output to reference.
    """
    model_id = "meta-llama/Llama-3.1-8B"
    device = "cuda"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Reference model (PyTorch SDPA)
    print("Loading reference model (PyTorch SDPA)...")
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="sdpa"
    )
    model_ref.eval()
    
    # FlashCore model (Triton)
    print("Loading FlashCore model (Triton kernels)...")
    model_fc = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager"  # Will replace with FlashCore
    )
    replace_llama_attention_with_flashcore(model_fc)
    model_fc.eval()
    
    # Test prompts
    prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In a galaxy far, far away",
        "def fibonacci(n):\n    ",
    ]
    
    for prompt in prompts:
        print(f"\nTesting prompt: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate with reference
        print("  Generating with PyTorch SDPA...")
        with torch.no_grad():
            outputs_ref = model_ref.generate(
                **inputs,
                max_new_tokens=50,  # Shorter for testing
                do_sample=False,    # Greedy for reproducibility
                use_cache=True
            )
        text_ref = tokenizer.decode(outputs_ref[0], skip_special_tokens=True)
        
        # Generate with FlashCore
        print("  Generating with FlashCore Triton...")
        with torch.no_grad():
            outputs_fc = model_fc.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                use_cache=True
            )
        text_fc = tokenizer.decode(outputs_fc[0], skip_special_tokens=True)
        
        # Compare
        print(f"  Reference:  {text_ref}")
        print(f"  FlashCore:  {text_fc}")
        
        # Should be identical (deterministic greedy decoding)
        if text_ref == text_fc:
            print("  ✅ PASS - Outputs match exactly")
        else:
            print("  ❌ FAIL - Outputs differ")
            print(f"  Length: ref={len(text_ref)}, fc={len(text_fc)}")
            # Find first difference
            for i, (c_ref, c_fc) in enumerate(zip(text_ref, text_fc)):
                if c_ref != c_fc:
                    print(f"  First diff at char {i}: ref='{c_ref}' fc='{c_fc}'")
                    break
            raise AssertionError("Outputs differ!")
    
    print("\n✅ All correctness tests passed!")

if __name__ == "__main__":
    test_llama31_generation_correctness()
```

### **Test 2: Performance Benchmark**

```python
# benchmarks/benchmark_llama31.py

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from integration.llama_flashcore import replace_llama_attention_with_flashcore

def benchmark_decode(model, tokenizer, cache_lengths=[512, 1024, 2048, 4096]):
    """Benchmark decode phase (single token generation)."""
    device = model.device
    results = []
    
    for cache_len in cache_lengths:
        print(f"\nBenchmarking decode with cache_len={cache_len}...")
        
        # Create initial cache with prefill
        prompt = "test " * (cache_len // 2)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cache_len).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(
                    input_ids=inputs['input_ids'][:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
        
        # Benchmark decode steps
        decode_times = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'][:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - start)
            
            past_key_values = outputs.past_key_values
        
        avg_time_ms = (sum(decode_times) / len(decode_times)) * 1000
        tokens_per_sec = 1000 / avg_time_ms
        
        results.append({
            'cache_len': cache_len,
            'time_ms': avg_time_ms,
            'tokens_per_sec': tokens_per_sec
        })
        
        print(f"  Decode: {avg_time_ms:.2f}ms ({tokens_per_sec:.1f} tok/s)")
    
    return results


def main():
    model_id = "meta-llama/Llama-3.1-8B"
    device = "cuda"
    
    print("=" * 80)
    print("LLAMA 3.1 8B BENCHMARK: FlashCore Triton vs PyTorch SDPA")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Benchmark PyTorch SDPA (baseline)
    print("\n" + "=" * 80)
    print("BASELINE: PyTorch SDPA")
    print("=" * 80)
    model_sdpa = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="sdpa"
    )
    model_sdpa.eval()
    
    decode_sdpa = benchmark_decode(model_sdpa, tokenizer)
    
    # Clean up
    del model_sdpa
    torch.cuda.empty_cache()
    
    # Benchmark FlashCore Triton
    print("\n" + "=" * 80)
    print("FLASHCORE: Triton Kernels")
    print("=" * 80)
    model_fc = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device
    )
    replace_llama_attention_with_flashcore(model_fc)
    model_fc.eval()
    
    decode_fc = benchmark_decode(model_fc, tokenizer)
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: FlashCore Triton vs PyTorch SDPA")
    print("=" * 80)
    print("\nDecode Speedup:")
    for ref, fc in zip(decode_sdpa, decode_fc):
        speedup = ref['time_ms'] / fc['time_ms']
        symbol = '✅' if speedup >= 1.0 else '⚠️'
        print(f"Cache={ref['cache_len']:4d}: {speedup:.2f}× {symbol} ({fc['time_ms']:.2f}ms vs {ref['time_ms']:.2f}ms)")
    
    # Check if target met (<10ms for cache=2048)
    target_config = next(r for r in decode_fc if r['cache_len'] == 2048)
    if target_config['time_ms'] < 10.0:
        print(f"\n✅ TARGET MET: Decode @2048 = {target_config['time_ms']:.2f}ms < 10ms")
    else:
        print(f"\n⚠️ TARGET MISSED: Decode @2048 = {target_config['time_ms']:.2f}ms > 10ms")

if __name__ == "__main__":
    main()
```

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Step 1: Integration Module (8-10 hours)**
- [ ] Create `integration/llama_flashcore.py`
- [ ] Implement `LlamaFlashCoreAttention` class
- [ ] Handle RoPE application (before attention)
- [ ] Handle cache format conversion (HF ↔ FlashCore)
- [ ] Implement `replace_llama_attention_with_flashcore()` utility
- [ ] Test with single layer first

### **Step 2: Correctness Testing (6-8 hours)**
- [ ] Implement `test_llama31_generation_correctness()`
- [ ] Test with multiple prompts
- [ ] Verify token-by-token outputs match
- [ ] Debug any discrepancies

### **Step 3: Performance Benchmarking (6-8 hours)**
- [ ] Implement decode benchmark
- [ ] Compare to PyTorch SDPA baseline
- [ ] Verify <10ms target for decode@2048
- [ ] Profile if targets not met

### **Step 4: Documentation (3-4 hours)**
- [ ] Write usage guide
- [ ] Document API
- [ ] Create migration guide
- [ ] Document known limitations

---

## 🎯 **ACCEPTANCE CRITERIA**

### **Functional**
- ✅ Generates identical output to HuggingFace reference (50 tokens)
- ✅ No crashes or errors during generation
- ✅ KV cache managed correctly
- ✅ Works with all 32 layers

### **Performance**
- ✅ Decode: <10ms for cache=2048 (target)
- ✅ Competitive with PyTorch SDPA baseline
- ✅ No memory leaks

### **Documentation**
- ✅ Clear installation instructions
- ✅ Usage examples with LLaMA 3.1
- ✅ Performance comparison table

---

## 🚀 **USAGE EXAMPLE**

### **Quick Start**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from integration.llama_flashcore import replace_llama_attention_with_flashcore

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="cuda"
)

# Replace attention with FlashCore Triton kernels
replace_llama_attention_with_flashcore(model)

# Use normally
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
inputs = tokenizer("Once upon a time", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## 🎯 **SUCCESS METRICS**

**After Phase 4 Completion**:

✅ **Functional**: LLaMA 3.1 8B inference working  
✅ **Performance**: <10ms decode latency  
✅ **Memory**: 4× savings from GQA  
✅ **Correctness**: 100% match to reference  

**Grade Transformation**:
```
FROM: C- (toy kernel, no users)
TO:   A- (production-ready LLM inference)
```

---

## 🚀 **READY TO IMPLEMENT**

**Prerequisites**: Phases 1-3 (KV Cache, GQA, Causal) complete in Triton

**Next Step**: Create `integration/llama_flashcore.py`

**Command for Cursor**:
```
Implement LLaMA 3.1 8B integration using FlashCore Triton kernels.
Create integration module with drop-in replacement for LlamaAttention.
Validate correctness with test_llama31_generation_correctness().
Benchmark with benchmark_llama31.py.
Target: Identical outputs, <10ms decode latency.
```

---

**Status**: ⏳ Awaiting Phases 1-3 completion (Triton)  
**Estimated Time**: 20-25 hours  
**Priority**: 🎯 VALIDATION (Proves production readiness)

