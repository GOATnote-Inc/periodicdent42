#!/usr/bin/env python3
# Copyright 2025 GOATnote Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLaMA 3.1 Integration for FlashCore Triton Kernels

Drop-in replacement for HuggingFace LlamaAttention that uses FlashCore's
optimized Triton kernels for attention computation.

Supports:
- ✅ GQA (32:8 ratio for LLaMA 3.1 8B)
- ✅ KV Cache (incremental inference)
- ✅ Causal Masking (autoregressive generation)
- ✅ RoPE (rotary position embeddings, applied before attention)
- ✅ Seamless integration with HuggingFace transformers

Usage:
    from transformers import AutoModelForCausalLM
    from flashcore.llama_integration import replace_llama_attention_with_flashcore
    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    replace_llama_attention_with_flashcore(model)
    # Now model uses FlashCore Triton kernels
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import gc


def replace_llama_attention_with_flashcore(model, verbose: bool = True):
    """
    Replace all LlamaAttention modules in a model with FlashCore Triton version.
    
    This function monkey-patches the attention layers to use FlashCore's optimized
    Triton kernels while preserving all weights and model behavior.
    
    Args:
        model: HuggingFace LlamaForCausalLM model
        verbose: Print replacement status
    
    Returns:
        model: Modified model with FlashCore attention
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
        >>> replace_llama_attention_with_flashcore(model)
        >>> # Now use model normally with FlashCore kernels
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
    except ImportError:
        raise ImportError(
            "HuggingFace transformers not found. Install with: pip install transformers"
        )
    
    from flashcore.fast.attention_production import attention_with_kv_cache
    
    # Define the FlashCore attention class inside this function to ensure
    # it has access to the attention_with_kv_cache function
    class LlamaFlashCoreAttention(LlamaAttention):
        """
        Drop-in replacement for LlamaAttention using FlashCore Triton kernels.
        
        This class inherits from LlamaAttention and overrides only the forward
        method to use FlashCore's optimized attention implementation. All other
        functionality (QKV projections, RoPE, output projection) remains unchanged.
        """
        
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
            """
            Forward pass using FlashCore Triton kernels.
            
            Args:
                hidden_states: [batch, seq_len, hidden_dim]
                attention_mask: Optional attention mask (not used with FlashCore causal)
                position_ids: Position IDs for RoPE
                past_key_value: Optional cached key/value states
                output_attentions: If True, return attention weights (not supported)
                use_cache: If True, return updated cache
                cache_position: Cache position indices (transformers 4.36+)
            
            Returns:
                attn_output: Attention output [batch, seq_len, hidden_dim]
                attn_weights: None (not supported by FlashCore)
                past_key_value: Updated cache if use_cache=True
            """
            if output_attentions:
                raise NotImplementedError(
                    "output_attentions=True not supported with FlashCore. "
                    "Use attn_implementation='eager' or 'sdpa' instead."
                )
            
            bsz, q_len, _ = hidden_states.size()
            
            # QKV projections (unchanged from original LlamaAttention)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            
            # Reshape to multi-head format: [B, H, S, D]
            # Get attributes from config (newer transformers API)
            num_query_heads = getattr(self, 'num_heads', self.config.num_attention_heads)
            num_kv_heads = getattr(self, 'num_key_value_heads', self.config.num_key_value_heads)
            head_dim = getattr(self, 'head_dim', self.config.hidden_size // num_query_heads)
            
            query_states = query_states.view(
                bsz, q_len, num_query_heads, head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, num_kv_heads, head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, num_kv_heads, head_dim
            ).transpose(1, 2)
            
            # Apply RoPE (done BEFORE attention, unchanged from original)
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            
            # Handle cache format conversion (HuggingFace -> FlashCore)
            if past_key_value is not None:
                # Check if using new DynamicCache format (transformers 4.36+)
                if hasattr(past_key_value, 'key_cache'):
                    # New format: DynamicCache object
                    # Note: DynamicCache doesn't track seq_lens, so we can't use it directly
                    # FlashCore will need to maintain its own 3-tuple format
                    raise NotImplementedError(
                        "DynamicCache format not yet supported. "
                        "Use tuple format for now: past_key_value=(K, V, seq_lens)"
                    )
                else:
                    # Tuple format: check if FlashCore 3-tuple or HuggingFace 2-tuple
                    if len(past_key_value) == 3:
                        # Already FlashCore format (K, V, seq_lens)
                        past_kv_tuple = past_key_value
                    elif len(past_key_value) == 2:
                        # HuggingFace 2-tuple: This is ambiguous! We don't know the fill length.
                        # For now, reject it and require 3-tuple format
                        raise ValueError(
                            "2-tuple cache format is ambiguous (no seq_lens tracking). "
                            "Please use FlashCore 3-tuple format: (K_cache, V_cache, seq_lens)"
                        )
                    else:
                        raise ValueError(f"Unexpected cache tuple length: {len(past_key_value)}")
            else:
                past_kv_tuple = None
            
            # FLASHCORE TRITON ATTENTION (replaces SDPA/eager implementation)
            # This is the only change from the original LlamaAttention
            attn_output, updated_cache = attention_with_kv_cache(
                query=query_states,                       # [B, H_q, S, D]
                key=key_states,                           # [B, H_kv, S, D]
                value=value_states,                       # [B, H_kv, S, D]
                past_key_value=past_kv_tuple,            # Optional cached KV
                is_causal=True,                           # LLaMA is autoregressive
                update_cache=use_cache,                   # Update cache if requested
                num_query_heads=num_query_heads,          # Use local variable
                num_kv_heads=num_kv_heads,                # Use local variable
            )
            
            # Reshape output back to [B, S, H*D]
            attn_output = attn_output.transpose(1, 2).contiguous()
            hidden_size = getattr(self, 'hidden_size', self.config.hidden_size)
            attn_output = attn_output.reshape(bsz, q_len, hidden_size)
            
            # Output projection (unchanged from original)
            attn_output = self.o_proj(attn_output)
            
            # Return cache in FlashCore 3-tuple format
            # (LLaMA generation loops need to pass this back to us)
            if use_cache:
                cache_to_return = updated_cache  # (K_cache, V_cache, seq_lens)
            else:
                cache_to_return = None
            
            # Return: (output, attention_weights=None, cache)
            return attn_output, None, cache_to_return
        
        def _apply_rotary_pos_emb(self, q, k, cos, sin):
            """
            Apply rotary position embeddings to query and key tensors.
            
            This method is called from within the forward pass and handles
            the application of RoPE to the query and key states.
            """
            # Import the function from transformers
            try:
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                return apply_rotary_pos_emb(q, k, cos, sin)
            except ImportError:
                # Fallback for older transformers versions
                # Implement basic RoPE if needed
                raise NotImplementedError(
                    "apply_rotary_pos_emb not found in transformers. "
                    "Please update transformers: pip install --upgrade transformers"
                )
    
    # Replace all LlamaAttention modules in the model
    replaced_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LlamaAttention):
            # Get parent module and attribute name
            parent_name, _, attr_name = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # Create FlashCore version (preserves config and weights)
            flashcore_attn = LlamaFlashCoreAttention(
                config=module.config,
                layer_idx=module.layer_idx
            )
            
            # Copy weights from original module
            flashcore_attn.load_state_dict(module.state_dict())
            flashcore_attn.to(module.q_proj.weight.device)
            flashcore_attn.to(module.q_proj.weight.dtype)
            
            # Replace the module
            setattr(parent, attr_name, flashcore_attn)
            replaced_count += 1
    
    if verbose:
        print(f"✅ Replaced {replaced_count} attention layers with FlashCore Triton kernels")
    
    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model


def get_flashcore_attention_stats(model):
    """
    Get statistics about FlashCore attention usage in a model.
    
    Args:
        model: HuggingFace model (potentially with FlashCore attention)
    
    Returns:
        dict: Statistics including layer count, memory savings, etc.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
    except ImportError:
        return {"error": "transformers not installed"}
    
    total_layers = 0
    flashcore_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            total_layers += 1
            # Check if this is our FlashCore version
            if hasattr(module, '__class__') and 'FlashCore' in module.__class__.__name__:
                flashcore_layers += 1
    
    stats = {
        'total_attention_layers': total_layers,
        'flashcore_layers': flashcore_layers,
        'using_flashcore': flashcore_layers > 0,
        'coverage': f"{flashcore_layers}/{total_layers}" if total_layers > 0 else "0/0",
    }
    
    # Add memory savings estimate for GQA
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'num_attention_heads') and hasattr(config, 'num_key_value_heads'):
            H_q = config.num_attention_heads
            H_kv = config.num_key_value_heads
            if H_q > H_kv:
                savings_ratio = H_q / H_kv
                stats['gqa_memory_savings'] = f"{savings_ratio:.1f}×"
                stats['gqa_ratio'] = f"{H_q}:{H_kv}"
    
    return stats


# Convenience function for quick integration
def load_llama_with_flashcore(model_name: str, **kwargs):
    """
    Load a LLaMA model and automatically replace attention with FlashCore.
    
    Args:
        model_name: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B")
        **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained()
    
    Returns:
        model: LlamaForCausalLM with FlashCore attention
    
    Example:
        >>> model = load_llama_with_flashcore(
        ...     "meta-llama/Llama-3.1-8B",
        ...     torch_dtype=torch.float16,
        ...     device_map="cuda"
        ... )
        >>> # Model is ready to use with FlashCore kernels
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "HuggingFace transformers not found. Install with: pip install transformers"
        )
    
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    
    print("Replacing attention with FlashCore Triton kernels...")
    replace_llama_attention_with_flashcore(model, verbose=True)
    
    # Print stats
    stats = get_flashcore_attention_stats(model)
    print(f"FlashCore integration complete:")
    print(f"  - Coverage: {stats['coverage']} layers")
    if 'gqa_ratio' in stats:
        print(f"  - GQA ratio: {stats['gqa_ratio']}")
        print(f"  - Memory savings: {stats['gqa_memory_savings']}")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("FlashCore LLaMA 3.1 Integration Example")
    print("=" * 80)
    
    import sys
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "meta-llama/Llama-3.1-8B"
    
    print(f"\nLoading {model_name} with FlashCore Triton kernels...\n")
    
    try:
        model = load_llama_with_flashcore(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print("\n✅ Model loaded successfully with FlashCore!")
        print("\nStats:")
        stats = get_flashcore_attention_stats(model)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

