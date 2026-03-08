import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def compute_attention_metrics(attention_weights):
    """
    attention_weights shape: (batch_size, num_heads, seq_len, seq_len)
    """
    # Average across batch and heads to get (seq_len, seq_len)
    avg_attention = attention_weights.mean(dim=(0, 1)).detach().numpy()
    seq_len = avg_attention.shape[0]
    
    # Calculate uniform attention baseline
    uniform_attn = np.ones(seq_len) / seq_len
    uniform_entropy = entropy(uniform_attn)
    
    # Calculate entropy for each token's attention distribution
    entropies = []
    sparsities = []
    
    for i in range(seq_len):
        attn_dist = avg_attention[i, :i+1] # Causal mask: only look at past
        # Normalize to sum to 1
        if attn_dist.sum() > 0:
            attn_dist = attn_dist / attn_dist.sum()
        else:
            continue
            
        # Entropy
        ent = entropy(attn_dist)
        entropies.append(ent)
        
        # Sparsity: Gini index or just L1/L2 ratio, let's use % of weights > uniform
        sparsity = np.mean(attn_dist > (1.0 / len(attn_dist)))
        sparsities.append(sparsity)
        
    return {
        "avg_entropy": float(np.mean(entropies)),
        "uniform_entropy": float(uniform_entropy),
        "avg_sparsity": float(np.mean(sparsities))
    }, avg_attention

def main():
    set_seed(42)
    os.makedirs("results/plots", exist_ok=True)
    
    print("Loading GPT-2 model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2", output_attentions=True)
    
    text = "The attention mechanism is the core of generative AI. Human perception has bottlenecks and cannot process all information; attention is the bottleneck through which the human brain and AI neural network filters and focuses information. The theory model of the internet is also built precisely on human attention or connection."
    
    inputs = tokenizer(text, return_tensors="pt")
    
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        
    # outputs.attentions is a tuple of (layer_1_attn, ..., layer_n_attn)
    # Each is shape (batch_size, num_heads, seq_len, seq_len)
    attentions = outputs.attentions
    
    results = {}
    layer_entropies = []
    
    for layer_idx, layer_attn in enumerate(attentions):
        metrics, avg_attn = compute_attention_metrics(layer_attn)
        results[f"layer_{layer_idx}"] = metrics
        layer_entropies.append(metrics["avg_entropy"])
        
        # Plot attention map for the last layer
        if layer_idx == len(attentions) - 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_attn, cmap="viridis")
            plt.title(f"GPT-2 Last Layer Attention Map\nEntropy: {metrics['avg_entropy']:.2f} (Uniform: {metrics['uniform_entropy']:.2f})")
            plt.xlabel("Key Position")
            plt.ylabel("Query Position")
            plt.tight_layout()
            plt.savefig("results/plots/gpt2_attention_map.png")
            plt.close()
            
    # Save metrics
    with open("results/generative_ai_attention_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Plot entropy across layers
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(layer_entropies)), layer_entropies, marker='o', label="GPT-2 Attention Entropy")
    plt.axhline(y=results["layer_0"]["uniform_entropy"], color='r', linestyle='--', label="Uniform Attention (Baseline)")
    plt.title("Attention Entropy Across GPT-2 Layers")
    plt.xlabel("Layer")
    plt.ylabel("Entropy (nats)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/gpt2_entropy_across_layers.png")
    plt.close()
    
    print(f"Mean Attention Entropy: {np.mean(layer_entropies):.2f}")
    print(f"Uniform Attention Entropy: {results['layer_0']['uniform_entropy']:.2f}")
    print("Experiment 1 complete. Results saved to results/")

if __name__ == "__main__":
    main()
