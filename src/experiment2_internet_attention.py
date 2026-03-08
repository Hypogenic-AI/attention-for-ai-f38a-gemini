import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

# Simulate Internet User Behavior Data
# Users view items in a sequence. Some items strongly depend on previous items (representing human attention/interest flow).
def generate_synthetic_data(num_users=1000, seq_len=10, num_items=50):
    data = []
    # Create some transition rules to simulate "interest"
    # E.g., if you view item i, you are likely to view item (i+1) or (i+2)
    for _ in range(num_users):
        seq = [np.random.randint(1, num_items)]
        for _ in range(seq_len - 1):
            if np.random.rand() > 0.3:
                # Follow attention/interest
                next_item = (seq[-1] % (num_items - 1)) + 1
            else:
                # Random exploration
                next_item = np.random.randint(1, num_items)
            seq.append(next_item)
        data.append(seq)
    return np.array(data)

# Baseline: Most Popular Recommender
class PopularityBaseline:
    def __init__(self):
        self.item_counts = {}
        
    def fit(self, data):
        for seq in data:
            for item in seq[:-1]: # Don't train on last item
                self.item_counts[item] = self.item_counts.get(item, 0) + 1
                
    def predict(self, k=10):
        # Return top k popular items
        sorted_items = sorted(self.item_counts.items(), key=lambda x: x[1], reverse=True)
        return [item for item, count in sorted_items[:k]]

# Self-Attention Recommender (Simplified SASRec)
class SelfAttentionRec(nn.Module):
    def __init__(self, num_items, embed_dim=32, num_heads=1):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(100, embed_dim) # Max seq len
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, num_items + 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # Embeddings
        e = self.item_emb(x) + self.pos_emb(positions)
        
        # Causal mask to prevent looking ahead
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        # Self-attention
        attn_out, attn_weights = self.attn(e, e, e, attn_mask=mask)
        
        # Predict next item (using last hidden state)
        logits = self.fc(attn_out[:, -1, :])
        return logits, attn_weights

def evaluate(model, test_data, baseline=False, pop_model=None, k=5):
    hit = 0
    total = 0
    
    if baseline:
        preds = pop_model.predict(k)
        for seq in test_data:
            target = seq[-1]
            if target in preds:
                hit += 1
            total += 1
        return hit / total
        
    model.eval()
    with torch.no_grad():
        for seq in test_data:
            x = torch.tensor([seq[:-1]])
            target = seq[-1]
            logits, _ = model(x)
            
            # Get top k
            top_k = torch.topk(logits[0], k).indices.tolist()
            if target in top_k:
                hit += 1
            total += 1
            
    return hit / total

def main():
    set_seed(42)
    os.makedirs("results/plots", exist_ok=True)
    
    # 1. Prepare Data
    num_items = 50
    data = generate_synthetic_data(num_users=2000, seq_len=15, num_items=num_items)
    
    train_data = data[:1600]
    test_data = data[1600:]
    
    # 2. Train Popularity Baseline
    print("Training Popularity Baseline...")
    pop_model = PopularityBaseline()
    pop_model.fit(train_data)
    pop_hr = evaluate(None, test_data, baseline=True, pop_model=pop_model, k=5)
    print(f"Popularity Baseline HR@5: {pop_hr:.4f}")
    
    # 3. Train Self-Attention Model
    print("Training Self-Attention Model...")
    model = SelfAttentionRec(num_items=num_items)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 15
    batch_size = 64
    
    train_x = torch.tensor(train_data[:, :-1])
    train_y = torch.tensor(train_data[:, -1])
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        permutation = torch.randperm(train_x.size(0))
        
        for i in range(0, train_x.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_x[indices], train_y[indices]
            
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        losses.append(epoch_loss / (train_x.size(0) / batch_size))
        
    attn_hr = evaluate(model, test_data, k=5)
    print(f"Self-Attention HR@5: {attn_hr:.4f}")
    
    # Extract attention weights for a sample
    model.eval()
    sample_seq = torch.tensor([test_data[0][:-1]])
    _, attn_weights = model(sample_seq)
    attn_weights = attn_weights[0].detach().numpy()
    
    # 4. Save Results and Plots
    results = {
        "baseline_hr_at_5": pop_hr,
        "attention_hr_at_5": attn_hr,
        "improvement_pct": ((attn_hr - pop_hr) / pop_hr) * 100
    }
    
    with open("results/internet_attention_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Plot Training Loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, marker='o')
    plt.title("Self-Attention Recommender Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.savefig("results/plots/sasrec_training_loss.png")
    plt.close()
    
    # Plot Comparison
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["Popularity Baseline", "Self-Attention (SASRec)"], [pop_hr, attn_hr], color=['gray', 'blue'])
    plt.title("Internet Attention Modeling: Hit Rate @ 5")
    plt.ylabel("HR@5 (Higher is better)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("results/plots/recommender_comparison.png")
    plt.close()
    
    print("Experiment 2 complete. Results saved to results/")

if __name__ == "__main__":
    main()
