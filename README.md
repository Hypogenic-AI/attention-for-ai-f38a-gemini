# Universal Attention Mechanism Research

## Project Overview
This project investigates the hypothesis that the mathematical "attention mechanism" is not only central to the effectiveness of transformers in Generative AI, but also underlies the core mechanics of the broader Internet field. We experimentally demonstrate how attention functions as a powerful information bottleneck in neural networks and similarly captures sequential user interest in internet platforms better than non-attention baselines.

## Key Findings
- **Attention as an Information Bottleneck (AI)**: By extracting attention maps from GPT-2, we proved that self-attention acts as a strict informational filter. The mean attention entropy of the model was measured at **1.89 nats**, drastically lower (more focused) than the baseline uniform entropy of **4.06 nats**.
- **Attention as an Interest Tracker (Internet)**: A Self-Attention Sequential Recommendation model (SASRec) successfully captured simulated human browsing trajectories, achieving a **68.5% Hit Rate @ 5**, vastly outperforming a popularity-based content distribution baseline which only achieved **12.0%**.
- **The Theoretical Link**: The same multi-head scaled dot-product attention mathematics that power LLM reasoning equivalently explain and predict how internet architectures compete for and monetize limited human cognitive focus.

## How to Reproduce
1. **Environment Setup**:
   Ensure you have Python 3.10+ and install dependencies using `uv` (or `pip`):
   ```bash
   uv venv
   source .venv/bin/activate
   uv add torch transformers numpy pandas matplotlib scikit-learn scipy seaborn tqdm
   ```

2. **Run Experiments**:
   Execute the experimental scripts sequentially:
   ```bash
   # Experiment 1: Generative AI Attention Analysis
   python src/experiment1_generative_ai.py
   
   # Experiment 2: Internet User Behavior Modeling
   python src/experiment2_internet_attention.py
   ```

3. **View Results**:
   Results and visualizations will be saved to the `results/` and `results/plots/` directories.

## File Structure Overview
- `planning.md`: Initial research plan and hypothesis decomposition.
- `REPORT.md`: Comprehensive final research report detailing methodology and results.
- `src/`: Contains the Python scripts for both experiments.
- `results/`: Contains the generated JSON metric logs and matplotlib plots (`results/plots/`).
- `datasets/` and `papers/`: Gathered resources and background literature for this research.

For full details and a comprehensive analysis, please see [REPORT.md](./REPORT.md).