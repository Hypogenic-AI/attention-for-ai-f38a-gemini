# Literature Review

## Research Area Overview
The hypothesis explores the universality of the "attention mechanism," suggesting it extends beyond its implementation in Transformer architectures (like LLMs) into the broader Generative AI field and the "attention economy" of the Internet. The literature supports this by showing how transformers have revolutionized multiple domains (CV, Recommendation) and how digital platforms strategically capture human attention using AI-driven mechanisms.

### Key Papers

#### Paper 1: Attention Is All You Need
- **Authors**: Vaswani et al.
- **Year**: 2017
- **Source**: NeurIPS
- **Key Contribution**: Introduced the Transformer architecture and the self-attention mechanism, replacing RNNs/CNNs for sequence modeling.
- **Methodology**: Scaled dot-product attention and multi-head attention.
- **Relevance**: Foundation for the entire "Attention" movement in AI.

#### Paper 2: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **Authors**: Dosovitskiy et al.
- **Year**: 2021 (arXiv 2020)
- **Source**: ICLR
- **Key Contribution**: Showed that the transformer architecture (ViT) can outperform CNNs in computer vision when pre-trained on large datasets.
- **Relevance**: Supports the "entire generative AI field" part of the hypothesis by demonstrating attention's effectiveness in vision.

#### Paper 3: Attention is all they need: Cognitive science and the (techno)political economy of attention in humans and machines
- **Authors**: González de la Torre et al.
- **Year**: 2024
- **Source**: arXiv
- **Key Contribution**: Analyzes the "attention economy" from a cognitive science and political economy perspective, linking AI-driven engagement to the capture of human attentional patterns.
- **Methodology**: 4E approach to cognitive science (embodied, extended, enactive, ecological).
- **Relevance**: Directly links the "attention mechanism" in AI to the "attention economy" of the Internet and human behavior.

#### Paper 4: The Rising Entropy of English in the Attention Economy
- **Authors**: Charlie Pilgrim, Weisi Guo, Thomas T. Hills
- **Year**: 2021
- **Source**: arXiv
- **Key Contribution**: Provides empirical evidence that media producers generate higher entropy content in shorter formats to compete for human attention.
- **Methodology**: Ecological model of the attention economy, Zipf's law, information foraging.
- **Relevance**: Demonstrates how the "Internet field" is structured around competing for and utilizing human attention.

#### Paper 5: Reusable Self-Attention-based Recommender System for Fashion
- **Authors**: Several
- **Year**: 2022
- **Source**: arXiv
- **Key Contribution**: Implementation of self-attention for sequential recommendation in an Internet product context.
- **Relevance**: Shows practical application of attention mechanisms in core Internet product features (Recommendation).

### Common Methodologies
- **Self-Attention**: Used in NLP, CV, and Recommender Systems to capture dependencies regardless of distance.
- **Information Foraging**: Used to model how humans search for and consume information in the attention economy.
- **Entropy Analysis**: Used to measure information density and competition in the attention market.

### Standard Baselines
- **RNNs/CNNs**: Traditional baselines for sequence and image modeling.
- **Collaborative Filtering**: Traditional baseline for recommender systems.
- **Zipf's Law**: Baseline for understanding word/item frequency distributions.

### Datasets in the Literature
- **SQuAD, GLUE**: NLP benchmarks.
- **ImageNet**: Computer Vision benchmark.
- **MovieLens, Amazon Reviews**: Recommender systems benchmarks.
- **Clickstream Data**: Used for analyzing Internet user behavior.

### Gaps and Opportunities
- **Cross-modal Attention**: How attention mechanisms unify different types of data (text, image, audio, behavior).
- **Cognitive Impact**: Long-term effects of AI-mediated attention capture on human autonomy.
- **Efficiency**: Reducing the computational cost of "attention" in large-scale systems.

### Recommendations for Our Experiment
- **Dataset**: Use a mix of "Model Attention" (e.g., extracting maps from ViT/GPT) and "Human Attention" (e.g., Clickstream/MovieLens).
- **Metric**: Entropy of attention distributions, Click-through rates, Attention map sparsity.
- **Code**: Build on `minGPT` for model analysis and `SASRec` for behavior analysis.
