# Resources Catalog

## Summary
This catalog summarizes all resources gathered for the research project "The idea Attention is all you need not only applicable to transformers and LLMs, but also applicable to the entire generative AI and Internet field."

### Papers
Total papers downloaded: 9

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Attention Is All You Need | Vaswani et al. | 2017 | papers/1706.03762v7_Attention_Is_All_You_Need.pdf | Foundation of self-attention. |
| An Image is Worth 16x16 Words | Dosovitskiy et al. | 2021 | papers/2010.11929v2_An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale.pdf | Transformers in CV. |
| Attention is all they need | González de la Torre et al. | 2024 | papers/2405.06478v1_Attention_is_all_they_need_Cognitive_science_and_the_(techno)political_economy_of_attention_in_human.pdf | Cognitive science of attention economy. |
| The Rising Entropy of English | Pilgrim et al. | 2021 | papers/2112.02102_Rising_Entropy_Attention_Economy.pdf | Competition for human attention. |
| Reusable Self-Attention Recommender | Several | 2022 | papers/2211.16366v1_Reusable_Self-Attention-based_Recommender_System_for_Fashion.pdf | Attention in recommendations. |
| Not All Attention Is All You Need | - | 2021 | papers/2104.04692v3_Not_All_Attention_Is_All_You_Need.pdf | Critical view on attention. |
| An Image is Worth 16x16 Words, What is a Video Worth? | - | 2021 | papers/2103.13915v2_An_Image_is_Worth_16x16_Words,_What_is_a_Video_Worth?.pdf | Transformers for video. |
| Core Tokensets for Data-efficient Sequential Training | - | 2024 | papers/2410.05800v1_Core_Tokensets_for_Data-efficient_Sequential_Training_of_Transformers.pdf | Efficient transformers. |
| Element-wise Attention Is All You Need | - | 2025 | papers/2501.05730v1_Element-wise_Attention_Is_All_You_Need.pdf | Variation of attention mechanism. |

### Datasets
Total datasets gathered: 3 (Samples)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MovieLens Ratings | HuggingFace | 100 (Sample) | Recommendation | datasets/movielens/sample.csv | Classic interaction data. |
| E-commerce Behavior | HuggingFace | 100 (Sample) | User Behavior | datasets/ecommerce/sample.csv | Real-world product interaction. |
| Clickstream | HuggingFace | Not Saved (Loading Issue) | Internet Attention | - | Fallback to E-commerce behavior. |

### Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Transformers | github.com/huggingface/transformers | SOTA implementation | code/transformers_hf | Comprehensive library. |
| minGPT | github.com/karpathy/minGPT | Core LLM logic | code/minGPT | Easy to analyze. |
| SASRec | github.com/kang205/SASRec | Sequential Rec | code/SASRec | Attention for behavior. |

## Recommendations for Experiment Design

1. **Primary dataset(s)**: Use `movielens` and `ecommerce` for behavior analysis, and `ImageNet` (via `transformers` library) for model analysis.
2. **Baseline methods**: Compare self-attention based models (SASRec) against traditional baselines (Matrix Factorization) to show the "attention advantage" in Internet products.
3. **Evaluation metrics**:
   - **Model side**: Attention entropy, attention map sparsity.
   - **User side**: CTR, engagement time, session length.
4. **Code to adapt/reuse**: Use `minGPT` to extract attention maps from small language models and `SASRec` to demonstrate attention in recommendation.
