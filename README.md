# Hindi Ambiguity Resolution using Qwen2.5 + LoRA

NLP — Fine-tuning a Transformer (< 2B parameters) to handle ambiguity in Hindi language.

## Problem Statement
Ambiguity in NLP occurs when a word or sentence has multiple possible interpretations. Hindi is especially challenging due to its morphological richness and code-switching (Hinglish).

## Types of Ambiguity Covered
| Type | Example | Meanings |
| Time Ambiguity | कल | yesterday / tomorrow |
| Morphological | सोना | sleep / gold |
| Pronoun | वह गया | who left? |
| Sarcasm | बहुत होशियार हो | genuine / sarcastic |
| Idiom | पानी में मछली | literal / idiomatic |
| Hinglish | mood off है | code-switched meaning |
| Syntactic | sentence structure unclear | multiple parse trees |
| Pragmatic | context-dependent meaning | intent varies |
| Place | place name vs common noun | ambiguous reference |
| Object-Action | object vs action meaning | same word, different role |

## Model Details
| Component | Detail |
| Base Model | Qwen/Qwen2.5-0.5B |
| Parameters | ~500M (under 2B limit) |
| Method | LoRA (PEFT) |
| Trainable Params | ~1-2% of total |
| Dataset | 1000 Hindi sentences, 10 categories |
| Training | 3 epochs, SFTTrainer, fp16 |
| Platform | Kaggle (T4 GPU) |

## Dataset
- 1000 annotated Hindi sentences
- 10 ambiguity categories
- 391 unique ambiguous words
- Custom annotated

## How to Run
1. Open `hindi_ambiguity_nlp_qwen_lora.ipynb` on Kaggle
2. Add `hindi_ambiguity_dataset.csv` as a Kaggle dataset
3. Enable GPU T4 and Internet
4. Run all cells

## Results
Model successfully learns to disambiguate Hindi words based on context.
Example:
- `कल वह बाज़ार गया था` → कल = **yesterday**
- `कल हम दिल्ली जाएंगे` → कल = **tomorrow**

## Tech Stack
- Python 3.10
- PyTorch
- HuggingFace Transformers
- PEFT (LoRA)
- TRL (SFTTrainer)
- Qwen2.5-0.5B
