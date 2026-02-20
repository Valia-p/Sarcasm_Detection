# 🧠 Multimodal Sarcasm Detection using Text & Emoji Representations

> When emojis speak louder than words: Investigating multimodal sarcasm detection in social media using Transformer-based architectures.

---

## 📌 Overview

Sarcasm detection is a challenging Natural Language Processing (NLP) task where the intended meaning of a sentence differs from its literal interpretation.

In social media communication, sarcasm is often conveyed not only through text but also through paralinguistic cues such as emojis. While most existing sarcasm detection models rely solely on textual input, emojis frequently act as pragmatic markers that reinforce or contradict textual meaning.

This project investigates whether incorporating emoji-derived visual information can improve sarcasm detection performance compared to a strong text-only Transformer baseline.

We implement and compare:

- A **text-only baseline** based on RoBERTa  
- A **multimodal variant** combining textual and emoji-based visual representations  

Both models are evaluated on the **iSarcasmEval dataset** using a controlled experimental setup.

---

## 🎯 Objectives

- Build a strong Transformer-based baseline for sarcasm detection
- Investigate the contribution of emoji-based visual signals
- Compare text-only vs multimodal architectures
- Analyze performance trade-offs between precision and recall
- Evaluate sarcasm detection under imbalanced classification settings

---

## 🏗️ Methodology

### 🔹 Text-only Baseline

The baseline model is based on:

- Pretrained **RoBERTa encoder**
- Cascaded:
  - Multi-Head Self-Attention blocks  
  - Depthwise Convolution layers  

This architecture allows the model to capture:

- Global contextual dependencies  
- Local linguistic patterns  

The final `[CLS]` embedding is passed through a linear classifier for sarcasm prediction.

---

### 🔹 Multimodal Variant (Text + Emoji Vision Features)

The multimodal model extends the baseline by incorporating emoji-based information:

1. Emojis are extracted from raw tweets  
2. Rendered into fixed-size **2×2 image grids**
3. Processed using a pretrained **CLIP Vision Encoder**
4. Emoji count is added as an additional scalar feature

Final prediction is based on the **late fusion** of:

- Textual embeddings  
- Visual emoji features  
- Emoji count feature  

---

## 📊 Dataset

Experiments are conducted on the **iSarcasmEval Dataset**:

- English tweets annotated for sarcasm  
- Imbalanced class distribution  
- Stratified Train / Validation split  
- Unseen Test Set used for final evaluation  

To mitigate class imbalance:

- Weighted Cross-Entropy Loss is used  
- Performance is evaluated primarily using **F1-score**

---

## ⚙️ Training Configuration

| Parameter        | Value     |
|------------------|----------|
| Optimizer        | AdamW    |
| Learning Rate    | 2e-5     |
| Weight Decay     | 0.01     |
| Epochs           | 4        |
| Scheduler        | Linear   |
| Loss Function    | Weighted Cross Entropy |
| Gradient Clipping| Enabled  |

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| Text-only Baseline | 0.7850 | 0.3366 | 0.5200 | 0.4086 |
| Multimodal Variant | 0.8093 | 0.3696 | 0.4750 | 0.4158 |

### 🔍 Key Observations

- Multimodal model slightly improves F1-score
- Precision increases significantly  
- Recall decreases slightly  
- Emoji information helps in specific sarcastic cases  
- Precision–Recall trade-off observed

The multimodal model:

- Produces fewer false positives  
- Misses slightly more sarcastic instances  

---

## 🧪 Qualitative Analysis

Emoji-based visual features:

- Improve classification in sarcasm cases involving emotional exaggeration
- Influence model decision boundaries
- Provide complementary signals beyond textual irony

However:

- Contribution is not consistent across all samples
- Textual sarcasm remains difficult without contextual knowledge

---

## 📉 Limitations

- Emojis are not present in all tweets  
- Emoji grid representation may lose semantic meaning  
- CLIP encoder mostly frozen during training  
- Sarcasm depends heavily on external context and world knowledge  

---

## 🚀 Future Work

- Learn emoji embeddings jointly with text  
- Apply cross-modal attention mechanisms  
- Fine-tune larger parts of the vision encoder  
- Extend evaluation to multilingual datasets  
- Model emoji semantics beyond visual appearance  

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- HuggingFace Transformers  
- RoBERTa  
- CLIP  
- NumPy  
- Pandas  
- Matplotlib  

---

