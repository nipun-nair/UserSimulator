# UserSimulator
# User Simulation for Recommender Systems via TextGrad Optimization

This repository contains the implementation of our prompt optimization framework for LLM-based user simulators in conversational recommender systems. The code accompanies our paper accepted at DASFAA 2026 (Industry Track).

## Overview

We present a practical system that applies TextGrad-based prompt optimization to automated user simulation, achieving human-aligned behavioral patterns without manual prompt engineering or model retraining. Our approach addresses three critical challenges:

- **Systematic positive bias** leading to unrealistic acceptance rates
- **Data leakage** through inadvertent target item exposure
- **Limited behavioral diversity** failing to represent realistic user heterogeneity

## Key Features

- 🔒 **Privacy-preserving**: Local deployment via Ollama with Llama 3.3:70B
- 🎯 **Automated optimization**: TextGrad-based prompt tuning with entropy metrics
- 📊 **Comprehensive evaluation**: 5 conversational tasks + novel NegFeedback metric
- 🚀 **Reproducible**: Complete pipeline from data preprocessing to evaluation

## Requirements

### Hardware
- **Minimum**: NVIDIA GPU with 24GB VRAM (RTX 3090, RTX A5000, A10G)
- **Recommended**: NVIDIA GPU with 40-48GB VRAM (A100, RTX A6000)
- **RAM**: 64GB system memory
- **Storage**: 100GB for models and data

### Software
```bash
# Core dependencies
Python 3.10+
PyTorch 2.0+
Ollama 0.1.26+
TextGrad 0.1.4

# See requirements.txt for full list
```

## Installation

### 1. Install Ollama
```bash
curl https://ollama.ai/install.sh | sh
```

### 2. Download Llama 3.3:70B
```bash
ollama pull llama3.3:70b
```

### 3. Clone Repository
```bash
git clone https://github.com/[your-username]/textgrad-user-simulation.git
cd textgrad-user-simulation
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download Word Embeddings
```bash
# Download GloVe embeddings for diversity metrics
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
python -c "from gensim.scripts.glove2word2vec import glove2word2vec; glove2word2vec('glove.6B.50d.txt', 'glove.6B.50d.word2vec.txt')"
```

## Quick Start

### Step 1: Preprocess Amazon Reviews Data
```bash
python data_implementation.py
```

This creates task-specific datasets in the `data/` directory:
- `t1_items/` - ItemsTalk task
- `t2_bin_preference/` - Binary preference task
- `t3_open_preference/` - Open preference task
- `t4_requests/` - Recommendation request task
- `t6_feedback/` - Negative feedback task

### Step 2: Generate User Profiles and Responses
```bash
# Run all tasks (without and with TextGrad optimization)
python task_5items.py --start 0 --end 100 --target amazon_reviews

# Run specific tasks only
python task_5items.py --start 0 --end 10 --tasks 1 6  # ItemsTalk and NegFeedback only
```

**Arguments:**
- `--start`: Starting user index (default: 0)
- `--end`: Ending user index (default: 100)
- `--target`: Dataset target (default: amazon_reviews)
- `--tasks`: List of task IDs to run (default: [1,2,3,4,6])

### Step 3: Evaluate Results
```bash
python task_eval.py
```

This computes:
- Item entropy (ItemsTalk)
- Preference correlation (BinPref)
- Aspect/sentiment entropy (OpenPref)
- Diversity metrics: TTR, W2V, sentence similarity (RecRequest)
- LLM-based rejection quality scores (NegFeedback)

### Step 4: Generate GPT Baselines (Optional)
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your_key_here"

# Run baseline comparisons
python baseline_generate.py
python task_eval_baseline.py
```

## Project Structure

```
.
├── data/                           # Generated datasets
│   ├── t1_items/                  # ItemsTalk task data
│   ├── t2_bin_preference/         # Binary preference data
│   ├── t3_open_preference/        # Open preference data
│   ├── t4_requests/               # Recommendation request data
│   └── t6_feedback/               # Negative feedback data
├── data_implementation.py          # Data preprocessing pipeline
├── profile_creation.py             # Synthetic user profile generation
├── task_5items.py                  # Main task execution pipeline
├── textgrad_ollama.py              # TextGrad optimization implementation
├── task_eval.py                    # Evaluation metrics computation
├── baseline_generate.py            # GPT baseline generation
├── task_eval_baseline.py           # Baseline evaluation
├── human_eval.py                   # Human evaluation analysis
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Detailed Usage

### Profile Generation

Synthetic user profiles include:
- **Name**: Ethnicity-aware name generation
- **Age**: Random sampling (5-80 years)
- **Ethnicity**: 10 diverse categories
- **Pickiness level**: Acceptance threshold modeling

Profiles are generated via `RandomPerson` class in `profile_creation.py`.

### Profile Summarization

To prevent data leakage, user viewing histories are:
1. Randomly sampled (up to 5 movies)
2. Converted to natural language preferences
3. Used instead of raw item lists

Example:
```
Raw: ["The Godfather (1972)", "Goodfellas (1990)", "Casino (1995)"]
→ Summarized: "prefers character-driven crime dramas with moral ambiguity"
```

### TextGrad Optimization

Prompts are optimized using:
- **Entropy-based diversity metrics**: Item, aspect, sentiment entropy
- **Bias correction**: Yes-rating correlation, popularity penalization
- **Rejection quality**: NegFeedback scoring

Optimization runs for 3 epochs by default (configurable in `textgrad_ollama.py`).

### Task Descriptions

| Task | Description | Metrics |
|------|-------------|---------|
| **ItemsTalk** | Generate diverse movie mentions | Item entropy |
| **BinPref** | Binary preference alignment | Pearson correlation |
| **OpenPref** | Detailed review generation | Aspect/sentiment entropy |
| **RecRequest** | Implicit recommendation requests | TTR, W2V, sentence diversity |
| **NegFeedback** | Rejection with rationale | LLM-based scoring (0-10) |

## Evaluation Metrics

### Diversity Metrics
- **Item Entropy**: Measures unpredictability in movie selections
- **Aspect Entropy**: Distribution of review aspects (plot, acting, etc.)
- **Sentiment Entropy**: Variation in sentiment polarity
- **TTR (Type-Token Ratio)**: Vocabulary diversity
- **W2V Diversity**: Word embedding-based semantic diversity
- **Sentence Diversity**: Cosine distance between sentence embeddings

### Alignment Metrics
- **BinPref Correlation**: Pearson correlation between ratings and acceptance
- **NegFeedback Score**: Human-aligned rejection quality (0-10 scale)

## Expected Runtime

On NVIDIA A100 (40GB):
- **Profile generation**: ~2.5 hours (100 users)
- **TextGrad optimization**: ~10-14 hours (15 iterations)
- **Task evaluation**: ~1.5 hours
- **Total**: ~14-18 GPU-hours for 100 users

## Results

Our method achieves:
- **92% human-level item diversity** (9.269 vs. 10.067 entropy)
- **2.7× better preference correlation** (0.726 vs. 0.265 for GPT-4)
- **Realistic sentiment patterns** (1.274 vs. 0.000 for GPT baselines)
- **Higher NegFeedback correlation** (r=0.484 vs. r=-0.244 for GPT-3.5)

## Troubleshooting

### Common Issues

**1. Ollama connection error**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start Ollama service
ollama serve
```

**2. GPU out of memory**
```python
# Reduce batch size or use model quantization
# Edit textgrad_ollama.py:
CONFIG["model_name"] = "llama3.3:70b-q4_0"  # 4-bit quantized
```

**3. TextGrad optimization fails**
```
Error: "IndexError in optimizer.step()"
```
**Solution**: Smaller models (<70B) may not follow TextGrad instructions properly. Use Llama 3.3:70B or larger.

**4. Slow evaluation**
```bash
# Enable GPU for sentence transformers
# Edit task_eval.py line 22:
sent_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{nair2026prompt,
  title={Prompt Optimization for User Simulation in Recommender Systems: A Multi-Objective Framework},
  author={Nair, Nipun B and Wu, Tongtong and Wang, Teresa},
  booktitle={Proceedings of the 31st International Conference on Database Systems for Advanced Applications (DASFAA)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Amazon Reviews 2023 dataset by McAuley Lab
- TextGrad framework by Yuksekgonul et al.
- Ollama for local LLM deployment
- Monash University for computational resources

## Contact

For questions or issues:
- **Email**: nipun.nair@monash.edu
- **GitHub Issues**: [Create an issue](https://github.com/[your-username]/textgrad-user-simulation/issues)

## Future Work

- [ ] Cross-domain validation (books, music, e-commerce)
- [ ] Scale to 1000+ users for statistical power
- [ ] Multi-modal user simulation (text + images)
- [ ] Real-time adaptation for production systems
- [ ] Integration with popular RecSys frameworks (LensKit, Surprise)

---

**Note**: This is research code. For production deployment, additional testing and optimization are recommended.
