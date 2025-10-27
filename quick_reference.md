# 🚀 Quick Reference

## Essential Commands

### Setup
```bash
# First time setup
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.template .env
nano .env                         # Add your API key
python setup.py                   # Validate setup
```

### Training
```bash
# Basic training
python train_roberta.py

# Quick test (3 epochs)
python train_roberta.py --epochs 3

# Full training with testing
python train_roberta.py --epochs 15 --test

# High-performance (requires good GPU)
python train_roberta.py --epochs 20 --batch_size 32
```

### Running the App
```bash
# Start the app
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8080

# Kill existing Streamlit process
pkill -f streamlit
```

### Testing
```bash
# Validate setup
python setup.py

# Run predefined test cases
python test_rag.py

# Interactive testing mode
python test_rag.py --interactive

# Test specific question
python test_rag.py --test "I feel anxious all the time"

# Analyze retrieval quality
python test_rag.py --analyze

# Or use Jupyter notebook
jupyter notebook RAG.ipynb
```

---

## File Locations

| File/Directory | Purpose | In Git? |
|----------------|---------|---------|
| `app.py` | Main application | ✅ Yes |
| `train_roberta.py` | Training script | ✅ Yes |
| `RAG.ipynb` | Testing notebook | ✅ Yes |
| `.env` | API keys | ❌ No (secret) |
| `models/*.pth` | Trained models | ❌ No (too large) |
| `data/*.csv` | Datasets | ❌ No (privacy) |
| `outputs/*.csv` | Chat logs | ❌ No (user data) |

---

## Training Arguments

```bash
python train_roberta.py \
  --data_path data/counselchat-data.csv \
  --output_dir models \
  --epochs 10 \
  --batch_size 16 \
  --lr 3e-5 \
  --max_length 512 \
  --test
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | `data/counselchat-data.csv` | Dataset location |
| `--output_dir` | `models` | Save location |
| `--epochs` | `10` | Training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `3e-5` | Learning rate |
| `--max_length` | `512` | Max tokens |
| `--test` | `False` | Test after training |

---

## Common Workflows

### Daily Development
```bash
source venv/bin/activate
streamlit run app.py
```

### Retrain Model
```bash
source venv/bin/activate
python train_roberta.py --epochs 15
streamlit run app.py
```

### Test Pipeline
```bash
source venv/bin/activate
jupyter notebook RAG.ipynb
# Run cells interactively
```

### Update and Deploy
```bash
git pull origin main
pip install -r requirements.txt
python setup.py
python train_roberta.py --epochs 10
streamlit run app.py
```

---

## Troubleshooting One-Liners

```bash
# Check Python version
python --version

# Check if packages installed
pip list | grep -E "streamlit|torch|transformers"

# Check if API key set
cat .env | grep OPENAI_API_KEY

# Check if model exists
ls -lh models/rag_model_checkpoint.pth

# Check if dataset exists
wc -l data/counselchat-data.csv

# Kill stuck Streamlit
pkill -f streamlit

# Clear Streamlit cache
rm -rf ~/.streamlit/cache

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall everything
pip install -r requirements.txt --force-reinstall
```

---

## Environment Variables

### .env file structure
```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional
MODEL_PATH=models/rag_model_checkpoint.pth
DATA_PATH=data/counselchat-data.csv
DEVICE=cuda
```

---

## Git Commands

```bash
# Initial setup
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/username/counselchat.git
git push -u origin main

# Regular workflow
git status
git add .
git commit -m "Description of changes"
git push

# Before committing: CHECK FOR SECRETS
git diff                           # Review changes
cat .env                           # Verify not in git
cat .gitignore                     # Verify .env is listed
```

---

## Performance Benchmarks

### Training Time (approx)

| Hardware | Epochs | Time |
|----------|--------|------|
| CPU only | 10 | ~2-3 hours |
| GPU (RTX 3060) | 10 | ~15-20 min |
| GPU (RTX 4090) | 10 | ~8-10 min |

### Inference Time

| Hardware | Per Request |
|----------|-------------|
| CPU | ~2-3 seconds |
| GPU | ~0.5 seconds |

---

## Useful Python Snippets

### Test API Key
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
print(f"API key loaded: {api_key[:10]}...")
```

### Quick Topic Prediction
```python
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

checkpoint = torch.load('models/rag_model_checkpoint.pth')
tokenizer = checkpoint['tokenizer']
le = checkpoint['label_encoder']
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(le.classes_))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

text = "I feel anxious all the time"
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
outputs = model(**inputs)
predicted = torch.argmax(outputs.logits, dim=1).item()
print(f"Topic: {le.inverse_transform([predicted])[0]}")
```

### Check Dataset
```python
import pandas as pd

df = pd.read_csv('data/counselchat-data.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Topics: {df['topics'].nunique()}")
print(df['topics'].value_counts())
```

---

## Keyboard Shortcuts

### Streamlit App
- `Ctrl+C` - Stop the app
- `R` - Refresh the app
- `Ctrl+Shift+R` - Hard refresh

### Jupyter Notebook
- `Shift+Enter` - Run cell
- `Ctrl+Enter` - Run cell (stay)
- `A` - Insert cell above
- `B` - Insert cell below
- `DD` - Delete cell

---

## Port Reference

| Service | Default Port |
|---------|--------------|
| Streamlit | 8501 |
| Jupyter | 8888 |

Change Streamlit port:
```bash
streamlit run app.py --server.port 8080
```

---

## Resource Links

- **Streamlit Docs**: https://docs.streamlit.io
- **Transformers**: https://huggingface.co/docs/transformers
- **PyTorch**: https://pytorch.org/docs
- **OpenAI API**: https://platform.openai.com/docs
- **scikit-learn**: https://scikit-learn.org/stable/

---

## Version Info

Check versions:
```bash
python --version
pip show streamlit
pip show torch
pip show transformers
pip show openai
```

---

**Keep this reference handy! 📌**
