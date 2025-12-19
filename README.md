# CounselChat - An AI Powered Mental Health Assist for Councelers

An AI-powered companion for mental health counselors that combines Machine Learning (RoBERTa) and Large Language Models (GPT-4 mini) to deliver empathetic, insightful, and practical mental health advice.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red.svg)

## 🎯 Features

- **Topic Classification**: Automatically categorizes mental health concerns into 9 main topics using fine-tuned RoBERTa
- **Retrieval-Augmented Generation (RAG)**: Retrieves relevant professional responses from a counseling dataset
- **Personalized Advice**: Generates tailored, empathetic advice using GPT-4o mini
- **Interactive Chat Interface**: User-friendly Streamlit interface with chat history
- **Topic Filtering**: Improves retrieval accuracy by filtering similar cases by predicted topic

## 🏗️ Architecture

```
User Input → RoBERTa Classifier → Topic Prediction
                                        ↓
                            TF-IDF Similarity Search (filtered by topic)
                                        ↓
                            Retrieve Top-K Similar Q&A Pairs
                                        ↓
                            GPT-4o mini with Retrieved Context
                                        ↓
                            Empathetic, Actionable Advice
```

## 📋 Mental Health Topics Covered

1. **Anxiety & Stress** - General anxiety, stress management, sleep issues
2. **Relationships** - Intimacy, communication, social relationships
3. **Family & Parenting** - Family conflicts, parenting challenges
4. **Mental Health Disorders** - Depression, grief, diagnosis support
5. **Addiction & Abuse** - Substance abuse, self-harm, eating disorders
6. **Violence & Safety** - Domestic violence, anger management
7. **Professional & Legal** - Career counseling, workplace issues
8. **Identity & Spirituality** - LGBTQ+, spirituality, human sexuality
9. **Behavioral Changes** - Counseling fundamentals, behavior modification

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster inference)
- OpenAI API key

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/counselchat.git
cd counselchat
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

1. Copy the template:
```bash
cp .env.template .env
```

2. Edit `.env` and add your OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**🔒 Security Note:** Never commit your `.env` file to version control!


### Step 5: Add Your Files

Place the following files in their respective directories:
- `models/rag_model_checkpoint.pth` - Your trained RoBERTa model
- `data/counselchat-data.csv` - Your counseling dataset

Or train a new model:
```bash
python train_roberta.py --epochs 10
```

## 📊 Dataset Format

Your `counselchat-data.csv` should have the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `questionText` | Patient's question/concern | Yes |
| `answerText` | Professional counselor's response | Yes |
| `topic_group` | Mental health topic category | Optional* |

*Optional but recommended for better topic filtering in RAG

Example:
```csv
questionText,answerText,topic_group
"I feel anxious all the time","Consider practicing mindfulness...","Anxiety & Stress"
```

## 🎮 Usage

### Training the Model

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

### Testing the RAG Pipeline

```bash
# Run predefined tests
python test_rag.py

# Interactive mode
python test_rag.py --interactive

# Test specific question
python test_rag.py --test "I'm feeling anxious"

# Analyze retrieval quality
python test_rag.py --analyze
```

### Using the Jupyter Notebook

For experimentation and debugging:

```bash
jupyter notebook RAG.ipynb
```

### Testing Different Inputs

Try these example inputs in the chat:
- "I feel anxious all the time, what should I do?"
- "My partner and I are having communication issues"
- "I want to quit smoking but can't seem to stop"
- "I'm experiencing burnout from my high-pressure job"

## 🛠️ Training Your Own Model

### Option 1: Using Python Script (Recommended)

```bash
# Train with default settings
python train_roberta.py

# Train with custom settings
python train_roberta.py --epochs 15 --batch_size 32 --lr 2e-5

# Train and test immediately
python train_roberta.py --test
```

### Option 2: Using Jupyter Notebook

1. Prepare your dataset with `questionText` and `topics` columns
2. Open `notebooks/RoBERTa.ipynb`
3. Run all cells to train the model

See `USAGE_GUIDE.md` for detailed training instructions.

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

```bash
# Custom port
streamlit run app.py --server.port 8080

# Kill existing Streamlit process
pkill -f streamlit
```

## 📁 Project Structure

```
counselchat/
│
├── app.py                          # Main Streamlit application
├── train_roberta.py                # Model training script
├── setup.py                        # Setup validation
├── RAG.ipynb                       # RAG pipeline notebook
│
├── models/
│   ├── rag_model_checkpoint.pth    # Trained model (not in git)
│   └── training_history.json       # Training metrics
│
├── data/
│   └── counselchat-data.csv        # Dataset (not in git)
│
├── outputs/
│   └── chat_history.csv            # Saved conversations
│
├── notebooks/
│   └── RoBERTa.ipynb               # Original training notebook
│
├── .env                            # Environment variables (not in git)
├── .env.template                   # Template for .env
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── USAGE_GUIDE.md                  # Detailed usage instructions
```

## 🔧 Configuration

### Model Settings

Edit these parameters in `app.py`:

```python
# Number of similar cases to retrieve
top_k = 3

# Similarity threshold for filtering
threshold = 0.1

# Max tokens for GPT-3.5 response
max_tokens = 500
```

### RAG Parameters

You can adjust:
- **top_k**: Number of similar cases to retrieve (default: 3)
- **similarity_threshold**: Minimum similarity score (default: 0.1)
- **topic_filtering**: Enable/disable topic-based filtering (default: enabled)

## 🎨 Customization

### Adding New Topics

Edit the `topic_groups` dictionary in `RoBERTa.ipynb`:

```python
topic_groups = {
    'Your New Topic': ['keyword1', 'keyword2', 'keyword3'],
    # ... existing topics
}
```

### Changing the LLM

Replace GPT-4o mini with another model by modifying the `generate_llm_response` function:

```python
response = client.chat.completions.create(
    model="gpt-4",  # Change model here
    messages=[...],
    max_tokens=500
)
```

## ⚠️ Common Issues & Solutions

### Issue: "Model checkpoint not found"
**Solution**: Ensure `rag_model_checkpoint.pth` is in the `models/` directory

### Issue: "OpenAI API key not found"
**Solution**: Check that `.env` file exists and contains `OPENAI_API_KEY=your-key`

### Issue: "Rate limit exceeded"
**Solution**: The app has built-in retry logic. Wait a moment and try again.

### Issue: CUDA out of memory
**Solution**: Reduce batch size or use CPU by setting `DEVICE=cpu` in `.env`

## 📊 Model Performance

- **Training Accuracy**: ~80.8% on validation set
- **Topics**: 9 mental health categories
- **Dataset Size**: 1,376 question-answer pairs
- **Model**: RoBERTa-base fine-tuned for sequence classification

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: CounselChat mental health Q&A dataset
- Models: Hugging Face Transformers, OpenAI GPT-4o mini
- Framework: Streamlit for the web interface

## 📧 Contact

**Pramoth Guhan**
- GitHub: [@pramothguhan](https://github.com/pramothguhan)
- Email: guhan.p@northeastern.edu

## ⚖️ Disclaimer

This tool is designed to **assist** mental health professionals and should not be used as a replacement for professional mental health care. Always consult qualified healthcare providers for mental health concerns.

---

**Made with ❤️ for mental health awareness**
