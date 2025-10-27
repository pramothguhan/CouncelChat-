"""
Legacy's Mental Health CounselChat - RAG Testing Script
Test the complete Retrieval-Augmented Generation pipeline

Usage:
    python test_rag.py                    # Run predefined test cases
    python test_rag.py --interactive      # Interactive mode
    python test_rag.py --test "your question here"
"""

from dotenv import load_dotenv
import os
from pathlib import Path

# 1. Load the .env file before accessing os.getenv
# (explicit path is extra-safe if your entry script lives in src/)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# 2. NOW read the values
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")
DATA_PATH = os.getenv("DATA_PATH")
DEVICE = os.getenv("DEVICE")

print("API key loaded?", OPENAI_API_KEY is not None)
print("Model path:", MODEL_PATH)
print("Data path:", DATA_PATH)
print("Device:", DEVICE)

import argparse
import pandas as pd
import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import time
from openai import OpenAIError
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()


# ========================================
# CONFIGURATION
# ========================================

class Config:
    """Configuration for RAG testing."""
    def __init__(self):
        self.model_path = 'models/rag_model_checkpoint.pth'
        self.data_path = 'data/counselchat-data.csv'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 512
        self.top_k = 3
        self.similarity_threshold = 0.1


# ========================================
# LOAD MODEL AND DATA
# ========================================

def load_model(config):
    """Load the trained RoBERTa model, tokenizer, and label encoder."""
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}\n")
    
    if not os.path.exists(config.model_path):
        print(f"❌ Model not found at: {config.model_path}")
        print("Please train the model first: python train_roberta.py")
        exit(1)
    
    print(f"Loading checkpoint from: {config.model_path}")
    checkpoint = torch.load(config.model_path, map_location=config.device, weights_only=False)
    
    tokenizer = checkpoint['tokenizer']
    le = checkpoint['label_encoder']
    
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=len(le.classes_)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"Device: {config.device}")
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Classes: {list(le.classes_)}")
    
    return model, tokenizer, le


def load_dataset(config):
    """Load and prepare the dataset for RAG."""
    print(f"\n{'='*60}")
    print("LOADING DATASET")
    print(f"{'='*60}\n")
    
    if not os.path.exists(config.data_path):
        print(f"❌ Dataset not found at: {config.data_path}")
        exit(1)
    
    print(f"Loading dataset from: {config.data_path}")
    df = pd.read_csv(config.data_path)
    
    # Keep necessary columns
    required_cols = ['questionText', 'answerText']
    if 'topic_group' in df.columns:
        required_cols.append('topic_group')
    
    df = df[required_cols].dropna()
    df['questionText'] = df['questionText'].astype(str)
    df['answerText'] = df['answerText'].astype(str)
    
    print(f"Dataset shape: {df.shape}")
    
    # Fit TF-IDF on questions (we search questions, retrieve answers)
    print("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['questionText'])
    
    print(f"✅ TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    
    return df, vectorizer, tfidf_matrix


# ========================================
# PREDICTION FUNCTIONS
# ========================================

def predict_topic(text, model, tokenizer, le, config):
    """Predict mental health topic from input text."""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=config.max_length
    ).to(config.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    return le.inverse_transform([predicted_class])[0]


def find_most_similar_answers(question, predicted_topic, vectorizer, tfidf_matrix, df, config):
    """
    Find top-K most similar questions and retrieve their answers.
    
    Args:
        question: User's input question
        predicted_topic: Topic predicted by classifier
        vectorizer: Fitted TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix of dataset questions
        df: DataFrame with questionText, answerText, topic_group
        config: Configuration object
    
    Returns:
        List of dicts with 'question', 'answer', 'score'
    """
    # Filter by topic if available
    if 'topic_group' in df.columns:
        topic_filtered_df = df[df['topic_group'] == predicted_topic].reset_index(drop=True)
        
        if len(topic_filtered_df) > 0:
            # Recalculate TF-IDF for filtered subset
            filtered_tfidf = vectorizer.transform(topic_filtered_df['questionText'])
            working_df = topic_filtered_df
            working_tfidf = filtered_tfidf
        else:
            # No matches in topic, use full dataset
            working_df = df
            working_tfidf = tfidf_matrix
    else:
        working_df = df
        working_tfidf = tfidf_matrix
    
    # Vectorize user question
    question_vector = vectorizer.transform([question])
    
    # Calculate similarities
    similarities = cosine_similarity(question_vector, working_tfidf).flatten()
    
    # Get top K indices
    top_indices = similarities.argsort()[-config.top_k:][::-1]
    
    # Build results
    results = []
    for idx in top_indices:
        if similarities[idx] > config.similarity_threshold:
            results.append({
                'question': working_df.iloc[idx]['questionText'],
                'answer': working_df.iloc[idx]['answerText'],
                'score': float(similarities[idx])
            })
    
    return results if results else [{
        'question': 'No similar case found',
        'answer': 'Providing general guidance.',
        'score': 0.0
    }]


# ========================================
# LLM RESPONSE GENERATION
# ========================================

def setup_openai():
    """Setup OpenAI client and return it."""
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set OPENAI_API_KEY in your .env file")
        exit(1)

    client = OpenAI(api_key=api_key)
    return client


def generate_llm_response(topic, user_input, retrieved_answers, client, retries=3):
    """Generate advice using the OpenAI Chat Completions API with retrieved context."""

    # Build context from retrieved answers
    context = "\n\n".join([
        f"Reference Case {i+1} (Similarity: {ans['score']:.2%}):\n"
        f"Patient Question: {ans['question']}\n"
        f"Counselor Response: {ans['answer']}"
        for i, ans in enumerate(retrieved_answers[:3])
    ])

    prompt = f"""
You are an empathetic mental health assistant.

**Patient's Concern:**
"{user_input}"

**Topic:** {topic}

**Similar Cases from Professional Counselors:**
{context}

**Instructions:**
- Directly address: "{user_input}"
- If references are too specific, generalize appropriately
- Be warm, empathetic, and actionable

**Provide a structured response:**
1. **Understanding the Challenge:** Validate their feelings (2-3 sentences)
2. **Practical Advice:** 2-3 clear, actionable steps
3. **Emotional Support:** Coping strategies
4. **Encouragement:** Positive closing message

Keep response under 400 tokens.
""".strip()

    last_error = None

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",  # smaller/cheaper reasoning model; update if you want another
                messages=[
                    {
                        "role": "system",
                        "content": "You are an empathetic mental health assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.7,
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            # catch generic Exception, since OpenAIError import in your file may mismatch sdk version
            print(f"❌ OpenAI API Error: {e}")
            last_error = e
            # light exponential backoff
            time.sleep(2 ** attempt)

    return "⚠️ Unable to generate advice due to API issues."


# ========================================
# TESTING FUNCTIONS
# ========================================

def run_single_test(question, model, tokenizer, le, vectorizer, tfidf_matrix, df, client, config, verbose=True):
    """Run a single test case through the pipeline."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"📝 User Input: {question}")
        print(f"{'='*60}")
    
    # Step 1: Predict Topic
    predicted_topic = predict_topic(question, model, tokenizer, le, config)
    if verbose:
        print(f"🔮 Predicted Topic: {predicted_topic}")
    
    # Step 2: Retrieve Similar Cases
    retrieved = find_most_similar_answers(
        question,
        predicted_topic,
        vectorizer,
        tfidf_matrix,
        df,
        config
    )
    
    if verbose:
        print(f"\n📚 Retrieved {len(retrieved)} Similar Cases:")
        for j, case in enumerate(retrieved, 1):
            print(f"\nCase {j} (Score: {case['score']:.3f}):")
            print(f"  Q: {case['question'][:100]}...")
            print(f"  A: {case['answer'][:100]}...")
    
    # Step 3: Generate LLM Response
    if verbose:
        print("\n🤖 Generating LLM Response...")
    
    llm_advice = generate_llm_response(predicted_topic, question, retrieved, client)
    
    if verbose:
        print(f"\n💡 LLM Advice:\n{llm_advice}")
        print(f"\n{'='*60}\n")
    
    return {
        'question': question,
        'topic': predicted_topic,
        'retrieved': retrieved,
        'advice': llm_advice
    }


def run_predefined_tests(model, tokenizer, le, vectorizer, tfidf_matrix, df, client, config):
    """Run predefined test cases."""
    
    print(f"\n{'='*60}")
    print("RUNNING PREDEFINED TEST CASES")
    print(f"{'='*60}")
    
    test_cases = [
        "I feel anxious all the time, what should I do?",
        "My partner and I are having communication issues",
        "I want to quit smoking but can't seem to stop",
        "I'm struggling to come to terms with my sexual orientation",
        "I'm experiencing burnout from my high-pressure job",
        "My child is showing signs of depression",
        "I have anger management issues that are affecting my relationships",
        "I feel disconnected from my spouse after years of marriage"
    ]
    
    results = []
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'#'*60}")
        print(f"TEST CASE {i}/{len(test_cases)}")
        print(f"{'#'*60}")
        
        result = run_single_test(
            test_input,
            model,
            tokenizer,
            le,
            vectorizer,
            tfidf_matrix,
            df,
            client,
            config,
            verbose=True
        )
        
        results.append(result)
        
        # Add delay to avoid rate limits
        if i < len(test_cases):
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"✅ Completed {len(results)} test cases")
    print(f"{'='*60}")
    
    return results


def run_interactive_mode(model, tokenizer, le, vectorizer, tfidf_matrix, df, client, config):
    """Run interactive testing mode."""
    
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print(f"{'='*60}")
    print("\nEnter patient concerns (type 'quit', 'exit', or 'q' to exit)")
    print("Type 'help' for available commands\n")
    
    while True:
        user_input = input("💬 Patient's concern: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting interactive mode.")
            break
        
        if user_input.lower() == 'help':
            print("\nAvailable commands:")
            print("  'quit', 'exit', 'q' - Exit interactive mode")
            print("  'help' - Show this help message")
            print("  Any other text - Test the RAG pipeline\n")
            continue
        
        if not user_input:
            print("⚠️ Please enter a valid concern.\n")
            continue
        
        # Run test
        run_single_test(
            user_input,
            model,
            tokenizer,
            le,
            vectorizer,
            tfidf_matrix,
            df,
            client,
            config,
            verbose=True
        )


# ========================================
# ANALYSIS FUNCTIONS
# ========================================

def analyze_retrieval_quality(model, tokenizer, le, vectorizer, tfidf_matrix, df, config):
    """Analyze the quality of retrieval for different topics."""
    
    print(f"\n{'='*60}")
    print("ANALYZING RETRIEVAL QUALITY")
    print(f"{'='*60}\n")
    
    # Sample questions per topic
    topic_samples = {
        'Anxiety & Stress': "I feel overwhelmed with stress and can't sleep",
        'Relationships': "My relationship is falling apart",
        'Family & Parenting': "My teenager won't listen to me",
        'Mental Health Disorders': "I think I might have depression",
        'Addiction & Abuse': "I can't stop drinking",
        'Identity & Spirituality': "I'm questioning my sexual identity",
        'Professional & Legal': "I'm facing workplace harassment",
        'Behavioral Changes': "I want to change my negative habits",
        'Violence & Safety': "I'm in an abusive relationship"
    }
    
    print("Testing retrieval quality for each topic:\n")
    
    for topic, sample_question in topic_samples.items():
        predicted_topic = predict_topic(sample_question, model, tokenizer, le, config)
        retrieved = find_most_similar_answers(
            sample_question,
            predicted_topic,
            vectorizer,
            tfidf_matrix,
            df,
            config
        )
        
        avg_score = np.mean([r['score'] for r in retrieved]) if retrieved else 0
        
        topic_match = "✅" if predicted_topic == topic else "❌"
        
        print(f"{topic_match} {topic}")
        print(f"   Predicted: {predicted_topic}")
        print(f"   Avg Similarity: {avg_score:.3f}")
        print(f"   Retrieved: {len(retrieved)} cases\n")


# ========================================
# MAIN FUNCTION
# ========================================

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description='Test the RAG pipeline for mental health counseling'
    )
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--test',
        '-t',
        type=str,
        help='Test a specific question'
    )
    parser.add_argument(
        '--analyze',
        '-a',
        action='store_true',
        help='Analyze retrieval quality'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/rag_model_checkpoint.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/counselchat-data.csv',
        help='Path to dataset'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Number of similar cases to retrieve'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.model_path = args.model_path
    config.data_path = args.data_path
    config.top_k = args.top_k
    
    print("="*60)
    print("🧠 LEGACY'S MENTAL HEALTH COUNSELCHAT")
    print("    RAG Pipeline Testing")
    print("="*60)
    
    # Load model and data
    model, tokenizer, le = load_model(config)
    df, vectorizer, tfidf_matrix = load_dataset(config)
    client = setup_openai()
    
    # Run based on mode
    if args.analyze:
        analyze_retrieval_quality(model, tokenizer, le, vectorizer, tfidf_matrix, df, config)
    
    elif args.test:
        run_single_test(
            args.test,
            model,
            tokenizer,
            le,
            vectorizer,
            tfidf_matrix,
            df,
            client,
            config,
            verbose=True
        )
    
    elif args.interactive:
        run_interactive_mode(
            model,
            tokenizer,
            le,
            vectorizer,
            tfidf_matrix,
            df,
            client,
            config
        )
    
    else:
        # Default: run predefined tests
        run_predefined_tests(
            model,
            tokenizer,
            le,
            vectorizer,
            tfidf_matrix,
            df,
            client,
            config
        )
    
    print("\n" + "="*60)
    print("✅ Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
