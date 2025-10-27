import streamlit as st
import openai
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import time
from openai import OpenAIError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="CounselChat", layout="wide")

# ----------------------------
# 🛠️ Load Model and Tokenizer
# ----------------------------
@st.cache_resource
def load_model():
    """Load the trained RoBERTa model, tokenizer, and label encoder."""
    try:
        checkpoint = torch.load(
            'models/rag_model_checkpoint.pth', 
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False
        )
        tokenizer = checkpoint['tokenizer']
        le = checkpoint['label_encoder']
        model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base', 
            num_labels=len(le.classes_)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.eval()
        return model, tokenizer, le
    except FileNotFoundError:
        st.error("❌ Model checkpoint not found. Please ensure 'models/rag_model_checkpoint.pth' exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, tokenizer, le = load_model()

# ----------------------------
# 📚 Load Dataset for RAG
# ----------------------------
@st.cache_data
def load_dataset():
    """Load the counseling dataset and prepare TF-IDF vectorizer."""
    try:
        df = pd.read_csv('data/counselchat-data.csv')
        # Keep both questionText and answerText, and add topic_group if available
        required_cols = ['questionText', 'answerText']
        if 'topic_group' in df.columns:
            required_cols.append('topic_group')
        
        df = df[required_cols].dropna()
        
        # Fit vectorizer on questionText (we search questions, retrieve answers)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(df['questionText'])
        
        return df, vectorizer, tfidf_matrix
    except FileNotFoundError:
        st.error("❌ Dataset not found. Please ensure 'data/counselchat-data.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.stop()

df, vectorizer, tfidf_matrix = load_dataset()

# ----------------------------
# 🤖 Prediction Functions
# ----------------------------
def predict_topic(text):
    """Predict the mental health topic category for the given text."""
    try:
        inputs = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        return le.inverse_transform([predicted_class])[0]
    except Exception as e:
        st.error(f"❌ Error in topic prediction: {e}")
        return "Unknown"


def find_most_similar_answers(text, predicted_topic, vectorizer, tfidf_matrix, df, top_k=3):
    """
    Find top-K most similar questions and retrieve their answers.
    Filters by predicted topic first for better relevance.
    
    Args:
        text: User's input question
        predicted_topic: Topic predicted by RoBERTa model
        vectorizer: Fitted TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix of all questions
        df: DataFrame containing questions and answers
        top_k: Number of similar answers to retrieve
    
    Returns:
        List of dictionaries containing question, answer, and similarity score
    """
    try:
        # Filter by topic if available
        if 'topic_group' in df.columns:
            topic_filtered_df = df[df['topic_group'] == predicted_topic].reset_index(drop=True)
            
            if len(topic_filtered_df) > 0:
                # Re-create TF-IDF matrix for filtered data
                filtered_tfidf = vectorizer.transform(topic_filtered_df['questionText'])
                working_df = topic_filtered_df
                working_tfidf = filtered_tfidf
            else:
                # Fallback to full dataset if no topic matches
                working_df = df
                working_tfidf = tfidf_matrix
        else:
            working_df = df
            working_tfidf = tfidf_matrix
        
        # Calculate similarity scores
        input_vector = vectorizer.transform([text])
        similarity_scores = cosine_similarity(input_vector, working_tfidf).flatten()
        
        # Get top K indices
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        # Retrieve answers with scores
        retrieved_answers = []
        for idx in top_indices:
            if similarity_scores[idx] > 0.1:  # Threshold filter
                retrieved_answers.append({
                    'question': working_df.iloc[idx]['questionText'],
                    'answer': working_df.iloc[idx]['answerText'],
                    'score': float(similarity_scores[idx])
                })
        
        return retrieved_answers if retrieved_answers else [{
            'question': 'No similar case found',
            'answer': 'Providing general advice based on topic.',
            'score': 0.0
        }]
    
    except Exception as e:
        st.error(f"❌ Error in similarity search: {e}")
        return [{
            'question': 'Error occurred',
            'answer': 'Unable to retrieve similar cases.',
            'score': 0.0
        }]


# ----------------------------
# 🤖 LLM Response Generation
# ----------------------------

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

client = openai.OpenAI(api_key=api_key)


def generate_llm_response(topic, user_input, retrieved_answers, retries=3):
    """Generate empathetic advice using GPT-4o mini with retrieved context."""
    
    # Format retrieved context
    context = "\n\n".join([
        f"Similar Case {i+1} (Relevance: {ans['score']:.2%}):\n"
        f"Question: {ans['question']}\n"
        f"Professional Response: {ans['answer']}"
        for i, ans in enumerate(retrieved_answers[:3])
    ])
    
    prompt = f"""
You are an empathetic mental health assistant helping counselors provide advice to their patients.

**Patient's Concern:**
"{user_input}"

**Identified Topic:** {topic}

**Reference Cases from Professional Counselors:**
{context}

**Instructions:**
- Directly address the patient's specific concern: "{user_input}"
- If the reference cases are too specific or narrow, adapt and generalize the advice appropriately
- Use the reference cases as inspiration, not as strict templates
- Be empathetic, supportive, and non-judgmental
- IMPORTANT: Use consistent markdown formatting throughout

**Please provide a structured response using this EXACT format:**

**1. Understanding the Challenge:**
[2-3 sentences acknowledging and validating the patient's feelings]

**2. Practical Advice:**
- First actionable recommendation
- Second actionable recommendation  
- Third actionable recommendation (if applicable)

**3. Emotional Support Strategies:**
- First coping strategy
- Second coping strategy
- Third coping strategy (if applicable)

**4. Final Encouragement:**
[1-2 sentences with a positive, reassuring message that inspires hope]

Keep the response conversational, warm, and within 400 tokens. Use bullet points (-) for all lists, not numbered sub-points.
"""
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an empathetic mental health assistant. You MUST use the exact markdown format provided: **1. Understanding the Challenge:** followed by paragraphs, then **2. Practical Advice:** with bullet points using -, and so on. Never deviate from this structure."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        
        except openai.RateLimitError:
            wait_time = 2 ** attempt
            if attempt < retries - 1:
                st.warning(f"⚠️ Rate limit reached. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return "⚠️ Unable to generate advice due to rate limits. Please try again in a moment."
        
        except OpenAIError as e:
            st.error(f"❌ OpenAI API Error: {e}")
            return "⚠️ Unable to generate advice at this time. Please try again later."
    
    return "⚠️ Unable to generate advice. Please try again."

# ----------------------------
# 💬 Streamlit Chat Interface
# ----------------------------
st.title("CounselChat - An AI Powered Mental Health bot")
st.write("""
**An AI-powered companion for mental health counselors.** 

Combining Machine Learning (RoBERTa topic classification) and Large Language Models (GPT-4o mini), 
this chatbot delivers empathetic, insightful, and practical advice to address a wide range of mental health concerns.
""")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This system uses:
    - **RoBERTa** for topic classification
    - **TF-IDF RAG** for retrieving similar cases
    - **GPT-4o mini** for generating personalized advice
    
    **Topics covered:**
    - Anxiety & Stress
    - Relationships
    - Family & Parenting
    - Mental Health Disorders
    - Addiction & Abuse
    - And more...
    """)
    
    st.header("⚙️ Settings")
    show_retrieved = st.checkbox("Show retrieved similar cases", value=False)

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Describe your patient's challenge here...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show processing indicator
    with st.spinner("🔍 Analyzing and generating advice..."):
        # Predict Topic
        predicted_topic = predict_topic(user_input)
        
        # Retrieve Similar Answers (Top-K with topic filtering)
        retrieved_answers = find_most_similar_answers(
            user_input, 
            predicted_topic, 
            vectorizer, 
            tfidf_matrix, 
            df, 
            top_k=3
        )
        
        # Generate LLM Response
        llm_advice = generate_llm_response(predicted_topic, user_input, retrieved_answers)
    
    # Format response
    response_content = f"**📋 Predicted Topic:** {predicted_topic}\n\n**💡 Advice:**\n\n{llm_advice}"
    
    # Optionally show retrieved cases
    if show_retrieved and retrieved_answers[0]['score'] > 0:
        response_content += "\n\n---\n**📚 Similar Cases Referenced:**\n"
        for i, ans in enumerate(retrieved_answers[:2], 1):
            response_content += f"\n*Case {i} (Similarity: {ans['score']:.2%})*\n"
            response_content += f"Q: {ans['question'][:100]}...\n"
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response_content)
    
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Action buttons
col1, col2 = st.columns([1, 5])

with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("💾 Save Chat History"):
        if st.session_state.messages:
            chat_df = pd.DataFrame(st.session_state.messages)
            chat_df.to_csv('outputs/chat_history.csv', index=False)
            st.success("✅ Chat history saved to 'outputs/chat_history.csv'")
        else:
            st.warning("⚠️ No chat history to save.")

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; font-size: 14px; color: gray;'>
    Developed by Pramoth Guhan | Powered by RoBERTa, TF-IDF RAG & GPT-4o mini | 
    <a href='https://github.com/pramothguhan'>GitHub</a>
</p>
""", unsafe_allow_html=True)