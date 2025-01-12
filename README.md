# Legacy's Mental Health CounselChat

Legacy's Mental Health CounselChat is an AI-powered chatbot designed to assist mental health counselors by providing empathetic, context-aware advice and resources. The project integrates advanced natural language processing (NLP) models to classify topics, retrieve relevant information, and generate structured responses, creating a seamless experience for counselors.

---

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Topic Classification**: Utilizes RoBERTa to classify user inputs into predefined mental health categories with an accuracy of 79.71%.
- **Context-Aware Retrieval**: Implements TF-IDF similarity search to retrieve relevant knowledge from a pre-defined corpus.
- **Empathetic Advice Generation**: Leverages OpenAI GPT-3.5 to generate tailored, empathetic responses for counselors.
- **Interactive Web App**: A user-friendly interface built with Streamlit for real-time interactions.
- **Modular Design**: Easily extensible components for future scalability.

---

## Tech Stack
- **Frontend**: Streamlit
- **NLP Models**: RoBERTa, TF-IDF
- **LLM Integration**: OpenAI GPT-3.5
- **Backend**: Python
- **Deployment**: Local/Cloud-hosted with Python

---

## Architecture
```
User Input --> RoBERTa (Topic Classification) --> TF-IDF (Relevant Retrieval) --> GPT-3.5 (Response Generation) --> Response to User
```
- **RoBERTa**: Classifies user input into predefined mental health topics.
- **TF-IDF**: Searches for contextually relevant entries from the knowledge base.
- **GPT-3.5**: Enhances retrieved content with structured, empathetic advice.

---

## Installation
1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/legacy-counselchat.git
    cd legacy-counselchat
    ```

2. **Install Dependencies**
    Ensure you have Python 3.8 or higher installed.
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up API Keys**
    - Obtain an OpenAI API key and add it to your environment variables:
      ```bash
      export OPENAI_API_KEY='your_openai_api_key'
      ```

4. **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## Usage
1. Open the web app in your browser (typically at `http://localhost:8501`).
2. Enter user queries into the text box.
3. View classified topics, retrieved information, and empathetic responses in real-time.

---

## Results
- **Accuracy**: Achieved 79.71% accuracy with the RoBERTa model for topic classification.
- **Efficiency**: Near real-time processing and response generation.
- **User-Centric**: Tailored responses based on input context and topic.

---

## Future Enhancements
- **Cloud Integration**: Deploy the chatbot on cloud platforms like AWS or Google Cloud for scalability.
- **Feedback Mechanism**: Incorporate feedback loops for counselors to refine response quality.
- **Multi-Modal Support**: Expand functionality to include voice and visual input/output.
- **Expanded Knowledge Base**: Enrich the corpus with additional mental health resources.

---

## Contributing
We welcome contributions to make Legacy's Mental Health CounselChat even better! To contribute:
1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Submit a pull request with a detailed description.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- The **Legacy Team** for their vision and support.
- OpenAI for providing GPT-3.5.
- Hugging Face for the RoBERTa model.

---
