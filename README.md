# 🧠 Document and Website Chat Application powered by RAG

This project is a lightweight Retrieval-Augmented Generation (RAG) chat interface built with **Streamlit** for the frontend and **Python** for the backend. It integrates with the **Groq API** to provide intelligent responses to user queries.

---

## 🚀 Features

- Streamlit-based interactive chat UI
- Backend RAG pipeline with local document retrieval
- Can upload .pdf files, .txt files and Website URLs as data sources for RAG
- Can dynamically upload additional sources during an ongoing conversation
- Groq API integration for high-performance LLM responses
- Plotly support for future interactive visualizations

---

## 📦 Tech Stack

- Python 3.10+
- LangChain
- Streamlit
- Plotly
- Groq API

---

## 🛠️ Setup Instructions

### Clone the repository
```bash
git clone https://github.com/viraj200524/Document-Website-Chat.git
```
Then navigate to Document-Website-Chat
```bash
cd Document-Website-Chat
```

### 1. Create and Activate Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

----

### 🔑 Set Up Groq API Key ###

🧾 How to Get a Groq API Key:

Go to https://console.groq.com/keys

Log in with your account or sign up if you don’t have one.

Click on Create API Key

Copy the generated key.

🔐 Add to .env file:
Create a .env file in the root directory (if not already present) and paste the key like this:

```bash
GROQ_API_KEY="<YOUR API KEY>"
```
---

### Install all requirements: ###

Run the following command to Install all requirements:
```bash
pip install -r requirements.txt
```
---

### Run the application: ###
```bash
streamlit run app.py
```

