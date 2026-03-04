# +SumIt Up! 
### A Summarizer Accelerator Tool

A multi-engine text summarization framework that runs your text through multiple algorithms simultaneously and compares results — helping you pick the best summarization approach for your use case.

---

##  Purpose

Built to support future use cases where a summarizer is required, this tool enables evaluation and selection of the most suitable approach based on:

- ✅ Accuracy of the summary (ROUGE scores)
- ✅ Response quality
- ✅ Time taken for summarization
- ✅ Overall suitability for the specific use case

---

##  Algorithms Compared

| Algorithm | Type | Description |
|---|---|---|
| **TF-IDF (LSA)** | Extractive | Scores sentences by term frequency and relevance |
| **TextRank** | Extractive | Graph-based ranking, similar to Google's PageRank |
| **BART (Neural)** | Abstractive | Deep learning model trained on millions of articles |

---

##  Metrics Measured

- **Time taken** (milliseconds)
- **Word count**
- **ROUGE-1** — word level overlap with reference
- **ROUGE-2** — phrase level overlap with reference
- **ROUGE-L** — sentence structure overlap with reference

---

##  Tech Stack

**Backend**
- Python 3.11
- FastAPI
- Sumy (TF-IDF, TextRank)
- HuggingFace Transformers (BART)
- ROUGE Score

**Frontend**
- HTML, CSS, JavaScript
- Retro pastel design 

---

##  How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/lakshmigopinath/summarizer-accelerator
cd summarizer-accelerator
```

### 2. Set up the backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Open the frontend
Open `frontend/index.html` in your browser.

### 4. Use the tool
- Paste your text
- Optionally paste a reference summary for ROUGE scoring
- Click **▶ Run All Algorithms**
- Compare results side by side!

---

## Project Structure
```
summarizer-accelerator/
│
├── backend/
│   ├── main.py          # FastAPI server + all algorithms
│   ├── requirements.txt # Python dependencies
│
├── frontend/
│   ├── index.html       # Main webpage
│   ├── style.css        # Retro pastel styling
│   └── app.js           # API calls + UI logic
│
└── README.md
```
