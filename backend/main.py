from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from rouge_score import rouge_scorer

# Sumy imports for TF-IDF
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str
    reference: str = ""

@app.get("/")
def home():
    return {"message": "Summarizer Accelerator API is running!"}

@app.post("/summarize/tfidf")
def summarize_tfidf(input: TextInput):
    start = time.time()

    parser = PlaintextParser.from_string(input.text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")

    sentences = summarizer(parser.document, 3)
    summary = " ".join(str(s) for s in sentences)

    end = time.time()
    time_taken = round((end - start) * 1000)

    return {
        "algorithm": "TF-IDF (LSA)",
        "summary": summary,
        "time_ms": time_taken,
        "word_count": len(summary.split())
    }
    
@app.post("/summarize/textrank")
def summarize_textrank(input: TextInput):
    start = time.time()

    parser = PlaintextParser.from_string(input.text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")

    sentences = summarizer(parser.document, 3)
    summary = " ".join(str(s) for s in sentences)

    end = time.time()
    time_taken = round((end - start) * 1000)

    return {
        "algorithm": "TextRank",
        "summary": summary,
        "time_ms": time_taken,
        "word_count": len(summary.split())
    }
    
from transformers import pipeline

# Load BART model once when server starts (not on every request)
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.post("/summarize/bart")
def summarize_bart(input: TextInput):
    start = time.time()

    result = bart_summarizer(
        input.text,
        max_length=130,
        min_length=30,
        do_sample=False
    )
    summary = result[0]["summary_text"]

    end = time.time()
    time_taken = round((end - start) * 1000)

    return {
        "algorithm": "BART (Neural)",
        "summary": summary,
        "time_ms": time_taken,
        "word_count": len(summary.split())
    }
    
@app.post("/summarize/all")
def summarize_all(input: TextInput):
    tfidf = summarize_tfidf(input)
    textrank = summarize_textrank(input)
    bart = summarize_bart(input)

    # Calculate ROUGE scores if reference is provided
    if input.reference.strip():
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for result in [tfidf, textrank, bart]:
            scores = scorer.score(input.reference, result['summary'])
            result['rouge1'] = round(scores['rouge1'].fmeasure, 3)
            result['rouge2'] = round(scores['rouge2'].fmeasure, 3)
            result['rougeL'] = round(scores['rougeL'].fmeasure, 3)

    return {
        "results": [tfidf, textrank, bart]
    }