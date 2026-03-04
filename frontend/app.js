
// Update word count as user types
document.getElementById('inputText').addEventListener('input', function() {
    const text = this.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    document.getElementById('wordCount').textContent = words + ' words';
});

// Set a card to loading state
function setLoading(cardId) {
    const card = document.getElementById(cardId);
    card.querySelector('.card-body').innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            Generating summary...
        </div>
    `;
}

// Display result in a card
function setResult(cardId, data) {
    const rouge = data.rouge1 !== undefined ? `
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">${data.rouge1}</div>
                <div class="metric-label">ROUGE-1</div>
            </div>
            <div class="metric">
                <div class="metric-value">${data.rouge2}</div>
                <div class="metric-label">ROUGE-2</div>
            </div>
            <div class="metric">
                <div class="metric-value">${data.rougeL}</div>
                <div class="metric-label">ROUGE-L</div>
            </div>
            <div class="metric">
                <div class="metric-value">${data.time_ms}ms</div>
                <div class="metric-label">Time Taken</div>
            </div>
        </div>` : `
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">${data.time_ms}ms</div>
                <div class="metric-label">Time Taken</div>
            </div>
            <div class="metric">
                <div class="metric-value">${data.word_count}</div>
                <div class="metric-label">Word Count</div>
            </div>
        </div>`;

    const card = document.getElementById(cardId);
    card.querySelector('.card-body').innerHTML = `
        <p class="summary-text">${data.summary}</p>
        ${rouge}
    `;
}

// Set error in a card
function setError(cardId) {
    const card = document.getElementById(cardId);
    card.querySelector('.card-body').innerHTML = `
        <div class="placeholder">Error generating summary.</div>
    `;
}

// Main function — calls the backend
async function runAll() {
    const text = document.getElementById('inputText').value.trim();

    if (!text) {
        alert('Please enter some text first!');
        return;
    }

    // Disable button and show loading on all cards
    const btn = document.getElementById('summarizeBtn');
    btn.disabled = true;
    btn.textContent = 'Running...';

    setLoading('card-tfidf');
    setLoading('card-textrank');
    setLoading('card-bart');

    try {
        const response = await fetch('http://127.0.0.1:8000/summarize/all', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                text: text,
                reference: document.getElementById('referenceText').value.trim()
            })
        });

        const data = await response.json();

        // Display each result
        data.results.forEach(result => {
            if (result.algorithm === 'TF-IDF (LSA)') setResult('card-tfidf', result);
            if (result.algorithm === 'TextRank') setResult('card-textrank', result);
            if (result.algorithm === 'BART (Neural)') setResult('card-bart', result);
        });

    } catch (error) {
        setError('card-tfidf');
        setError('card-textrank');
        setError('card-bart');
    }

    // Re-enable button
    btn.disabled = false;
    btn.textContent = '▶ Run All Algorithms';
}