document.addEventListener('DOMContentLoaded', function () {
    // Check model status when page loads
    checkModelStatus();

    // Tab 
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.getAttribute('data-tab');

            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });


            document.querySelectorAll('.tab').forEach(t => {
                t.classList.remove('active');
            });


            tab.classList.add('active');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Form submission
    const messageForm = document.getElementById('message-form');
    if (messageForm) {
        messageForm.addEventListener('submit', function (e) {
            e.preventDefault();
            checkMessage();
        });
    }


    loadHistory();
});

function checkModelStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            const warningElement = document.getElementById('warning');
            if (warningElement) {
                if (!data.ready) {
                    warningElement.style.display = 'block';


                    let modelStatusHtml = '<strong>Model Status:</strong><br>';
                    for (const [model, status] of Object.entries(data.models)) {
                        modelStatusHtml += `${model}: ${status ? 'Loaded' : 'Not loaded'}<br>`;
                    }

                    document.getElementById('model-status').innerHTML = modelStatusHtml;
                } else {
                    warningElement.style.display = 'none';
                }
            }


            if (data.stats) {
                updateDashboardStats(data.stats);
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);

            if (document.getElementById('warning')) {
                document.getElementById('warning').style.display = 'block';
            }
        });
}

function checkMessage() {
    const text = document.getElementById('message-input').value;
    const model = document.getElementById('model-select').value;
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');

    if (!text) {
        showAlert('Please enter a message.');
        return;
    }

    loadingDiv.style.display = 'block';
    resultDiv.style.display = 'none';

    // POST Request 
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `text=${encodeURIComponent(text)}&model=${encodeURIComponent(model)}`
    })
        .then(response => response.json())
        .then(data => {
            loadingDiv.style.display = 'none';
            resultDiv.style.display = 'block';


            if (data.error) {
                resultDiv.className = 'result';
                resultDiv.innerHTML = `<strong>Error:</strong> ${data.error}<br>
            <div class="steps">
                <p>Please run the full pipeline:</p>
                <ol>
                    <li>Run <code>python run.py</code> in your terminal</li>
                    <li>Choose option 5 to run the full pipeline</li>
                </ol>
            </div>`;
                return;
            }

            const result = data.combined || data.lstm || data.tfidf;

            if (result) {
                const isSpam = result.is_spam;
                const confidence = (result.confidence * 100).toFixed(2);
                const modelName = result.model;

                resultDiv.className = `result ${isSpam ? 'spam' : 'ham'}`;
                resultDiv.innerHTML = `
                <h3>Result: ${isSpam ? 'SPAM/FAKE' : 'LEGITIMATE'}</h3>
                <div class="progress">
                    <div class="progress-bar" style="width: ${confidence}%"></div>
                </div>
                <p>Confidence: ${confidence}%</p>
                <p>Model used: ${modelName}</p>
            `;

                // Save to history
                saveToHistory(text, isSpam, confidence, modelName);

                // Update stats
                updateStats(isSpam);
            } else {
                resultDiv.className = 'result warning';
                resultDiv.innerHTML = `
                <strong>No results available.</strong> Make sure the models are properly loaded.
                <div class="steps">
                    <p>Please run the full pipeline:</p>
                    <ol>
                        <li>Run <code>python run.py</code> in your terminal</li>
                        <li>Choose option 5 to run the full pipeline</li>
                    </ol>
                </div>
            `;
            }
        })
        .catch(error => {
            loadingDiv.style.display = 'none';
            resultDiv.style.display = 'block';
            resultDiv.className = 'result warning';
            resultDiv.innerHTML = `
            <strong>Error:</strong> ${error}
            <div class="steps">
                <p>Please make sure the server is running and try again.</p>
            </div>
        `;
            console.error('Error:', error);
        });
}

function showAlert(message) {
    alert(message);
}

function saveToHistory(text, isSpam, confidence, model) {
    const history = getHistory();
    const timestamp = new Date().toISOString();

    history.unshift({
        text: text.length > 50 ? text.substring(0, 50) + '...' : text,
        isSpam,
        confidence,
        model,
        timestamp
    });

    if (history.length > 10) {
        history.pop();
    }

    // Save to localStorage
    localStorage.setItem('messageHistory', JSON.stringify(history));

    // Update history display
    loadHistory();
}

function getHistory() {
    const history = localStorage.getItem('messageHistory');
    return history ? JSON.parse(history) : [];
}

function loadHistory() {
    const historyElement = document.getElementById('history-list');
    if (!historyElement) return;

    const history = getHistory();

    if (history.length === 0) {
        historyElement.innerHTML = '<tr><td colspan="5">No history available</td></tr>';
        return;
    }

    let historyHtml = '';

    history.forEach((entry, index) => {
        const date = new Date(entry.timestamp).toLocaleString();
        historyHtml += `
            <tr>
                <td>${entry.text}</td>
                <td><span class="badge badge-${entry.isSpam ? 'spam' : 'ham'}">${entry.isSpam ? 'SPAM' : 'LEGITIMATE'}</span></td>
                <td>${entry.confidence}%</td>
                <td>${entry.model}</td>
                <td>${date}</td>
            </tr>
        `;
    });

    historyElement.innerHTML = historyHtml;
}

function updateStats(isSpam) {

    let stats = localStorage.getItem('messageStats');
    stats = stats ? JSON.parse(stats) : { total: 0, spam: 0, ham: 0 };


    stats.total++;
    if (isSpam) {
        stats.spam++;
    } else {
        stats.ham++;
    }


    localStorage.setItem('messageStats', JSON.stringify(stats));

    if (document.getElementById('total-checked')) {
        document.getElementById('total-checked').textContent = stats.total;
        document.getElementById('spam-detected').textContent = stats.spam;
        document.getElementById('ham-detected').textContent = stats.ham;

        const spamPercentage = stats.total > 0 ? (stats.spam / stats.total * 100).toFixed(1) : '0';
        const hamPercentage = stats.total > 0 ? (stats.ham / stats.total * 100).toFixed(1) : '0';

        document.getElementById('spam-percentage').textContent = spamPercentage + '%';
        document.getElementById('ham-percentage').textContent = hamPercentage + '%';
    }
}

function updateDashboardStats(stats) {
    if (document.getElementById('total-checked')) {
        document.getElementById('total-checked').textContent = stats.total || 0;
        document.getElementById('spam-detected').textContent = stats.spam || 0;
        document.getElementById('ham-detected').textContent = stats.ham || 0;


        const spamPercentage = stats.total > 0 ? (stats.spam / stats.total * 100).toFixed(1) : '0';
        const hamPercentage = stats.total > 0 ? (stats.ham / stats.total * 100).toFixed(1) : '0';

        document.getElementById('spam-percentage').textContent = spamPercentage + '%';
        document.getElementById('ham-percentage').textContent = hamPercentage + '%';
    }
}

function clearHistory() {
    if (confirm('Are you sure you want to clear your history?')) {
        localStorage.removeItem('messageHistory');
        loadHistory();
    }
}
