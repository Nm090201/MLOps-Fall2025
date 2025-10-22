document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const resultDiv = document.getElementById('result');
    
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    resultDiv.className = 'show';
    resultDiv.textContent = `Quality Score: ${data.predicted_quality} / 10`;
});