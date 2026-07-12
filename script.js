const churnForm = document.getElementById('churnForm');
const tenureInput = document.getElementById('tenure');
const tenureLabel = document.getElementById('tenureValue');
const statusText = document.getElementById('statusText');
const resultCard = document.getElementById('resultCard');
const predictionText = document.getElementById('predictionText');
const churnProbability = document.getElementById('churnProbability');
const stayProbability = document.getElementById('stayProbability');
const submitButton = churnForm.querySelector('.cta-button');

function setStatus(message, isError = false) {
    statusText.textContent = message;
    statusText.style.background = isError ? 'rgba(255, 92, 92, 0.16)' : 'rgba(255, 255, 255, 0.08)';
    statusText.style.color = isError ? '#ffb3b3' : '#dbe5ff';
}

function updateTenureLabel() {
    tenureLabel.textContent = tenureInput.value;
}

tenureInput.addEventListener('input', updateTenureLabel);
updateTenureLabel();

churnForm.addEventListener('submit', async function (event) {
    event.preventDefault();
    resultCard.classList.add('hidden');
    setStatus('Predicting...', false);
    submitButton.disabled = true;

    const payload = {
        tenure: parseInt(tenureInput.value, 10),
        monthlyCharges: parseFloat(document.getElementById('monthlyCharges').value),
        contract: document.getElementById('contract').value,
        internetService: document.getElementById('internetService').value,
        techSupport: document.getElementById('techSupport').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            setStatus('Prediction failed', true);
            throw new Error(data.error || 'Prediction API error');
        }

        displayResult(data);
    } catch (error) {
        console.error('Prediction error:', error);
        setStatus('Unable to predict, check the model backend.', true);
    } finally {
        submitButton.disabled = false;
    }
});

function displayResult(data) {
    const churnValue = Number(data.churnProbability).toFixed(2);
    const stayValue = Number(data.stayProbability).toFixed(2);

    if (Number(data.prediction) === 1) {
        predictionText.textContent = '⚠️ Customer is likely to churn';
        predictionText.style.color = '#ffb3b3';
    } else {
        predictionText.textContent = '✅ Customer is likely to stay';
        predictionText.style.color = '#b6f7c1';
    }

    churnProbability.textContent = `${churnValue}%`;
    stayProbability.textContent = `${stayValue}%`;

    resultCard.classList.remove('hidden');
    setStatus('Prediction loaded successfully');
    resultCard.scrollIntoView({ behavior: 'smooth' });
}
