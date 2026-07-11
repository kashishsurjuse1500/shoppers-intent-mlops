// This file handles form submission and API call to /predict endpoint

document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault(); // prevent default form submission

    // collect all form values into an object
    const data = {
        Administrative: parseInt(document.getElementById('Administrative').value),
        Administrative_Duration: parseFloat(document.getElementById('Administrative_Duration').value),
        Informational: parseInt(document.getElementById('Informational').value),
        Informational_Duration: parseFloat(document.getElementById('Informational_Duration').value),
        ProductRelated: parseInt(document.getElementById('ProductRelated').value),
        ProductRelated_Duration: parseFloat(document.getElementById('ProductRelated_Duration').value),
        BounceRates: parseFloat(document.getElementById('BounceRates').value),
        ExitRates: parseFloat(document.getElementById('ExitRates').value),
        PageValues: parseFloat(document.getElementById('PageValues').value),
        SpecialDay: parseFloat(document.getElementById('SpecialDay').value),
        Month: parseInt(document.getElementById('Month').value),
        OperatingSystems: parseInt(document.getElementById('OperatingSystems').value),
        Browser: parseInt(document.getElementById('Browser').value),
        Region: parseInt(document.getElementById('Region').value),
        TrafficType: parseInt(document.getElementById('TrafficType').value),
        VisitorType: parseInt(document.getElementById('VisitorType').value),
        Weekend: parseInt(document.getElementById('Weekend').value),
    };

    const resultDiv = document.getElementById('result'); // get result div
    resultDiv.className = 'result';                      // reset classes
    resultDiv.innerHTML = '⏳ Predicting...';            // show loading text
    resultDiv.classList.remove('hidden');                // show result div

    try {
        // call predict API endpoint
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)  // send data as json
        });

        const result = await response.json(); // parse response

        // display result based on prediction
        if (result.prediction === 1) {
            resultDiv.classList.add('purchase');  // green style
            resultDiv.innerHTML = `
                ✅ ${result.message}
                <small>Confidence: ${(result.probability * 100).toFixed(1)}%</small>
            `;
        } else {
            resultDiv.classList.add('no-purchase');  // red style
            resultDiv.innerHTML = `
                ❌ ${result.message}
                <small>Confidence: ${((1 - result.probability) * 100).toFixed(1)}%</small>
            `;
        }

    } catch (error) {
        resultDiv.classList.add('no-purchase');        // red style for error
        resultDiv.innerHTML = `⚠️ Error: ${error.message}`;  // show error message
    }
});