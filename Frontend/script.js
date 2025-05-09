const allSymptoms = ['itching', 'skin rash', 'nodal skin eruptions',
    'continuous sneezing', 'shivering', 'chills', 'joint pain',
    'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting',
    'vomiting', 'burning micturition', 'spotting urination', 'fatigue',
    'weight gain', 'anxiety', 'cold hands and feets', 'mood swings',
    'weight loss', 'restlessness', 'lethargy', 'patches in throat',
    'irregular sugar level', 'cough', 'high fever', 'sunken eyes',
    'breathlessness', 'sweating', 'dehydration', 'indigestion',
    'headache', 'yellowish skin', 'dark urine', 'nausea',
    'loss of appetite', 'pain behind the eyes', 'back pain',
    'constipation', 'abdominal pain', 'diarrhoea', 'mild fever',
    'yellow urine', 'yellowing of eyes', 'acute liver failure',
    'fluid overload', 'swelling of stomach', 'swelled lymph nodes',
    'malaise', 'blurred and distorted vision', 'phlegm',
    'throat irritation', 'redness of eyes', 'sinus pressure',
    'runny nose', 'congestion', 'chest pain', 'weakness in limbs',
    'fast heart rate', 'pain during bowel movements',
    'pain in anal region', 'bloody stool', 'irritation in anus',
    'neck pain', 'dizziness', 'cramps', 'bruising', 'obesity',
    'swollen legs', 'swollen blood vessels', 'puffy face and eyes',
    'enlarged thyroid', 'brittle nails', 'swollen extremeties',
    'excessive hunger', 'extra marital contacts',
    'drying and tingling lips', 'slurred speech', 'knee pain',
    'hip joint pain', 'muscle weakness', 'stiff neck',
    'swelling joints', 'movement stiffness', 'spinning movements',
    'loss of balance', 'unsteadiness', 'weakness of one body side',
    'loss of smell', 'bladder discomfort', 'foul smell ofurine',
    'continuous feel of urine', 'passage of gases', 'internal itching',
    'toxic look (typhos)', 'depression', 'irritability', 'muscle pain',
    'altered sensorium', 'red spots over body', 'belly pain',
    'abnormal menstruation', 'dischromic patches',
    'watering from eyes', 'increased appetite', 'polyuria',
    'family history', 'mucoid sputum', 'rusty sputum',
    'lack of concentration', 'visual disturbances',
    'receiving blood transfusion', 'receiving unsterile injections',
    'coma', 'stomach bleeding', 'distention of abdomen',
    'history of alcohol consumption', 'blood in sputum',
    'prominent veins on calf', 'palpitations', 'painful walking',
    'pus filled pimples', 'blackheads', 'scurring', 'skin peeling',
    'silver like dusting', 'small dents in nails',
    'inflammatory nails', 'blister', 'red sore around nose',
    'yellow crust ooze', 'prognosis'];

let selectedSymptoms = [];

const input = document.getElementById('symptomInput');
const autocompleteList = document.getElementById('autocomplete-list');
const selectedSymptomsContainer = document.getElementById('selected-symptoms');

//show Selected Symptom
function renderSelectedSymptoms(){
    selectedSymptomsContainer.innerHTML = '';
    selectedSymptoms.forEach(symptom => {
        const tag = document.createElement('span');
        tag.className = 'symptom.tag';
        tag.textContent = symptom;
        selectedSymptomsContainer.appendChild(tag);
    });
}

//Listen to input changes
input.addEventListener('input',function(){
    const query = this.value.toLowerCase();
    autocompleteList.innerHTML = '';

    if(!query) return;

    const matches = allSymptoms.filter(symptom => symptom.includes(query) && !selectedSymptoms.includes(symptom));

    matches.forEach(symptom => {
        const div = document.createElement('div');
        div.textContent = symptom;
        div.addEventListener('click',() => {
            selectedSymptoms.push(symptom);
            input.value='';
            autocompleteList.innerHTML = '';
            renderSelectedSymptoms();
        });
        autocompleteList.appendChild(div);
    });
});



//Send to backend
function sendSymptoms(){

   fetch('https://smart-disease-prediction-system.onrender.com/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: selectedSymptoms })
    })
    .then(response => response.json())
    .then(data => {
        const resultContainer = document.getElementById('result');
        resultContainer.innerHTML = '';

        data.forEach(prediction => {
            const { disease, confidence, description, precautions } = prediction;

            const diseaseCard = document.createElement('div');
            diseaseCard.className = 'disease-card';

            diseaseCard.innerHTML = `
                <h3> Predicted Disease: ${disease} (${confidence}%)</h3>
                <p><strong>🧾 Description:</strong> ${description}</p>
                <p><strong>✅ Home Precautions:</strong></p>
                <ul>
                    ${precautions
                        .filter(p => p && p.toLowerCase() !== 'nan') // clean up
                        .map(p => `<li>- ${p}</li>`).join('')}
                </ul>
                <hr/>
            `;

            resultContainer.appendChild(diseaseCard);
        });
    })
    .catch(err => {
        console.error(err);
        document.getElementById('result').innerText='Error occured.';
    });
}
