import pandas as pd
import joblib
from flask import Flask, request, render_template_string
import plotly.express as px
import plotly.graph_objects as go

# Load the trained ML model and data
MODEL_PATH = "trained_model.pkl"
DATA_FILE = "complete_filtered_genes.csv"
model = joblib.load(MODEL_PATH)
gene_data = pd.read_csv(DATA_FILE)

# Extract features used in training
TRAIN_FEATURES = ['CADD_Score_normalized', 'Log_FC_normalized',
                  'MGI_Phenotpe_normalized', 'Phen2Gene_Score_normalized',
                  'EP_Asso_normalized']

# Initialize Flask app
app = Flask(__name__)

# HTML Template
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <title>Gene Association Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f9; margin: 20px; }
        .container { max-width: 900px; margin: auto; }
        h1 { text-align: center; margin-top: 20px; color: #2c3e50; }
        .result, .error { margin-top: 20px; padding: 15px; border-radius: 8px; }
        .result { background-color: #e8f5e9; color: #2e7d32; }
        .error { background-color: #ffebee; color: #c62828; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gene Association Prediction for Epilepsy</h1>
        <form method="POST" class="mb-4">
            <div class="input-group mb-3">
                <input type="text" name="gene" class="form-control" placeholder="Enter Gene Name (e.g., SCN2A)" required>
                <button class="btn btn-primary" type="submit">Predict</button>
            </div>
        </form>

        {% if result %}
        <div class="result">
            <h3>Prediction Result</h3>
            <p><strong>Gene:</strong> {{ gene }}</p>
            <p><strong>Weighted Score:</strong> {{ score }}</p>
            <p><strong>Prediction:</strong> {{ prediction }}</p>
        </div>

        <div id="graph" style="margin-top: 30px;">
            <h4>Gene Feature Representation</h4>
            <div id="feature_plot"></div>
        </div>
        {% endif %}

        {% if error %}
        <div class="error">
            <h3>Error:</h3>
            <p>{{ error }}</p>
        </div>
        {% endif %}
    </div>
    <script>
        var graphJSON = {{ plot | safe }};
        Plotly.newPlot('feature_plot', graphJSON);
    </script>
</body>
</html>
"""

# Flask route for prediction
@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    error = None
    gene = None
    score = None
    prediction = None
    plot = None

    if request.method == 'POST':
        gene = request.form['gene'].strip().upper()

        try:
            # Find the gene in the dataset
            gene_row = gene_data[gene_data['Gene'] == gene]
            if gene_row.empty:
                score = 0.0
                prediction = 'Not Associated with EP'
                result = True
                plot = generate_bar_plot({}, gene, 0.0)
            else:
                # Extract only training features
                features = gene_row[TRAIN_FEATURES].values[0]
                score = float(gene_row['Weighted_Score'].iloc[0])

                # Predict using the model
                pred = model.predict([features])
                prediction = 'Associated with EP' if pred[0] == 1 else 'Not Associated with EP'
                result = True

                # Generate plot
                clean_feature_dict = {feature.replace('_normalized', ''): features[i]
                                      for i, feature in enumerate(TRAIN_FEATURES)}
                plot = generate_bar_plot(clean_feature_dict, gene, score)

        except Exception as e:
            error = f"An error occurred: {e}"

    return render_template_string(HTML_TEMPLATE, gene=gene, score=score, prediction=prediction, 
                                  result=result, error=error, plot=plot)

# Function to generate Plotly graph
def generate_bar_plot(features, gene_name, score):
    if not features:
        return go.Figure().to_json()

    fig = px.bar(
        x=list(features.keys()),
        y=list(features.values()),
        title=f"Feature Scores for {gene_name} (Weighted Score: {score})",
        labels={"x": "Features", "y": "Score"},
        color=list(features.values()),
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=400, width=800)
    return fig.to_json()

# Run Flask app
if __name__ == '__main__':
    print("Running Gene Prediction Web App on http://127.0.0.1:5000")
    app.run(debug=True)
