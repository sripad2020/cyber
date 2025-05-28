import cirq
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier,XGBRFClassifier
from lazypredict.Supervised import LazyClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, average_precision_score,
                            confusion_matrix, precision_recall_curve, roc_curve,
                            classification_report)
import shap


# Load and preprocess data
data = pd.read_csv('The_ESM.csv')
X = data.drop(['target', 'Epitope'], axis=1)
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_qubits = 4
chunk_size = X_scaled.shape[1] // n_qubits
def chunk_features(X, n_qubits):
    return np.array([
        [np.mean(x[i*chunk_size:(i+1)*chunk_size]) for i in range(n_qubits)]
        for x in X
    ])
X_reduced = chunk_features(X_scaled, n_qubits)

# Cirq setup
qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
simulator = cirq.Simulator()

def expectation_Z_from_state(state_vector, qubit_index, n_qubits):
    exp_val = 0.0
    for i, amplitude in enumerate(state_vector):
        prob = np.abs(amplitude)**2
        bit = (i >> (n_qubits - qubit_index - 1)) & 1  # extract bit of qubit_index
        z_val = 1 if bit == 0 else -1
        exp_val += z_val * prob
    return exp_val

def quantum_circuit_encoding(x):
    circuit = cirq.Circuit()
    # Angle encoding
    for i in range(n_qubits):
        circuit.append(cirq.ry(x[i]).on(qubits[i]))
    # CNOT entanglement
    for i in range(n_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    result = simulator.simulate(circuit)
    state_vector = result.final_state_vector
    # Manually compute Z expectation for each qubit
    return [expectation_Z_from_state(state_vector, i, n_qubits) for i in range(n_qubits)]

# Quantum transformation
X_q = np.array([quantum_circuit_encoding(x) for x in X_reduced])

# Classical classification
X_train, X_test, y_train, y_test = train_test_split(X_q, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "XGBoost RF": XGBRFClassifier(),
    "LightGBM": LGBMClassifier(),
    "Extra Trees": ExtraTreesClassifier()
}

# Train models and store predictions
results = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)

    predictions[name] = y_pred
    probabilities[name] = y_proba

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba),
        "PR AUC": average_precision_score(y_test, y_proba)
    }

# Convert results to DataFrame for visualization
results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

## Visualization 1: Metrics Comparison Bar Chart
fig1 = px.bar(results_df,
              x="Model",
              y=["Accuracy", "Precision", "Recall", "F1 Score"],
              barmode="group",
              title="Model Performance Comparison",
              labels={"value": "Score", "variable": "Metric"},
              color_discrete_sequence=px.colors.qualitative.Plotly)
fig1.update_layout(yaxis_range=[0, 1])
fig1.show()

## Visualization 2: ROC AUC Comparison Bar Chart
fig2 = px.bar(results_df,
              x="Model",
              y="ROC AUC",
              title="ROC AUC Score Comparison",
              labels={"ROC AUC": "ROC AUC Score"},
              color="Model",
              color_discrete_sequence=px.colors.qualitative.Plotly)
fig2.update_layout(yaxis_range=[0, 1])
fig2.show()

## Visualization 3: PR AUC Comparison Bar Chart
fig3 = px.bar(results_df,
              x="Model",
              y="PR AUC",
              title="PR AUC Score Comparison",
              labels={"PR AUC": "PR AUC Score"},
              color="Model",
              color_discrete_sequence=px.colors.qualitative.Plotly)
fig3.update_layout(yaxis_range=[0, 1])
fig3.show()

## Visualization 4: Confusion Matrices Grid
fig4 = make_subplots(rows=2, cols=3,
                     subplot_titles=[f"Confusion Matrix: {model}" for model in models.keys()])

for i, (name, model) in enumerate(models.items()):
    cm = confusion_matrix(y_test, predictions[name])
    row = (i // 3) + 1
    col = (i % 3) + 1

    fig4.add_trace(
        go.Heatmap(
            z=cm,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale="Blues",
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            hoverinfo="text"
        ),
        row=row, col=col
    )

fig4.update_layout(title_text="Confusion Matrices Comparison", height=600, width=900)
fig4.show()

## Visualization 5: ROC Curves Comparison
fig5 = go.Figure()

for name, model in models.items():
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc_score = roc_auc_score(y_test, probabilities[name])

    fig5.add_trace(
        go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{name} (AUC = {auc_score:.2f})",
            line=dict(width=2)
        )
    )

fig5.add_trace(
    go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random (AUC = 0.5)",
        line=dict(dash="dash", color="gray")
    )
)

fig5.update_layout(
    title="ROC Curves Comparison",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain="domain"),
    width=700, height=600
)
fig5.show()

## Visualization 6: Precision-Recall Curves Comparison
fig6 = go.Figure()

for name, model in models.items():
    precision, recall, _ = precision_recall_curve(y_test, probabilities[name])
    pr_auc = average_precision_score(y_test, probabilities[name])

    fig6.add_trace(
        go.Scatter(
            x=recall, y=precision,
            mode="lines",
            name=f"{name} (AUC = {pr_auc:.2f})",
            line=dict(width=2)
        )
    )

fig6.update_layout(
    title="Precision-Recall Curves Comparison",
    xaxis_title="Recall",
    yaxis_title="Precision",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain="domain"),
    width=700, height=600
)
fig6.show()

## Visualization 7: Feature Importance Comparison
fig7 = make_subplots(rows=2, cols=3,
                     subplot_titles=[f"Feature Importance: {model}" for model in models.keys()])

for i, (name, model) in enumerate(models.items()):
    try:
        # Different models have different ways to get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.zeros(X_train.shape[1])

        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in
                                                                             range(X_train.shape[1])]
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        importance_df = importance_df.sort_values("Importance", ascending=False).head(10)

        row = (i // 3) + 1
        col = (i % 3) + 1

        fig7.add_trace(
            go.Bar(
                x=importance_df["Importance"],
                y=importance_df["Feature"],
                orientation="h",
                name=name,
                marker_color=px.colors.qualitative.Plotly[i]
            ),
            row=row, col=col
        )
    except:
        continue

fig7.update_layout(title_text="Feature Importance Comparison", height=600, width=900)
fig7.show()

## Visualization 8: SHAP Summary Plot (for one model as example)
# Initialize SHAP explainer for one model (Random Forest in this case)
explainer = shap.TreeExplainer(models["Random Forest"])
shap_values = explainer.shap_values(X_test)

fig8 = go.Figure()

# Create beeswarm plot manually
for i in range(len(shap_values[0][0])):
    fig8.add_trace(
        go.Scatter(
            x=shap_values[0][:, i],
            y=np.random.rand(len(shap_values[0])) * 2 - 1,  # Jitter for visibility
            mode="markers",
            marker=dict(size=5, opacity=0.5),
            name=f"Feature {i}",
            hovertext=X_test.columns[i] if hasattr(X_test, 'columns') else f"Feature {i}"
        )
    )

fig8.update_layout(
    title="SHAP Values Summary (Random Forest)",
    xaxis_title="SHAP Value",
    yaxis_title="",
    showlegend=False,
    height=600
)
fig8.show()

## Visualization 9: Model Metrics Radar Chart
fig9 = go.Figure()

for i, row in results_df.iterrows():
    fig9.add_trace(
        go.Scatterpolar(
            r=[row["Accuracy"], row["Precision"], row["Recall"], row["F1 Score"], row["ROC AUC"], row["PR AUC"]],
            theta=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"],
            fill="toself",
            name=row["Model"]
        )
    )

fig9.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title="Model Metrics Radar Chart"
)
fig9.show()

## Visualization 10: Prediction Probability Distribution
fig10 = make_subplots(rows=2, cols=3,
                      subplot_titles=[f"Probability Distribution: {model}" for model in models.keys()])

for i, (name, model) in enumerate(models.items()):
    row = (i // 3) + 1
    col = (i % 3) + 1

    fig10.add_trace(
        go.Violin(
            x=y_test,
            y=probabilities[name],
            box_visible=True,
            points="all",
            name=name,
            jitter=0.1,
            pointpos=-1.8
        ),
        row=row, col=col
    )

fig10.update_layout(title_text="Prediction Probability Distribution by Class", height=600, width=900)
fig10.show()

## Visualization 11: Cumulative Gain Chart
fig11 = go.Figure()

for name, model in models.items():
    y_proba = probabilities[name]
    df = pd.DataFrame({"prob": y_proba, "y": y_test})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    df["cum_y"] = df["y"].cumsum()
    df["pct_population"] = (df.index + 1) / len(df)
    df["pct_outcome"] = df["cum_y"] / df["y"].sum()

    fig11.add_trace(
        go.Scatter(
            x=df["pct_population"],
            y=df["pct_outcome"],
            mode="lines",
            name=name
        )
    )

fig11.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(dash="dash", color="gray")
    )
)

fig11.update_layout(
    title="Cumulative Gain Chart",
    xaxis_title="Percentage of Population",
    yaxis_title="Percentage of Positive Outcomes",
    width=700, height=600
)
fig11.show()

## Visualization 12: Lift Chart
fig12 = go.Figure()

for name, model in models.items():
    y_proba = probabilities[name]
    df = pd.DataFrame({"prob": y_proba, "y": y_test})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)

    # Create 10 equal bins
    df["bin"] = pd.qcut(df["prob"], q=10, labels=False, duplicates="drop")
    lift_df = df.groupby("bin").agg({"y": ["mean", "count"]})
    lift_df.columns = ["response_rate", "count"]
    lift_df["lift"] = lift_df["response_rate"] / df["y"].mean()

    fig12.add_trace(
        go.Bar(
            x=lift_df.index,
            y=lift_df["lift"],
            name=name,
            opacity=0.7
        )
    )

fig12.update_layout(
    title="Lift Chart (Decile Analysis)",
    xaxis_title="Decile",
    yaxis_title="Lift",
    barmode="group",
    width=700, height=600
)
fig12.show()

## Visualization 13: Calibration Curves
from sklearn.calibration import calibration_curve

fig13 = go.Figure()

for name, model in models.items():
    prob_true, prob_pred = calibration_curve(y_test, probabilities[name], n_bins=10)

    fig13.add_trace(
        go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode="lines+markers",
            name=name
        )
    )

fig13.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfectly Calibrated",
        line=dict(dash="dash", color="gray")
    )
)

fig13.update_layout(
    title="Calibration Curves",
    xaxis_title="Mean Predicted Probability",
    yaxis_title="Fraction of Positives",
    width=700, height=600
)
fig13.show()

## Visualization 14: Decision Boundary (for 2D projection)
# Note: This works best if you select 2 important features
from sklearn.decomposition import PCA

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Retrain models on PCA-transformed data
pca_models = {}
pca_predictions = {}

for name, model in models.items():
    # Create new instance to avoid overwriting original models
    new_model = type(model)()
    new_model.fit(X_pca, y_train)
    pca_models[name] = new_model
    pca_predictions[name] = new_model.predict(pca.transform(X_test))

# Create grid for decision boundary
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

fig14 = make_subplots(rows=2, cols=3,
                      subplot_titles=[f"Decision Boundary: {model}" for model in models.keys()])

for i, (name, model) in enumerate(pca_models.items()):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    row = (i // 3) + 1
    col = (i % 3) + 1

    # Decision boundary
    fig14.add_trace(
        go.Contour(
            x=np.arange(x_min, x_max, 0.1),
            y=np.arange(y_min, y_max, 0.1),
            z=Z,
            colorscale=["blue", "red"],
            showscale=False,
            opacity=0.3
        ),
        row=row, col=col
    )

    # Actual points
    fig14.add_trace(
        go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode="markers",
            marker=dict(color=y_train, colorscale=["blue", "red"]),
            showlegend=False
        ),
        row=row, col=col
    )

fig14.update_layout(title_text="Decision Boundaries (PCA Projection)", height=600, width=900)
fig14.show()

## Visualization 15: SHAP Dependence Plot (for one model and one feature)
# Select the most important feature for the Random Forest model
feature_importances = models["Random Forest"].feature_importances_
top_feature_idx = np.argmax(feature_importances)
top_feature_name = X_train.columns[top_feature_idx] if hasattr(X_train, 'columns') else f"Feature {top_feature_idx}"

fig15 = go.Figure()

fig15.add_trace(
    go.Scatter(
        x=X_test.iloc[:, top_feature_idx] if hasattr(X_test, 'iloc') else X_test[:, top_feature_idx],
        y=shap_values[0][:, top_feature_idx],
        mode="markers",
        marker=dict(
            color=y_test,
            colorscale=["blue", "red"],
            showscale=True
        ),
        name="SHAP Value"
    )
)

fig15.update_layout(
    title=f"SHAP Dependence Plot: {top_feature_name} (Random Forest)",
    xaxis_title=top_feature_name,
    yaxis_title="SHAP Value",
    width=700, height=600
)

fig15.show()
lazy=LazyClassifier()
models,prediction=lazy.fit(X_train,X_test,y_train,y_test)
print(models)