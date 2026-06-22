import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. Setup Paths 
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model.joblib"
DATA_PATH = SCRIPT_DIR / "sample_data.csv"

# ==========================================
# 2. Load the Model and Data
# ==========================================
model_data = joblib.load(MODEL_PATH)
pipe = model_data["pipeline"]
df_sample = pd.read_csv(DATA_PATH)

average_spend = df_sample['total_spend'].mean()

# ==========================================
# 3. Inference & Plotting Function
# ==========================================
def predict_spend(income, kidhome, teenhome, recency, web_visits):
    input_df = pd.DataFrame({
        'Income': [income],
        'Kidhome': [kidhome],
        'Teenhome': [teenhome],
        'Recency': [recency],
        'NumWebVisitsMonth': [web_visits]
    })
    
    prediction = pipe.predict(input_df)[0]
    
    if prediction <= 0:
        final_pred = 0
        text_result = (
            "### Predicted Total Spend: **$0.00**\n\n"
            "⚠️ **Note on this Prediction:**\n"
            "The model predicted a negative value for this customer, which has been capped at $0.00. "
            "Linear Regression models draw a straight line through the data and can sometimes extrapolate below zero. "
            "In real-world marketing analytics, this usually happens when a customer has characteristics that heavily "
            "discourage high spending according to the dataset (e.g., a combination of low income, multiple children at home, "
            "or low engagement)."
        )
    else:
        final_pred = prediction
        text_result = f"### Predicted Total Spend: **${final_pred:.2f}**"
    
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['Average Customer', 'Predicted Customer']
    values = [average_spend, final_pred]
    colors = ['#cccccc', '#1f77b4'] 
    
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel('Total Spend ($)')
    ax.set_title('Customer Spend Comparison')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, f'${yval:.0f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    
    return text_result, fig


# ==========================================
# 4. Build the User Interface (UI)
# ==========================================
with gr.Blocks() as app:
    gr.Markdown("# 🛒 Marketing Analytics: Customer Spend Predictor")
    gr.Markdown("This application uses **Linear Regression** to predict the total amount a customer is likely to spend based on their demographics and web behavior.")
    
    with gr.Tabs():
        with gr.TabItem("🔮 Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Enter Customer Details")
                    income_input = gr.Number(label="Annual Income ($)", value=50000)
                    kidhome_input = gr.Slider(minimum=0, maximum=5, step=1, label="Kids at home (Kidhome)", value=0)
                    teenhome_input = gr.Slider(minimum=0, maximum=5, step=1, label="Teens at home (Teenhome)", value=0)
                    recency_input = gr.Slider(minimum=0, maximum=100, step=1, label="Days since last purchase (Recency)", value=30)
                    web_visits_input = gr.Slider(minimum=0, maximum=30, step=1, label="Monthly Web Visits", value=5)
                    
                    predict_btn = gr.Button("Generate Prediction", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Prediction Results")
                    output_text = gr.Markdown() 
                    output_plot = gr.Plot(label="Visual Comparison")
            
            predict_btn.click(
                fn=predict_spend, 
                inputs=[income_input, kidhome_input, teenhome_input, recency_input, web_visits_input], 
                outputs=[output_text, output_plot]
            )
            
        with gr.TabItem("📊 Sample Data"):
            gr.Markdown("### Training Data Sample")
            gr.Markdown("A small sample (200 rows) of the original dataset to provide context.")
            gr.Dataframe(df_sample)
            
        with gr.TabItem("🤖 Model Info"):
            gr.Markdown("""
            ### About this Model
            * **Algorithm:** Linear Regression (with StandardScaler)
            * **Objective:** Predict the continuous variable `total_spend` (aggregation of meat, fish, wines, fruits, and sweets purchases).
            * **Preprocessing:** Missing values and extreme outliers (Income > $200k) were removed during training.
            * **Deployment:** Built with Gradio and hosted on Hugging Face Spaces.
            """)

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())