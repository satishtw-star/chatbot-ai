import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("LLM Comparison Dashboard")

# Load the CSV log
log_file = 'chat_eval_log.csv'
try:
    df = pd.read_csv(log_file)
except FileNotFoundError:
    st.warning("No chat evaluation log found yet.")
    st.stop()

# Filter/search UI
query_filter = st.text_input("Filter by query (optional):")
if query_filter:
    df = df[df['query'].str.contains(query_filter, case=False, na=False)]

# Extract model responses from the combined response
def split_responses(response):
    if "Model 1" in response and "Model 2" in response:
        parts = response.split("Model 2")
        model1_response = parts[0].replace("Model 1", "").strip()
        model2_response = parts[1].strip()
        return model1_response, model2_response
    return response, response

# Split responses into separate columns
df[['model1_response', 'model2_response']] = df.apply(
    lambda row: pd.Series(split_responses(row['response'])), axis=1
)

# Calculate average scores for each model
metrics = ['relevancy', 'faithfulness', 'contextual_precision', 
           'contextual_recall', 'contextual_relevancy', 'conversation_completeness']

# Sort by any metric
sort_metric = st.selectbox("Sort by metric:", metrics)
if st.checkbox("Sort by selected metric"):
    df = df.sort_values(sort_metric, ascending=False)

# Show comparison table
st.header("Model Comparison")
display_columns = [
    'timestamp', 'query', 
    'model1_response', 'model2_response'
] + metrics
st.dataframe(df[display_columns], use_container_width=True)

# Visualize metric distributions
st.header("Metric Distributions")
for metric in metrics:
    st.subheader(metric.replace('_', ' ').title())
    st.bar_chart(df[metric])

# Show best/worst examples
st.header("Best Responses")
best_responses = df.sort_values(sort_metric, ascending=False).head(3)
for _, row in best_responses.iterrows():
    st.markdown(f"**Query:** {row['query']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model 1 Response:**")
        st.markdown(row['model1_response'])
    with col2:
        st.markdown("**Model 2 Response:**")
        st.markdown(row['model2_response'])
    st.markdown("**Scores:**")
    for metric in metrics:
        st.markdown(f"- {metric.replace('_', ' ').title()}: {row[metric]:.2f}")
    st.markdown("---")

st.header("Worst Responses")
worst_responses = df.sort_values(sort_metric).head(3)
for _, row in worst_responses.iterrows():
    st.markdown(f"**Query:** {row['query']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model 1 Response:**")
        st.markdown(row['model1_response'])
    with col2:
        st.markdown("**Model 2 Response:**")
        st.markdown(row['model2_response'])
    st.markdown("**Scores:**")
    for metric in metrics:
        st.markdown(f"- {metric.replace('_', ' ').title()}: {row[metric]:.2f}")
    st.markdown("---") 