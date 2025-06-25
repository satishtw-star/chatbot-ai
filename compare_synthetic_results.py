import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("Synthetic LLM Evaluation Dashboard")

# Load the synthetic evaluation CSV log
log_file = 'synthetic_eval_log.csv'
try:
    df = pd.read_csv(log_file)
except FileNotFoundError:
    st.warning("No synthetic evaluation log found yet.")
    st.stop()

# Filter/search UI
query_filter = st.text_input("Filter by query (optional):")
if query_filter:
    df = df[df['query'].str.contains(query_filter, case=False, na=False)]

# Metrics for Model 1 only (since Model 2 is blank)
metrics = ['relevancy_1', 'faithfulness_1', 'contextual_precision_1', 
           'contextual_recall_1', 'contextual_relevancy_1', 'call_deflection_effectiveness_1', 'conversation_completeness']

# Sort by any metric
sort_metric = st.selectbox("Sort by metric:", metrics)
if st.checkbox("Sort by selected metric"):
    df = df.sort_values(sort_metric, ascending=False)

# Show comparison table
st.header("Synthetic Model Evaluation Table")
display_columns = [
    'timestamp', 'query', 'model1_response',
    'relevancy_1', 'faithfulness_1', 'contextual_precision_1',
    'contextual_recall_1', 'contextual_relevancy_1', 'call_deflection_effectiveness_1',
    'conversation_completeness'
]
st.dataframe(df[display_columns], use_container_width=True)

# Show best/worst examples
st.header("Best Responses")
best_responses = df.sort_values(sort_metric, ascending=False).head(3)
for _, row in best_responses.iterrows():
    st.markdown(f"**Query:** {row['query']}")
    st.markdown("**Model 1 Response:**")
    st.markdown(row['model1_response'])
    st.markdown("**Scores:**")
    for metric in metrics:
        st.markdown(f"- {metric.replace('_1', '').replace('_', ' ').title()}: {row[metric]:.2f}")
    st.markdown("---")

st.header("Worst Responses")
worst_responses = df.sort_values(sort_metric).head(3)
for _, row in worst_responses.iterrows():
    st.markdown(f"**Query:** {row['query']}")
    st.markdown("**Model 1 Response:**")
    st.markdown(row['model1_response'])
    st.markdown("**Scores:**")
    for metric in metrics:
        st.markdown(f"- {metric.replace('_1', '').replace('_', ' ').title()}: {row[metric]:.2f}")
    st.markdown("---") 