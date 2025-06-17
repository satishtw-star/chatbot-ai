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

# Calculate average scores for each model
metrics = ['relevancy', 'faithfulness', 'contextual_precision', 
           'contextual_recall', 'contextual_relevancy', 'conversation_completeness', 'call_deflection_effectiveness']

# Sort by any metric
sort_metric = st.selectbox("Sort by metric:", metrics)
if st.checkbox("Sort by selected metric"):
    df = df.sort_values(sort_metric, ascending=False)

# Show comparison table
st.header("Model Comparison")
display_columns = [
    'timestamp', 'query', 
    'model1_response', 'model2_response',
    'relevancy_1', 'relevancy_2',
    'faithfulness_1', 'faithfulness_2',
    'contextual_precision_1', 'contextual_precision_2',
    'contextual_recall_1', 'contextual_recall_2',
    'contextual_relevancy_1', 'contextual_relevancy_2',
    'conversation_completeness', # Conversation completeness applies to overall chat
    'call_deflection_effectiveness_1', 'call_deflection_effectiveness_2'
]
st.dataframe(df[display_columns], use_container_width=True)

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
    for metric_base in ['relevancy', 'faithfulness', 'contextual_precision', 
                        'contextual_recall', 'contextual_relevancy', 
                        'call_deflection_effectiveness']:
        st.markdown(f"- {metric_base.replace('_', ' ').title()} (Model 1): {row[f'{metric_base}_1']:.2f}")
        st.markdown(f"- {metric_base.replace('_', ' ').title()} (Model 2): {row[f'{metric_base}_2']:.2f}")
    st.markdown(f"- Conversation Completeness: {row['conversation_completeness']:.2f}")
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
    for metric_base in ['relevancy', 'faithfulness', 'contextual_precision', 
                        'contextual_recall', 'contextual_relevancy', 
                        'call_deflection_effectiveness']:
        st.markdown(f"- {metric_base.replace('_', ' ').title()} (Model 1): {row[f'{metric_base}_1']:.2f}")
        st.markdown(f"- {metric_base.replace('_', ' ').title()} (Model 2): {row[f'{metric_base}_2']:.2f}")
    st.markdown(f"- Conversation Completeness: {row['conversation_completeness']:.2f}")
    st.markdown("---") 