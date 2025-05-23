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
prompt_filter = st.text_input("Filter by prompt (optional):")
if prompt_filter:
    df = df[df['prompt'].str.contains(prompt_filter, case=False, na=False)]

# Sort by score difference
if st.checkbox("Sort by which model is better (score diff)"):
    df = df.assign(score_diff=df['score1'] - df['score2']).sort_values('score_diff', ascending=False)

# Show table with only 'prompt' renamed to 'user utterance'
renamed_df = df.rename(columns={
    'prompt': 'user utterance'
})
st.dataframe(renamed_df[['timestamp', 'user utterance', 'model1', 'response1', 'score1', 'model2', 'response2', 'score2']], use_container_width=True)

# Visualize score distributions
st.subheader("Score Distributions")
st.bar_chart(df[['score1', 'score2']])

# Show best/worst examples
st.subheader("Best Model 1 Wins")
best1 = df.sort_values('score1', ascending=False).head(3)
for _, row in best1.iterrows():
    st.markdown(f"**Prompt:** {row['prompt']}")
    st.markdown(f"**{row['model1']}**: {row['response1']}")
    st.markdown(f"**Score:** {row['score1']}")
    st.markdown(f"**{row['model2']}**: {row['response2']}")
    st.markdown(f"**Score:** {row['score2']}")
    st.markdown("---")

st.subheader("Best Model 2 Wins")
best2 = df.sort_values('score2', ascending=False).head(3)
for _, row in best2.iterrows():
    st.markdown(f"**Prompt:** {row['prompt']}")
    st.markdown(f"**{row['model2']}**: {row['response2']}")
    st.markdown(f"**Score:** {row['score2']}")
    st.markdown(f"**{row['model1']}**: {row['response1']}")
    st.markdown(f"**Score:** {row['score1']}")
    st.markdown("---") 