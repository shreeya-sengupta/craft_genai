import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(page_title="✨ Embedding Visualizer", layout="wide")

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()
st.sidebar.success("Model loaded successfully!")

# --------------------------
# Sidebar Settings
# --------------------------
st.sidebar.title("Embedding Visualizer Settings")

embedding_type = st.sidebar.radio(
    "Embedding Type",
    ["Sentence Embeddings", "Word Embeddings"]
)

text_to_visualise = st.text_area(
    label="📝 Enter text (one sentence or document per line)",
    placeholder="Type sentences or a paragraph here..."
)

# Clean text input
text_to_visualise = text_to_visualise.strip().split("\n")
text_to_visualise = [item.strip() for item in text_to_visualise if len(item.strip()) > 1]

dim_choice = st.sidebar.slider(
    "Number of dimensions to visualize (select up to 10)",
    2, 10, 2
)

# --------------------------
# Embedding Creation
# --------------------------
st.title("💡 Embedding Visualizer")

if text_to_visualise:
    with st.spinner("🔄 Creating embeddings..."):
        if embedding_type == "Sentence Embeddings":
            texts = text_to_visualise
        else:
            texts = list(set(" ".join(text_to_visualise).split()))

        embeddings = embedding_model.encode(texts)
        orig_dim = embeddings.shape[1]

    st.sidebar.info(f"Original embedding size: **{orig_dim}D**")

    # --------------------------
    # Similarity Calculation
    # --------------------------
    similarity_matrix = cosine_similarity(embeddings)
    df_sim = pd.DataFrame(similarity_matrix, index=texts, columns=texts)

    # Let user choose a reference text
    st.sidebar.subheader("Similarity Highlighting")
    reference_text = st.sidebar.selectbox("Select reference text", texts)
    ref_index = texts.index(reference_text)
    similarities = similarity_matrix[ref_index]

    # Dynamic Coloring Options
    st.sidebar.subheader("Color Points By")
    color_option = st.sidebar.selectbox(
        "Choose coloring scheme",
        ["Similarity to Reference", "Text Length"]
    )

    if color_option == "Text Length":
        color_values = [len(t) for t in texts]
        color_name = "Text Length"
    else:
        color_values = similarities
        color_name = f"Similarity to '{reference_text}'"

    # Create Tabs for better layout
    tab1, tab2, tab3 = st.tabs(
        ["Visualization", "Semantic Search", "Similarity Heatmap"]
    )

    # --------------------------
    # TAB 1: Visualization
    # --------------------------
    with tab1:
        st.subheader(f"{dim_choice}D Visualization of {embedding_type}")
        color_scale = px.colors.sequential.Viridis

        if dim_choice == 2:
            fig = px.scatter(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                text=texts,
                color=color_values,
                color_continuous_scale=color_scale,
                title=f"2D {embedding_type} Visualization — Color = {color_name}",
                labels={"x": "Dim 1", "y": "Dim 2", "color": color_option}
            )
            fig.update_traces(textposition="top center")

        elif dim_choice == 3:
            fig = px.scatter_3d(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                z=embeddings[:, 2],
                text=texts,
                color=color_values,
                color_continuous_scale=color_scale,
                labels={"Color": color_name},
                title=f"3D {embedding_type} Visualization — Color = {color_name}"
            )
            fig.update_traces(textposition="top center")

        else:
            cols = [f"Dim {i+1}" for i in range(dim_choice)]
            df_plot = pd.DataFrame(embeddings[:, :dim_choice], columns=cols)
            df_plot["Text"] = texts
            df_plot["ColorValue"] = color_values
            fig = px.parallel_coordinates(
                df_plot.drop(columns=["Text"]),
                color=color_values,
                color_continuous_scale=color_scale,
                title=f"{dim_choice}D {embedding_type} Visualization - Color - {color_name}"
            )

        st.plotly_chart(fig, use_container_width=True)


    # --------------------------
    # TAB 2: Semantic Search
    # --------------------------
    with tab2:
        st.subheader("🔍 Semantic Search Playground")

        query = st.text_input("Type a query to find semantically similar texts:")

        if query:
            q_emb = embedding_model.encode([query])
            sims = cosine_similarity(q_emb, embeddings)[0]
            top_idx = np.argsort(sims)[::-1][:10]

            st.markdown("Top 10 Most Similar Sentences")
            st.caption("Higher similarity → more semantically related to your query")

            # Create color-coded similarity bars
            for rank, i in enumerate(top_idx):
                sim_score = sims[i]
                color_intensity = int(sim_score * 255)
                bg_color = f"rgba(0, 128, 255, {sim_score})"
                st.markdown(
                    f"""
                    <div style="
                        background-color:{bg_color};
                        border-radius:10px;
                        padding:10px;
                        margin-bottom:6px;
                        color:white;
                        font-size:15px;
                    ">
                        <b>#{rank+1}</b> — {texts[i]}  
                        <span style="float:right;"> {sim_score:.3f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Plot similarity distribution
            st.markdown("### Similarity Distribution")
            df_similar = pd.DataFrame({
                "Text": [texts[i] for i in top_idx],
                "Similarity": [sims[i] for i in top_idx]
            })
            fig_similar = px.bar(
                df_similar,
                x="Text",
                y="Similarity",
                color="Similarity",
                color_continuous_scale="Blues",
                title="Similarity Scores to Query",
            )
            fig_similar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_similar, use_container_width=True)

            # Calculate top similarities
            top_sims = [sims[i] for i in top_idx[:5]]
            avg_sim = np.mean(top_sims)
            max_sim = np.max(top_sims)
            min_sim = np.min(top_sims)

            # Count how many texts have high similarity (>0.7)
            high_count = sum(s > 0.7 for s in sims)

            # Insight logic
            if avg_sim > 0.85:
                insight = f"Excellent match! Your query is highly similar to the top texts (avg similarity: {avg_sim:.2f}).\n" \
                          f"Top match: {max_sim:.2f}, bottom of top 5: {min_sim:.2f}. {high_count} texts in the dataset are very similar."
            elif avg_sim > 0.65:
                insight = f"Moderate match. Your query shares some meaning with top texts (avg similarity: {avg_sim:.2f}).\n" \
                          f"Consider rephrasing or adding keywords to get closer matches. {high_count} texts are fairly similar."
            else:
                insight = f"Low match. Your query seems semantically different from most texts (avg similarity: {avg_sim:.2f}).\n" \
                          f"Top match: {max_sim:.2f}. Try using terms closer to your dataset to improve similarity."

            st.info(insight)


            # Optional comparison
            st.markdown("---")
            st.markdown("### Compare Sentences Pairwise")
            col1, col2 = st.columns(2)
            with col1:
                text_a = st.selectbox("Choose first sentence", texts, key="compare_a")
            with col2:
                text_b = st.selectbox("Choose second sentence", texts, key="compare_b")

            if st.button("Compare Similarity"):
                a_emb = embedding_model.encode([text_a])
                b_emb = embedding_model.encode([text_b])
                sim_pair = cosine_similarity(a_emb, b_emb)[0][0]
                st.markdown(f"**Similarity Score:** `{sim_pair:.3f}`")
                bar_color = "green" if sim_pair > 0.7 else "orange" if sim_pair > 0.4 else "red"
                st.spinner("in progress")
                st.markdown(f"Semantic relationship: <span style='color:{bar_color};font-weight:bold;'>{'High' if sim_pair>0.7 else 'Medium' if sim_pair>0.4 else 'Low'}</span>", unsafe_allow_html=True)


    # --------------------------
    # TAB 3: Similarity Heatmap
    # --------------------------
    with tab3:
        st.subheader("Cosine Similarity Heatmap")
        fig_heatmap = px.imshow(
            df_sim,
            x=texts,
            y=texts,
            color_continuous_scale="Viridis",
            title="Cosine Similarity Matrix",
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)


else:
    st.info("Enter text in the box above to generate embeddings.")
