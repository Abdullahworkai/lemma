import streamlit as st
import pandas as pd
import spacy
from collections import defaultdict
import io

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.error("Please install spaCy model: python -m spacy download en_core_web_sm")
        return None

def get_lemma_and_variations(phrase, nlp):
    """
    Get the lemma (root form) and generate common variations of a word or phrase.
    Handles both single words and multi-word expressions (like 'watch out for').
    """
    doc = nlp(phrase.lower().strip())
    
    # For multi-word expressions, lemmatize each word
    lemmas = []
    tokens = []
    pos_tags = []
    
    for token in doc:
        if not token.is_punct and not token.is_space:
            lemmas.append(token.lemma_)
            tokens.append(token.text)
            pos_tags.append(token.pos_)
    
    # Create the lemmatized form
    lemma_phrase = " ".join(lemmas)
    
    # Generate variations based on POS tags
    variations = set([phrase.lower().strip(), lemma_phrase])
    
    # For single words, generate more variations
    if len(tokens) == 1:
        token = doc[0]
        variations.update(generate_word_variations(token, nlp))
    else:
        # For phrases, generate variations of each component
        phrase_variations = generate_phrase_variations(tokens, lemmas, pos_tags, nlp)
        variations.update(phrase_variations)
    
    return lemma_phrase, sorted(list(variations))

def generate_word_variations(token, nlp):
    """Generate morphological variations of a single word."""
    variations = set()
    lemma = token.lemma_
    pos = token.pos_
    
    # Add the lemma
    variations.add(lemma)
    variations.add(token.text)
    
    # Generate variations based on POS
    if pos == "VERB":
        # Verb forms: base, -s, -ing, -ed
        variations.add(lemma)  # base form
        variations.add(lemma + "s")  # third person
        variations.add(lemma + "es")  # third person (alternative)
        
        # Handle -ing form
        if lemma.endswith("e"):
            variations.add(lemma[:-1] + "ing")
        else:
            variations.add(lemma + "ing")
        
        # Handle -ed form
        if lemma.endswith("e"):
            variations.add(lemma + "d")
        else:
            variations.add(lemma + "ed")
        
        # Irregular verbs common patterns
        if lemma.endswith("y"):
            variations.add(lemma[:-1] + "ied")
            variations.add(lemma[:-1] + "ies")
    
    elif pos == "NOUN":
        # Noun forms: singular, plural
        variations.add(lemma)  # singular
        variations.add(lemma + "s")  # plural
        variations.add(lemma + "es")  # plural (alternative)
        
        if lemma.endswith("y"):
            variations.add(lemma[:-1] + "ies")  # happy -> happies
        if lemma.endswith("f"):
            variations.add(lemma[:-1] + "ves")  # leaf -> leaves
        if lemma.endswith("fe"):
            variations.add(lemma[:-2] + "ves")  # knife -> knives
    
    elif pos == "ADJ":
        # Adjective forms: base, comparative, superlative
        variations.add(lemma)  # base
        variations.add(lemma + "er")  # comparative
        variations.add(lemma + "est")  # superlative
        
        if lemma.endswith("y"):
            variations.add(lemma[:-1] + "ier")
            variations.add(lemma[:-1] + "iest")
    
    elif pos == "ADV":
        # Adverb forms
        variations.add(lemma)
        if lemma.endswith("ly"):
            variations.add(lemma[:-2])  # remove -ly
    
    return variations

def generate_phrase_variations(tokens, lemmas, pos_tags, nlp):
    """Generate variations for multi-word phrases."""
    variations = set()
    
    # Create variations by inflecting each word
    all_word_variations = []
    for i, (token, lemma, pos) in enumerate(zip(tokens, lemmas, pos_tags)):
        doc_token = nlp(token)[0]
        word_vars = list(generate_word_variations(doc_token, nlp))
        all_word_variations.append(word_vars)
    
    # Generate combinations (limit to avoid explosion)
    if len(tokens) <= 4:  # Only for reasonable phrase lengths
        from itertools import product
        
        # Limit variations per word to avoid too many combinations
        limited_variations = [vars[:5] for vars in all_word_variations]
        
        for combo in product(*limited_variations):
            variations.add(" ".join(combo))
    else:
        # For longer phrases, just add basic variations
        variations.add(" ".join(tokens))
        variations.add(" ".join(lemmas))
    
    return variations

def process_vocabulary_list(df, column_name, nlp):
    """Process the entire vocabulary list and generate lemmas and variations."""
    results = []
    
    for idx, row in df.iterrows():
        if pd.isna(row[column_name]) or str(row[column_name]).strip() == "":
            continue
            
        word_or_phrase = str(row[column_name]).strip()
        lemma, variations = get_lemma_and_variations(word_or_phrase, nlp)
        
        results.append({
            "Original": word_or_phrase,
            "Lemma (Root)": lemma,
            "Variations": ", ".join(variations),
            "Number of Variations": len(variations)
        })
    
    return pd.DataFrame(results)

# Streamlit App
def main():
    st.set_page_config(page_title="Vocabulary Lemmatization Tool", layout="wide")
    
    st.title("📚 Vocabulary Lemmatization Tool")
    st.markdown("""
    This tool helps identify root words (lemmas) and their variations from your vocabulary list.
    It's designed to recognize when students use different forms of the same word without penalizing them.
    
    **Key Features:**
    - ✅ Identifies root forms (lemmas) of words and phrases
    - ✅ Generates morphological variations (e.g., watching, watched, watches)
    - ✅ Handles multi-word expressions (e.g., "watch out for" → "watching out for")
    - ❌ Does NOT provide synonyms (e.g., won't suggest "observe" for "watch")
    """)
    
    # Load spaCy model
    nlp = load_spacy_model()
    if nlp is None:
        st.stop()
    
    st.sidebar.header("⚙️ Instructions")
    st.sidebar.markdown("""
    1. Upload a CSV file with your vocabulary words
    2. Select the column containing the vocabulary
    3. View and download the results with lemmas and variations
    
    **CSV Format Example:**
    ```
    Word
    analyze
    watch out for
    running
    beautiful
    ```
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your vocabulary CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📄 Original Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column selection
            columns = df.columns.tolist()
            selected_column = st.selectbox(
                "Select the column containing vocabulary words:",
                options=columns
            )
            
            if st.button("🚀 Generate Lemmas and Variations", type="primary"):
                with st.spinner("Processing vocabulary list..."):
                    results_df = process_vocabulary_list(df, selected_column, nlp)
                
                st.success(f"✅ Processed {len(results_df)} vocabulary items!")
                
                # Display results
                st.subheader("📊 Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Words/Phrases", len(results_df))
                with col2:
                    avg_variations = results_df["Number of Variations"].mean()
                    st.metric("Avg. Variations per Word", f"{avg_variations:.1f}")
                with col3:
                    total_variations = results_df["Number of Variations"].sum()
                    st.metric("Total Variations", total_variations)
                
                # Download button
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_data,
                    file_name="vocabulary_lemmas_variations.csv",
                    mime="text/csv"
                )
                
                # Example section
                with st.expander("🔍 View Detailed Examples"):
                    for idx, row in results_df.head(5).iterrows():
                        st.markdown(f"""
                        **Original:** {row['Original']}  
                        **Lemma:** {row['Lemma (Root)']}  
                        **Variations:** {row['Variations']}
                        """)
                        st.divider()
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted.")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show example
        st.subheader("💡 Example")
        example_data = pd.DataFrame({
            "Word": ["analyze", "watch out for", "running", "beautiful", "children"]
        })
        
        st.markdown("**Sample Input:**")
        st.dataframe(example_data, use_container_width=True)
        
        if st.button("Try Example"):
            with st.spinner("Processing example..."):
                results_df = process_vocabulary_list(example_data, "Word", nlp)
            st.markdown("**Sample Output:**")
            st.dataframe(results_df, use_container_width=True)

if __name__ == "__main__":
    main()
