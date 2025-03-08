import streamlit as st
import pandas as pd
import json
from misinformation_detector import MisinformationDetector

# Set page configuration
st.set_page_config(
    page_title="Misinformation Detector",
    page_icon="🔍",
    layout="wide"
)

# Initialize the detector
@st.cache_resource
def load_detector():
    return MisinformationDetector()

detector = load_detector()

# App title and description
st.title("🔍 Misinformation Detector")
st.markdown("""
This application analyzes content to detect potential misinformation using NLP techniques, 
fact verification, and source credibility analysis.
""")

# Input section
st.header("Content Analysis")
text_input = st.text_area("Enter the text to analyze:", height=200)
source_url = st.text_input("Source URL (optional):")

# Analysis button
if st.button("Analyze Content"):
    if text_input:
        with st.spinner("Analyzing content..."):
            # Run the analysis
            result = detector.analyze_content(text_input, source_url if source_url else None)
            
            # Display results
            st.header("Analysis Results")
            
            # Overall assessment with color coding
            assessment = result["assessment"]
            score = result["misinformation_score"]
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if assessment == "Likely Reliable":
                    st.success(f"Assessment: {assessment}")
                elif assessment == "Possibly Reliable":
                    st.success(f"Assessment: {assessment}")
                elif assessment == "Uncertain":
                    st.warning(f"Assessment: {assessment}")
                elif assessment == "Possibly Misinformation":
                    st.error(f"Assessment: {assessment}")
                else:  # Likely Misinformation
                    st.error(f"Assessment: {assessment}")
            
            with col2:
                st.progress(score)
                st.text(f"Misinformation Score: {score:.2f} (Higher = More likely to be misinformation)")
            
            # Tabs for detailed results
            tab1, tab2, tab3, tab4 = st.tabs(["Extracted Claims", "Source Analysis", "Linguistic Features", "Fact Checks"])
            
            with tab1:
                st.subheader("Extracted Claims")
                if result["claims"]:
                    for i, claim in enumerate(result["claims"]):
                        st.write(f"{i+1}. {claim}")
                else:
                    st.write("No specific claims were extracted.")
            
            with tab2:
                st.subheader("Source Analysis")
                if source_url:
                    st.write(f"Source URL: {source_url}")
                    st.write(f"Credibility Score: {result['source_credibility']:.2f} (Lower = More credible)")
                    
                    # Create a simple gauge
                    source_cred = 1 - result['source_credibility']  # Invert for display
                    if source_cred > 0.7:
                        st.success(f"Source appears credible (Score: {source_cred:.2f})")
                    elif source_cred > 0.4:
                        st.warning(f"Source has mixed credibility (Score: {source_cred:.2f})")
                    else:
                        st.error(f"Source has low credibility (Score: {source_cred:.2f})")
                else:
                    st.write("No source URL provided for analysis.")
            
            with tab3:
                st.subheader("Linguistic Features")
                features = result["linguistic_features"]
                
                # Create two columns for features
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Word Count", features["word_count"])
                    st.metric("Average Word Length", f"{features['avg_word_length']:.2f}")
                    st.metric("Uppercase Ratio", f"{features['uppercase_ratio']:.2f}")
                
                with col2:
                    st.metric("Sentence Count", features["sentence_count"])
                    st.metric("Emotional Word Ratio", f"{features['emotion_word_ratio']:.2f}")
                    st.metric("Exclamation Marks", features["exclamation_count"])
                
                # Display warning for potential red flags
                if features["emotion_word_ratio"] > 0.1 or features["exclamation_count"] > 2:
                    st.warning("⚠️ High emotional content detected - this can be a sign of sensationalism")
                
                if features["uppercase_ratio"] > 0.2:
                    st.warning("⚠️ Excessive use of uppercase detected - often used for emphasis in misleading content")
            
            with tab4:
                st.subheader("Fact Checks")
                if result["fact_checks"]:
                    for check in result["fact_checks"]:
                        st.write(f"**Claim:** {check['claim']}")
                        
                        if check["checks"]:
                            for i, fc in enumerate(check["checks"]):
                                st.write(f"Check #{i+1} from {fc['source']}:")
                                st.json(fc["result"])
                        else:
                            st.write("No fact checks available for this claim.")
                        
                        st.write("---")
                else:
                    st.write("No fact checks were performed.")
            
            # Raw JSON data (collapsed)
            with st.expander("View Raw Analysis Data"):
                st.json(result)
    else:
        st.error("Please enter some text to analyze.")

# About section
with st.expander("About this Application"):
    st.markdown("""
    This open-source misinformation detection application uses multiple approaches to analyze content:
    
    1. **Content Analysis**: Linguistic patterns, sentiment, and claim extraction
    2. **Source Credibility**: Evaluating the reputation of content sources
    3. **Fact Verification**: Checking claims against verified information
    4. **Machine Learning**: Using trained models to identify misinformation patterns
    
    This is an open-source tool and can be extended with additional data sources and detection methods.
    """)
