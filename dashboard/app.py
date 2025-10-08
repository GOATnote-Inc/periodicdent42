"""
Streamlit Dashboard for matprov

Interactive visualization for:
- Experiment timeline
- Prediction accuracy over time
- Model performance comparison
- Provenance explorer
- Candidate selection
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from matprov.registry.database import Database
from matprov.registry.queries import PredictionQueries
from matprov.registry.models import Model, Prediction, ExperimentOutcome, PredictionError
from sqlalchemy import select

# Page config
st.set_page_config(
    page_title="matprov Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    h1 {
        color: #1f77b4;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def get_database():
    """Get database connection"""
    return Database()

# Load data functions
@st.cache_data(ttl=60)
def load_models(_db):
    """Load all models"""
    with _db.session() as session:
        models = session.execute(select(Model).order_by(Model.training_date.desc())).scalars().all()
        return [
            {
                'id': m.id,
                'model_id': m.model_id,
                'version': m.version,
                'architecture': m.architecture,
                'training_date': m.training_date
            }
            for m in models
        ]

@st.cache_data(ttl=60)
def load_predictions(_db, model_id=None, limit=1000):
    """Load predictions"""
    with _db.session() as session:
        query = select(Prediction, Model, ExperimentOutcome, PredictionError).join(
            Model, Prediction.model_id == Model.id
        ).outerjoin(
            ExperimentOutcome, Prediction.id == ExperimentOutcome.prediction_id
        ).outerjoin(
            PredictionError, Prediction.id == PredictionError.prediction_id
        )
        
        if model_id:
            query = query.where(Model.model_id == model_id)
        
        query = query.limit(limit)
        
        results = session.execute(query).all()
        
        data = []
        for pred, model, outcome, error in results:
            row = {
                'prediction_id': pred.prediction_id,
                'material_formula': pred.material_formula,
                'predicted_tc': pred.predicted_tc,
                'uncertainty': pred.uncertainty,
                'predicted_class': pred.predicted_class,
                'confidence': pred.confidence,
                'prediction_date': pred.prediction_date,
                'model_id': model.model_id,
                'model_version': model.version,
                'has_outcome': outcome is not None
            }
            
            if outcome:
                row.update({
                    'experiment_id': outcome.experiment_id,
                    'actual_tc': outcome.actual_tc,
                    'validation_status': outcome.validation_status,
                    'phase_purity': outcome.phase_purity,
                    'experiment_date': outcome.experiment_date
                })
            
            if error:
                row.update({
                    'absolute_error': error.absolute_error,
                    'relative_error': error.relative_error
                })
            
            data.append(row)
        
        return pd.DataFrame(data)

def main():
    """Main dashboard"""
    
    # Header
    st.title("🔬 matprov Dashboard")
    st.markdown("**Materials Provenance Tracking & Analysis**")
    
    # Sidebar
    st.sidebar.header("Settings")
    db = get_database()
    
    # Load models
    models = load_models(db)
    if not models:
        st.warning("No models found. Please register a model first.")
        st.stop()
    
    model_options = {m['model_id']: f"{m['model_id']} (v{m['version']})" for m in models}
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    # Load data
    df = load_predictions(db, selected_model)
    
    if df.empty:
        st.warning(f"No predictions found for {selected_model}")
        st.stop()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "📈 Performance",
        "🔍 Predictions",
        "🎯 Candidates"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.header("Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_predictions = len(df)
        validated = df['has_outcome'].sum()
        unvalidated = total_predictions - validated
        
        if validated > 0:
            avg_error = df[df['has_outcome']]['absolute_error'].abs().mean()
        else:
            avg_error = 0
        
        col1.metric("Total Predictions", total_predictions)
        col2.metric("Validated", validated)
        col3.metric("Pending", unvalidated)
        col4.metric("Avg Error", f"{avg_error:.2f}K" if validated > 0 else "N/A")
        
        # Timeline
        st.subheader("Prediction Timeline")
        
        df_timeline = df.copy()
        df_timeline['date'] = pd.to_datetime(df_timeline['prediction_date']).dt.date
        timeline_counts = df_timeline.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            timeline_counts,
            x='date',
            y='count',
            title='Predictions Over Time',
            labels={'date': 'Date', 'count': 'Number of Predictions'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Class distribution
        st.subheader("Prediction Class Distribution")
        
        if 'predicted_class' in df.columns and df['predicted_class'].notna().any():
            class_counts = df['predicted_class'].value_counts()
            
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title='Predicted Classes'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Performance
    with tab2:
        st.header("Model Performance")
        
        df_validated = df[df['has_outcome']].copy()
        
        if len(df_validated) == 0:
            st.info("No validated predictions yet. Add experimental outcomes to see performance metrics.")
        else:
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            mae = df_validated['absolute_error'].abs().mean()
            rmse = (df_validated['absolute_error'] ** 2).mean() ** 0.5
            
            # R²
            y_true = df_validated['actual_tc']
            y_pred = df_validated['predicted_tc']
            ss_res = ((y_true - y_pred) ** 2).sum()
            ss_tot = ((y_true - y_true.mean()) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            col1.metric("MAE", f"{mae:.2f}K")
            col2.metric("RMSE", f"{rmse:.2f}K")
            col3.metric("R²", f"{r2:.4f}")
            
            # Predicted vs Actual scatter
            st.subheader("Predicted vs Actual Tc")
            
            fig = px.scatter(
                df_validated,
                x='predicted_tc',
                y='actual_tc',
                hover_data=['material_formula', 'experiment_id'],
                title='Predicted vs Actual Critical Temperature',
                labels={'predicted_tc': 'Predicted Tc (K)', 'actual_tc': 'Actual Tc (K)'}
            )
            
            # Add diagonal line (perfect prediction)
            min_tc = min(df_validated['predicted_tc'].min(), df_validated['actual_tc'].min())
            max_tc = max(df_validated['predicted_tc'].max(), df_validated['actual_tc'].max())
            fig.add_trace(go.Scatter(
                x=[min_tc, max_tc],
                y=[min_tc, max_tc],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='gray')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Error distribution
            st.subheader("Error Distribution")
            
            fig = px.histogram(
                df_validated,
                x='absolute_error',
                title='Prediction Error Distribution',
                labels={'absolute_error': 'Absolute Error (K)', 'count': 'Frequency'},
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Predictions
    with tab3:
        st.header("Prediction Explorer")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.selectbox(
                "Status",
                options=["All", "Validated", "Unvalidated"]
            )
        
        with col2:
            min_tc = st.number_input("Min Predicted Tc (K)", value=0.0, step=10.0)
        
        with col3:
            max_tc = st.number_input("Max Predicted Tc (K)", value=200.0, step=10.0)
        
        # Apply filters
        df_filtered = df.copy()
        
        if status_filter == "Validated":
            df_filtered = df_filtered[df_filtered['has_outcome']]
        elif status_filter == "Unvalidated":
            df_filtered = df_filtered[~df_filtered['has_outcome']]
        
        df_filtered = df_filtered[
            (df_filtered['predicted_tc'] >= min_tc) &
            (df_filtered['predicted_tc'] <= max_tc)
        ]
        
        st.write(f"Showing {len(df_filtered)} predictions")
        
        # Display table
        display_cols = [
            'prediction_id', 'material_formula', 'predicted_tc', 'uncertainty',
            'predicted_class', 'confidence', 'has_outcome'
        ]
        
        if status_filter == "Validated":
            display_cols.extend(['actual_tc', 'absolute_error', 'validation_status'])
        
        st.dataframe(
            df_filtered[display_cols].sort_values('predicted_tc', ascending=False),
            use_container_width=True
        )
    
    # TAB 4: Candidates
    with tab4:
        st.header("Candidates for Validation")
        
        st.markdown("""
        **Shannon Entropy-based Selection**: Prioritizes experiments that maximize information gain.
        
        High-value candidates have:
        - High uncertainty (model is unsure → informative)
        - Near classification boundaries (critical to validate)
        - Chemistry diversity (explore parameter space)
        """)
        
        # Get unvalidated predictions
        df_unvalidated = df[~df['has_outcome']].copy()
        
        if len(df_unvalidated) == 0:
            st.info("All predictions have been validated!")
        else:
            # Sort by uncertainty (approximation of entropy)
            if 'uncertainty' in df_unvalidated.columns:
                df_candidates = df_unvalidated.sort_values('uncertainty', ascending=False)
            else:
                df_candidates = df_unvalidated.sort_values('predicted_tc', ascending=False)
            
            st.subheader(f"Top 10 Candidates (out of {len(df_unvalidated)} unvalidated)")
            
            # Display top candidates
            display_cols = [
                'prediction_id', 'material_formula', 'predicted_tc',
                'uncertainty', 'predicted_class', 'confidence'
            ]
            
            st.dataframe(
                df_candidates[display_cols].head(10),
                use_container_width=True
            )
            
            # Visualize
            fig = px.scatter(
                df_candidates.head(50),
                x='predicted_tc',
                y='uncertainty',
                size='confidence',
                hover_data=['material_formula', 'prediction_id'],
                title='Top 50 Candidates (Tc vs Uncertainty)',
                labels={
                    'predicted_tc': 'Predicted Tc (K)',
                    'uncertainty': 'Uncertainty (K)',
                    'confidence': 'Confidence'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Export
            if st.button("Export Top 10 as JSON"):
                import json
                candidates_json = df_candidates[display_cols].head(10).to_dict(orient='records')
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(candidates_json, indent=2),
                    file_name="candidates.json",
                    mime="application/json"
                )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**matprov** v0.1.0")
    st.sidebar.markdown("Materials Provenance Tracking")


if __name__ == "__main__":
    main()

