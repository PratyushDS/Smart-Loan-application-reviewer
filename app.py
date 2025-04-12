import streamlit as st
import joblib
import pandas as pd
import yaml
import os
from PIL import Image
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

class LoanApprovalApp:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Load model artifacts
        self.model = joblib.load(self.config['model_path'])
        self.preprocessor = joblib.load(self.config['preprocessor_path'])
        self.feature_importance = self._load_feature_importance()
        
        # Configure page settings
        st.set_page_config(
            page_title="AI Loan Advisor",
            page_icon="🏦",
            layout="wide"
        )
        self._set_custom_style()
        
        # Business rules configuration
        self.thresholds = {
            'cibil_score': 650,
            'loan_to_income': 0.35,
            'asset_coverage': 0.6,
            'min_income': 300000
        }

    def _set_custom_style(self):
        st.markdown("""
        <style>
        .stApp {
            background-image: url("https://img.freepik.com/free-vector/businessman-watching-showing-money-moon-sky-vision-success-business-concept_1258-17979.jpg?ga=GA1.1.1455084856.1744460700&semt=ais_hybrid&w=740");
            background-size: 100%;
        }
        # .main-container {
        #     background-color: rgba(255, 255, 255, 0.93);
        #     padding: 3rem;
        #     border-radius: 15px;
        #     box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        # }
        .metric-box {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #f8f9fa;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def _load_feature_importance(self):
        try:
            fi_path = os.path.join(self.config['reports']['tables_dir'], 'feature_importance.xlsx')
            df = pd.read_excel(fi_path)
            return df.set_index('Feature')['Importance'].to_dict()
        except Exception as e:
            st.warning("Feature importance data not available")
            return {}

    def _get_rejection_reasons(self, input_df):
        reasons = []
        applicant_data = input_df.iloc[0].to_dict()
        
        # Check key financial ratios
        loan_to_income = applicant_data['loan_amount'] / applicant_data['income_annum']
        total_assets = sum([applicant_data[k] for k in ['residential_assets_value', 
                                                       'commercial_assets_value',
                                                       'luxury_assets_value',
                                                       'bank_asset_value']])
        asset_coverage = total_assets / applicant_data['loan_amount'] if applicant_data['loan_amount'] > 0 else 0
        
        # Check against thresholds
        if applicant_data['cibil_score'] < self.thresholds['cibil_score']:
            reasons.append(f"📉 Low CIBIL Score ({applicant_data['cibil_score']} < {self.thresholds['cibil_score']})")
        
        if loan_to_income > self.thresholds['loan_to_income']:
            reasons.append(f"💸 High Loan-to-Income Ratio ({loan_to_income:.1%} > {self.thresholds['loan_to_income']:.0%})")
        
        if asset_coverage < self.thresholds['asset_coverage']:
            reasons.append(f"🏠 Insufficient Asset Coverage ({asset_coverage:.1%} < {self.thresholds['asset_coverage']:.0%})")
        
        if applicant_data['income_annum'] < self.thresholds['min_income']:
            reasons.append(f"💰 Low Annual Income (₹{applicant_data['income_annum']:,.0f} < ₹{self.thresholds['min_income']:,.0f})")
        
        return reasons[:3]  # Return top 3 reasons

    def get_user_input(self):
        with st.sidebar:
            st.header("📝 Loan Application Form")
            
            # Add prediction button at the top of the sidebar
            predict_clicked = st.button("🚀 Get Prediction", 
                                      help="Click to process your application",
                                      use_container_width=True)
            
            with st.expander("🧑💼 Personal Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    data = {
                        'cibil_score': st.slider('📊 CIBIL Score', 300, 900, 700),
                        'education': st.selectbox('🎓 Education', ['Graduate', 'Not Graduate']),
                        'self_employed': st.selectbox('💼 Self Employed', ['Yes', 'No']),
                    }
                with col2:
                    data.update({
                        'income_annum': st.number_input('💰 Annual Income (₹)', 0, 10000000, 500000),
                        'loan_term': st.slider('⏳ Loan Term (years)', 1, 20, 5),
                    })

            with st.expander("🏦 Financial Details", expanded=True):
                data.update({
                    'loan_amount': st.number_input('💳 Loan Amount (₹)', 0, 10000000, 300000),
                    'residential_assets_value': st.number_input('🏠 Residential Assets (₹)', 0, 10000000, 500000),
                    'commercial_assets_value': st.number_input('🏢 Commercial Assets (₹)', 0, 10000000, 500000),
                    'luxury_assets_value': st.number_input('💎 Luxury Assets (₹)', 0, 10000000, 500000),
                    'bank_asset_value': st.number_input('📈 Bank Assets (₹)', 0, 10000000, 500000),
                })

            return pd.DataFrame([data]), predict_clicked

    def _display_approved(self, approval_prob):
        with st.container():
            st.success("""
            ## ✅ Loan Approved! 🎉
            Congratulations! Your application meets our approval criteria.
            """)
            cols = st.columns([1, 2])
            with cols[0]:
                st.metric("Approval Probability", f"{approval_prob:.1f}%")
                st.progress(int(approval_prob)/100)
            with cols[1]:
                if self.feature_importance:
                    st.write("### 🏆 Top Approval Factors")
                    for feature, importance in list(self.feature_importance.items())[:3]:
                        st.write(f"- {feature.replace('_', ' ').title()} ({importance:.1%})")


    def _display_prediction_details(self, processed_input, prediction_proba):
        with st.expander("🔍 Prediction Details", expanded=False):
            st.write("### Model Input Features")
            st.write(pd.DataFrame(processed_input, 
                                columns=self.preprocessor.get_feature_names_out()))
            
            st.write("### Prediction Probabilities")
            proba_df = pd.DataFrame({
                'Rejected Probability': [prediction_proba[0][0]],
                'Approved Probability': [prediction_proba[0][1]]
            })
            st.dataframe(proba_df.style.format("{:.2%}"))
            
            # Add decision threshold adjustment
            threshold = st.slider("🚨 Decision Threshold", 0.0, 1.0, 0.5, 0.01)
            final_pred = 1 if prediction_proba[0][1] >= threshold else 0
            st.write(f"Current Threshold: {threshold:.0%} | Final Decision: {'Approved' if final_pred else 'Rejected'}")

    def _display_rejected(self, approval_prob, reasons, input_df):
        with st.container():
            st.error("""
            ## ❌ Loan Not Approved 
            ### Primary reasons for rejection:
            """)
            
            applicant_data = input_df.iloc[0]
            top_factors = sorted(self.feature_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:3]

            for factor, importance in top_factors:
                factor_name = factor.replace('_', ' ').title()
                user_value = applicant_data[factor]
                
                # Custom explanations for key factors
                if factor == 'cibil_score':
                    req_score = 650
                    st.write(f"""
                    🔻 **Credit History Risk**  
                    Your CIBIL Score of **{user_value}** is below our minimum requirement of **{req_score}**.  
                    *This indicates higher risk of repayment defaults.*
                    """)
                    
                elif factor == 'loan_amount':
                    income = applicant_data['income_annum']
                    ratio = (user_value / income) if income > 0 else 0
                    st.write(f"""
                    💰 **High Loan Burden**  
                    Your loan amount of **₹{user_value:,.0f}** represents **{ratio:.0%}** of your annual income.  
                    *We recommend keeping this below 35% for approval.*
                    """)
                    
                elif factor == 'income_annum':
                    min_req = 500000
                    st.write(f"""
                    📉 **Income Limitations**  
                    Your annual income of **₹{user_value:,.0f}** is below our minimum threshold of **₹{min_req:,.0f}**.  
                    *Higher income demonstrates better repayment capacity.*
                    """)
                    
                elif 'asset' in factor:
                    total_assets = sum([applicant_data['residential_assets_value'],
                                    applicant_data['commercial_assets_value'],
                                    applicant_data['luxury_assets_value'],
                                    applicant_data['bank_asset_value']])
                    st.write(f"""
                    🏠 **Insufficient Collateral**  
                    Your total assets of **₹{total_assets:,.0f}** provide limited coverage for the loan amount.  
                    *We require assets covering at least 60% of the loan value.*
                    """)
                    
                else:
                    st.write(f"""
                    ⚠️ **{factor_name}**  
                    Your provided value (**{user_value}**) in this category contributed significantly to the decision.  
                    *This factor is weighted as {importance:.1%} in our evaluation model.*
                    """)
        
        st.markdown("---")
        cols = st.columns([1, 2])
        with cols[0]:
            st.metric("Approval Probability", f"{approval_prob:.1f}%")
            st.progress(int(approval_prob)/100)

    def run(self):
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.title("🏦 Smart Loan application reviewer")
        st.markdown("---")
        
        # Get inputs and button state from sidebar
        input_df, predict_clicked = self.get_user_input()
        
        if predict_clicked:
            with st.spinner("Analyzing application..."):
                try:
                    # Process input and predict
                    processed_input = self.preprocessor.transform(input_df)
                    prediction = self.model.predict(processed_input)
                    prediction_proba = self.model.predict_proba(processed_input)
                    approval_prob = prediction_proba[0][1] * 100
                    
                    # Show prediction results
                    if prediction[0] == 1:
                        self._display_approved(approval_prob)
                    else:
                        reasons = self._get_rejection_reasons(input_df)
                        self._display_rejected(approval_prob, reasons,input_df)
                    
                    # Show raw input data
                    st.markdown("### 📋 Application Summary")
                    # Create formatted summary dataframe
                    summary_df = pd.DataFrame({
                        "Input Factor": input_df.columns,
                        "Value": input_df.iloc[0]
                    })

                    # Apply formatting based on factor type
                    def format_value(row):
                        factor = row["Input Factor"]
                        value = row["Value"]
                        
                        if factor in ['income_annum', 'loan_amount', 'residential_assets_value',
                                    'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']:
                            return f'₹{value:,.0f}'
                        elif factor == 'cibil_score':
                            return f'{value:.0f}'
                        elif factor == 'loan_term':
                            return f'{value:.0f} years'
                        elif factor in ['education', 'self_employed']:
                            return value.title()
                        else:
                            return value

                    summary_df["Value"] = summary_df.apply(format_value, axis=1)

                    # Display with proper headers
                    st.dataframe(
                        summary_df.set_index("Input Factor"),
                        height=400,
                        use_container_width=True
                    )
            
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    app = LoanApprovalApp('config/config.yaml')
    app.run()