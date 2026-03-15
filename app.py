import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fpdf import FPDF
import io
import datetime
import calendar
import tempfile
import os

st.set_page_config(page_title="Data Insights & Forecast Dashboard", layout="wide")

st.title("📊 Universal Data Insights & Forecast Dashboard")
st.markdown("""
This app performs Exploratory Data Analysis (EDA), calculates key metrics (Revenue, Expenditure, Profit/Loss, ATV), provides colorful charts, performs Holt-Winters forecasting, and generates a downloadable PDF report.
No virtual environment is needed to run this—just Python, and you can easily deploy it on **Streamlit Community Cloud** with the included `requirements.txt`.
""")

st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_upload("Upload your dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Read data
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
        
    st.header("1. Exploratory Data Analysis (EDA)")
    with st.expander("🔍 View Raw Data & Summary"):
        st.write("### Raw Data (First 5 Rows)")
        st.dataframe(df.head())
        st.write("### Data Summary")
        st.dataframe(df.describe())
        st.write("### Missing Values")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Count"}))
        st.write("### Data Types")
        st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}))

    st.header("2. Data Mapping & Settings")
    columns = df.columns.tolist()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        date_col = st.selectbox("Select Date Column", ["None"] + columns, index=0)
    with col2:
        rev_col = st.selectbox("Select Revenue Column", ["None"] + columns, index=0)
    with col3:
        exp_col = st.selectbox("Select Expenditure/Budget Column", ["None"] + columns, index=0)
    with col4:
        tx_col = st.selectbox("Select Transactions Column (for ATV)", ["None"] + columns, index=0)
        
    st.sidebar.header("💸 Discount Settings")
    discount_option = st.sidebar.radio("Apply Discount to Revenue?", ["0%", "10%", "20%", "30%", "Custom"])
    
    if discount_option == "Custom":
        discount_value = st.sidebar.number_input("Enter Custom Discount (%)", min_value=0.0, max_value=100.0, value=0.0)
    else:
        discount_value = float(discount_option.strip("%"))
        
    st.sidebar.header("📈 Forecast Settings")
    forecast_option = st.sidebar.radio("Forecast Horizon (Years)", ["1", "3", "5", "7", "Custom"])
    if forecast_option == "Custom":
        forecast_years = st.sidebar.number_input("Enter Forecast Years", min_value=1, max_value=20, value=5)
    else:
        forecast_years = int(forecast_option)

    if rev_col != "None" and exp_col != "None":
        st.header("3. Key Performance Indicators (KPIs)")
        
        # Data preparation
        processed_df = df.copy()
        
        # Apply discount
        discount_multiplier = 1 - (discount_value / 100.0)
        processed_df['Adjusted_Revenue'] = processed_df[rev_col] * discount_multiplier
        
        total_revenue = processed_df['Adjusted_Revenue'].sum()
        total_expenditure = processed_df[exp_col].sum()
        profit_loss = total_revenue - total_expenditure
        
        atv_value = 0
        if tx_col != "None":
            total_tx = processed_df[tx_col].sum()
            if total_tx > 0:
                atv_value = total_revenue / total_tx
                
        # Metrics Display
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Revenue (Adj.)", f"${total_revenue:,.2f}")
        m2.metric("Total Expenditure", f"${total_expenditure:,.2f}")
        m3.metric("Profit / Loss", f"${profit_loss:,.2f}", delta="Profit" if profit_loss > 0 else "Loss", delta_color="normal" if profit_loss > 0 else "inverse")
        if tx_col != "None":
            m4.metric("Avg. Transaction Value (ATV)", f"${atv_value:,.2f}")
            
        st.header("4. AI Suggestions")
        if profit_loss > 0:
            st.success("🤖 **AI Suggestion**: You are rocking! No suggestions needed. Keep up the excellent performance!")
            ai_suggestion_text = "You are rocking! No suggestions needed. Keep up the excellent performance!"
        else:
            st.warning("🤖 **AI Suggestion**: You are operating at a loss. Consider the following:\n- Review and reduce operational expenditures.\n- Analyze the discount strategy; high discounts might be eating into margins.\n- Identify low-performing products/services and optimize them.")
            ai_suggestion_text = "You are operating at a loss. Consider reducing operational expenditures, reviewing discount strategies, and identifying low-performing areas."

        st.header("5. Visualizations & Forecasting")
        
        # Aggregating by date if available
        if date_col != "None":
            try:
                processed_df[date_col] = pd.to_datetime(processed_df[date_col])
                processed_df = processed_df.sort_values(by=date_col)
                # Aggregate by month for simpler processing if it's daily
                ts_df = processed_df.set_index(date_col).resample('M').sum(numeric_only=True).reset_index()
                
                # Plotly Chart
                fig = px.line(ts_df, x=date_col, y=['Adjusted_Revenue', exp_col], 
                              title="Revenue and Expenditure Over Time",
                              labels={"value": "Amount ($)", "variable": "Metric"})
                st.plotly_chart(fig, use_container_width=True)
                
                # Holt Winters Forecasting
                st.subheader(f"Holt-Winters Forecasting ({forecast_years} Years)")
                
                if len(ts_df) >= 24: # Need enough data points for seasonal Holt-Winters
                    ts_data = ts_df['Adjusted_Revenue'].values
                    # 12 months = 1 year frequency
                    periods_to_forecast = forecast_years * 12
                    
                    try:
                        model = ExponentialSmoothing(ts_data, trend="add", seasonal="add", seasonal_periods=12).fit()
                        forecast = model.forecast(periods_to_forecast)
                        
                        # Generate future dates
                        last_date = ts_df[date_col].iloc[-1]
                        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods_to_forecast, freq='M')
                        
                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Forecasted_Revenue': forecast
                        })
                        
                        # Plot Historical + Forecast
                        fig_forecast = go.Figure()
                        fig_forecast.add_trace(go.Scatter(x=ts_df[date_col], y=ts_df['Adjusted_Revenue'], mode='lines', name='Historical Revenue', line=dict(color='blue')))
                        fig_forecast.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted_Revenue'], mode='lines', name='Forecast', line=dict(color='orange', dash='dash')))
                        fig_forecast.update_layout(title=f"Revenue Forecast for Next {forecast_years} Years", xaxis_title="Date", yaxis_title="Revenue ($)")
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Not enough data or variability to perform seasonal Holt-Winters forecasting. Error: {e}")
                        fig_forecast = None
                else:
                    st.info("Need at least 24 months of historical date data to perform reliable seasonal Holt-Winters forecasting.")
                    fig_forecast = None
                    
            except Exception as e:
                st.error(f"Error processing dates for forecasting: {e}. Make sure the Date column contains valid dates.")
                fig = None
                fig_forecast = None
        else:
            st.info("Select a Date column to view time series charts and forecasts.")
            # Simple Bar chart if no date
            agg_df = processed_df[[rev_col, exp_col]].sum().reset_index()
            agg_df.columns = ["Metric", "Total"]
            fig = px.bar(agg_df, x="Metric", y="Total", title="Total Revenue vs Expenditure", color="Metric")
            st.plotly_chart(fig, use_container_width=True)
            fig_forecast = None

        st.header("6. Download PDF Report")
        st.write("Click below to generate and download a comprehensive PDF report containing your analysis and insights.")
        
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(200, 10, txt="Data Insights & Forecast Report", ln=True, align="C")
                    pdf.ln(10)
                    
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
                    pdf.ln(5)
                    
                    # KPIs
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Key Performance Indicators", ln=True)
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"Total Revenue (Adj. {discount_option} discount): ${total_revenue:,.2f}", ln=True)
                    pdf.cell(200, 10, txt=f"Total Expenditure: ${total_expenditure:,.2f}", ln=True)
                    pdf.cell(200, 10, txt=f"Profit / Loss: ${profit_loss:,.2f}", ln=True)
                    if tx_col != "None":
                        pdf.cell(200, 10, txt=f"Avg. Transaction Value (ATV): ${atv_value:,.2f}", ln=True)
                    pdf.ln(5)
                    
                    # AI Suggestions
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="AI Suggestions", ln=True)
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, txt=ai_suggestion_text)
                    pdf.ln(5)
                    
                    # Add Charts (Requires Kaleido)
                    def add_plotly_to_pdf(fig_obj, pdf_obj, title):
                        if fig_obj is not None:
                            try:
                                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                                    fig_obj.write_image(tmpfile.name, engine="kaleido")
                                    pdf_obj.add_page()
                                    pdf_obj.set_font("Arial", 'B', 14)
                                    pdf_obj.cell(200, 10, txt=title, ln=True)
                                    pdf_obj.image(tmpfile.name, x=10, w=190)
                                    tmp_path = tmpfile.name
                                os.unlink(tmp_path)
                            except Exception as e:
                                pdf_obj.cell(200, 10, txt=f"Could not render {title} due to Image Generation Engine missing.", ln=True)
                                
                    if 'fig' in locals() and fig is not None:
                        add_plotly_to_pdf(fig, pdf, "Historical Data Chart")
                        
                    if 'fig_forecast' in locals() and fig_forecast is not None:
                        add_plotly_to_pdf(fig_forecast, pdf, f"Forecasting ({forecast_years} Years)")

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                        pdf.output(tmp_pdf.name)
                        tmp_pdf_path = tmp_pdf.name
                    with open(tmp_pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    os.unlink(tmp_pdf_path)
                    
                    st.success("PDF Generated Successfully!")
                    st.download_button(
                        label="⬇️ Download PDF Data Report",
                        data=pdf_bytes,
                        file_name="Data_Report.pdf",
                        mime="application/pdf"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred while generating the PDF: {e}")
                    st.info("Note: PDF Generation might fail on some limited cloud environments if 'kaleido' fails to install native dependencies.")
else:
    st.info("Awaiting file upload...")
    
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit. Ready for deployment on Streamlit Community Cloud without any virtual environment.*")
