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

st.title("Universal Data Insights & Forecast Dashboard")
st.markdown("""
This app automatically cleans your data, sums your records, calculates key metrics (Revenue, Expenditure, Profit/Loss, ATV), provides charts, performs Holt-Winters forecasting, and generates a downloadable PDF report.
""")

st.sidebar.header("Upload Data")
# Note: Streamlit limits file uploads by default to 200MB. To allow 500MB, 
# you MUST run streamlit with: streamlit run app.py --server.maxUploadSize=500
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel) - max 500MB via config", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Read data
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully.")
        
        # --- EDA Data Cleaning & Status ---
        st.header("1. Exploratory Data Analysis & Cleaning Summary")
        
        original_rows = len(df)
        original_nulls = df.isnull().sum().sum()
        original_dupes = df.duplicated().sum()
        
        # Cleaning operations
        df_cleaned = df.drop_duplicates()
        # Strategy: fill numeric nulls with 0 or drop them, for simplicity here we fill numeric with 0 
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(0)
        # Drop rows if they still have critical nulls in categorical
        df_cleaned = df_cleaned.dropna()
        
        final_rows = len(df_cleaned)
        final_nulls = df_cleaned.isnull().sum().sum()
        final_dupes = df_cleaned.duplicated().sum()
        
        st.markdown(f"""
        **Data Cleaning Complete:**
        - **Total Records:** Started with **{original_rows}** rows → Cleaned to **{final_rows}** rows.
        - **Missing Values:** Found **{original_nulls}** missing values → Resolved to **{final_nulls}**.
        - **Duplicates Removed:** Found **{original_dupes}** duplicate rows → Now **{final_dupes}**.
        """)
        
        # Assign df to the cleaned version for the rest of processing
        df = df_cleaned
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
        
    st.sidebar.header("Discount Settings")
    discount_option = st.sidebar.radio("Apply Discount to Revenue?", ["0%", "10%", "20%", "30%", "Custom"])
    
    if discount_option == "Custom":
        discount_value = st.sidebar.number_input("Enter Custom Discount (%)", min_value=0.0, max_value=100.0, value=0.0)
    else:
        discount_value = float(discount_option.strip("%"))
        
    st.sidebar.header("Forecast Settings")
    forecast_option = st.sidebar.radio("Forecast Horizon (Years)", ["1", "3", "5", "7", "Custom"])
    if forecast_option == "Custom":
        forecast_years = st.sidebar.number_input("Enter Forecast Years", min_value=1, max_value=20, value=5)
    else:
        forecast_years = int(forecast_option)

    st.header("2. Key Performance Indicators (KPIs)")
    
    # Auto-detect columns based on common names or keyword matching
    def find_column(df, keywords):
        for col in df.columns:
            if any(keyword.lower() in str(col).lower() for keyword in keywords):
                return col
        return None

    # Try to automatically find Date, Revenue, Expenditure, and Transactions
    date_col = find_column(df, ['date', 'time', 'day', 'month', 'year'])
    rev_col = find_column(df, ['rev', 'sales', 'income', 'earn'])
    exp_col = find_column(df, ['exp', 'cost', 'spend', 'budget', 'out'])
    tx_col = find_column(df, ['tx', 'trans', 'qty', 'quantity', 'count', 'order'])
    
    # Fallbacks if columns can't be guessed by names
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if not rev_col and len(numeric_columns) > 0: rev_col = numeric_columns[0]
    if not exp_col and len(numeric_columns) > 1: exp_col = numeric_columns[1]
    if not tx_col and len(numeric_columns) > 2: tx_col = numeric_columns[2]
    
    if rev_col and exp_col:
        # Data preparation
        processed_df = df.copy()
        
        # Apply discount
        discount_multiplier = 1 - (discount_value / 100.0)
        processed_df['Adjusted_Revenue'] = processed_df[rev_col] * discount_multiplier
        
        total_revenue = processed_df['Adjusted_Revenue'].sum()
        total_expenditure = processed_df[exp_col].sum()
        profit_loss = total_revenue - total_expenditure
        
        atv_value = 0
        if tx_col:
            total_tx = processed_df[tx_col].sum()
            if total_tx > 0:
                atv_value = total_revenue / total_tx
                
        # Metrics Display (KPI cards only)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Revenue (Adj.)", f"Rs. {total_revenue:,.2f}")
        m2.metric("Total Budget/Expenditure", f"Rs. {total_expenditure:,.2f}")
        m3.metric("Total Profit / Loss", f"Rs. {profit_loss:,.2f}", delta="Profit" if profit_loss > 0 else "Loss", delta_color="normal" if profit_loss > 0 else "inverse")
        if tx_col:
            m4.metric("Avg. Transaction Value (ATV)", f"Rs. {atv_value:,.2f}")
            
        st.header("3. AI Recommendations")
        if profit_loss > 0:
            st.success("AI Analysis: Operations are highly profitable. No critical cost-reduction actions required at this time.")
            ai_suggestion_text = "Operations are highly profitable. No critical cost-reduction actions required at this time."
        else:
            st.warning("AI Analysis: Operations are currently running at a deficit. Recommendation: Audit operational expenditures, optimize product discounts, and identify underperforming segments.")
            ai_suggestion_text = "Operations are currently running at a deficit. Recommendation: Audit operational expenditures, optimize product discounts, and identify underperforming segments."

        st.header("4. Visualizations & Forecasting")
        
        # Aggregating by date if available
        if date_col:
            try:
                processed_df[date_col] = pd.to_datetime(processed_df[date_col])
                processed_df = processed_df.sort_values(by=date_col)
                # Aggregate by month for simpler processing
                ts_df = processed_df.set_index(date_col).resample('M').sum(numeric_only=True).reset_index()
                
                # Plotly Chart 1: Line Chart
                fig_line = px.line(ts_df, x=date_col, y=['Adjusted_Revenue', exp_col], 
                              title="Revenue and Expenditure Over Time",
                              labels={"value": "Amount (Rs.)", "variable": "Metric"})
                st.plotly_chart(fig_line, use_container_width=True)
                
                # Plotly Chart 2: Bar Chart 
                fig_bar_dates = px.histogram(ts_df, x=date_col, y=['Adjusted_Revenue', exp_col], barmode='group',
                                       title="Monthly Breakdown (Revenue vs Expenditure)",
                                       labels={"value": "Amount (Rs.)", "variable": "Metric"})
                st.plotly_chart(fig_bar_dates, use_container_width=True)

                # Plotly Chart 3: Area Chart
                ts_df['Profit'] = ts_df['Adjusted_Revenue'] - ts_df[exp_col]
                fig_area = px.area(ts_df, x=date_col, y='Profit', title="Profit/Loss Trend Over Time", labels={'Profit': 'Amount (Rs.)'})
                st.plotly_chart(fig_area, use_container_width=True)
                
                # Holt Winters Forecasting
                st.subheader(f"Holt-Winters Forecasting ({forecast_years} Years)")
                
                if len(ts_df) >= 24:
                    ts_data = ts_df['Adjusted_Revenue'].values
                    periods_to_forecast = forecast_years * 12
                    
                    try:
                        model = ExponentialSmoothing(ts_data, trend="add", seasonal="add", seasonal_periods=12).fit()
                        forecast = model.forecast(periods_to_forecast)
                        
                        last_date = ts_df[date_col].iloc[-1]
                        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods_to_forecast, freq='M')
                        
                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Forecasted_Revenue': forecast
                        })
                        
                        fig_forecast = go.Figure()
                        fig_forecast.add_trace(go.Scatter(x=ts_df[date_col], y=ts_df['Adjusted_Revenue'], mode='lines', name='Historical Revenue', line=dict(color='blue')))
                        fig_forecast.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted_Revenue'], mode='lines', name='Forecast', line=dict(color='orange', dash='dash')))
                        fig_forecast.update_layout(title=f"Revenue Forecast for Next {forecast_years} Years", xaxis_title="Date", yaxis_title="Revenue (Rs.)")
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Not enough variance to perform seasonal Holt-Winters forecasting. Error: {e}")
                        fig_forecast = None
                else:
                    st.info("Need at least 24 months of historical data for reliable seasonal forecasting.")
                    fig_forecast = None
                    
            except Exception as e:
                st.error(f"Error processing dates for charts: {e}.")
                fig_line, fig_bar_dates, fig_area, fig_forecast = None, None, None, None
        else:
            st.info("No usable Date column auto-detected. Showing summary charts instead.")
            agg_df = processed_df[[rev_col, exp_col]].sum().reset_index()
            agg_df.columns = ["Metric", "Total"]
            fig_bar_dates = px.bar(agg_df, x="Metric", y="Total", title="Total Revenue vs Expenditure", color="Metric")
            st.plotly_chart(fig_bar_dates, use_container_width=True)
            
            fig_pie = px.pie(agg_df, names="Metric", values="Total", title="Proportion of Revenue and Expenditure")
            st.plotly_chart(fig_pie, use_container_width=True)

            fig_line, fig_area, fig_forecast = None, None, None

        st.header("5. Download PDF Report")
        
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
                    
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Key Performance Indicators", ln=True)
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"Total Revenue (Adj.): Rs. {total_revenue:,.2f}", ln=True)
                    pdf.cell(200, 10, txt=f"Total Budget / Expenditure: Rs. {total_expenditure:,.2f}", ln=True)
                    pdf.cell(200, 10, txt=f"Total Profit / Loss: Rs. {profit_loss:,.2f}", ln=True)
                    if tx_col:
                        pdf.cell(200, 10, txt=f"Avg. Transaction Value: Rs. {atv_value:,.2f}", ln=True)
                    pdf.ln(5)
                    
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="AI Recommendations", ln=True)
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, txt=ai_suggestion_text)
                    pdf.ln(5)
                    
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
                            except:
                                pass 
                                
                    if 'fig_line' in locals() and fig_line: add_plotly_to_pdf(fig_line, pdf, "Revenue vs Expenditure Over Time")
                    if 'fig_bar_dates' in locals() and fig_bar_dates: add_plotly_to_pdf(fig_bar_dates, pdf, "Revenue / Expenditure Breakdown")
                    if 'fig_area' in locals() and fig_area: add_plotly_to_pdf(fig_area, pdf, "Profit Over Time")
                    if 'fig_forecast' in locals() and fig_forecast: add_plotly_to_pdf(fig_forecast, pdf, f"Forecasting ({forecast_years} Years)")

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                        pdf.output(tmp_pdf.name)
                        tmp_pdf_path = tmp_pdf.name
                    with open(tmp_pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    os.unlink(tmp_pdf_path)
                    
                    st.success("PDF Generated Successfully.")
                    st.download_button(label="Download PDF Data Report", data=pdf_bytes, file_name="Data_Report.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
    else:
        st.error("Could not determine Revenue or Expenditure columns.")
else:
    st.info("Awaiting file upload.")
