import streamlit as st
import pandas as pd
import plotly.express as px
import joblib  # For loading the trained model
from sklearn.preprocessing import RobustScaler, LabelEncoder

st.set_page_config(page_title="Churn Prediction App", page_icon=":chart_with_downwards_trend:", layout="wide")
st.title(" :chart_with_downwards_trend: Predict Customer Churn")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Upload the file for analysis
fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))

# Load the pre-trained model
model_path = "random_forest_model.pkl"
model = None

# Try loading the model if it exists in the current directory
try:
    model = joblib.load(model_path)
    st.success(f"Model '{model_path}' loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file '{model_path}' not found. Please ensure the model is in the same directory.")
    st.stop()

if fl is not None:
    df = pd.read_csv(fl, encoding="ISO-8859-1")
    st.write("Uploaded File Preview:")
    st.dataframe(df)

    # Ask if the first column needs to be deleted
    delete_first_col = st.radio("Do you want to delete the first column?", options=["Yes", "No"])

    if delete_first_col == "Yes":
        df = df.drop(df.columns[0], axis=1)
        st.write("First column has been deleted.")

    if 'Churn' in df.columns:
        st.write("Unique values in 'Churn' column:", df['Churn'].unique())
    else:
        st.error("The dataset must contain a 'Churn' column.")
        st.stop()

    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0, True: 1, False: 0})
    df = df.dropna(subset=['Churn'])

    df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')

    df_churned = df[df['Churn'] == 1]
    df_not_churned = df[df['Churn'] == 0]
    # Analyze Categorical Columns
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if categorical_cols:
        st.subheader("Analyze Categorical Columns with Respect to Churn")
        
        churn_status = st.radio("Select Customer Status", options=["Churned", "Not Churned"], horizontal=True)
        selected_col = st.selectbox("Select a Column for Analysis", options=categorical_cols)

        filtered_df = df_churned if churn_status == "Churned" else df_not_churned

        if not filtered_df[selected_col].dropna().value_counts().empty:
            # Add a slider for controlling pie chart size
            chart_size = st.slider(
                f"Adjust Pie Chart Size for {selected_col}", min_value=300, max_value=800, value=500, step=50
            )

            fig = px.pie(
                filtered_df,
                names=selected_col,
                title=f"{selected_col} Distribution ({churn_status})"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                width=chart_size,
                height=chart_size
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No valid data for '{selected_col}' in selected customer status.")
    else:
        st.warning("No categorical columns found in the dataset.")

    # Analyze Numerical Columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if numerical_cols:
        st.subheader("Analyze Numerical Columns with Respect to Churn")
        
        selected_num_col = st.selectbox("Select a Numerical Column", options=numerical_cols)
        plot_type = st.radio("Select Plot Type", options=["Histogram", "Box Plot", "Violin Plot"], horizontal=True)
        
        if plot_type == "Histogram":
            # Add customization options for the histogram
            bin_size = st.slider("Bin Size", min_value=1, max_value=100, value=10, step=1)
            opacity = st.slider("Bar Opacity", min_value=0.1, max_value=1.0, value=0.75, step=0.05)
            marginal = st.selectbox("Select Marginal Plot", options=[None, "rug", "box", "violin"], index=0)

            # Histogram for numerical values with respect to churn
            fig = px.histogram(
                df,
                x=selected_num_col,
                color="Churn",
                barmode="overlay",
                title=f"Histogram of {selected_num_col} by Churn",
                labels={"Churn": "Churn Status"},
                nbins=bin_size,
                opacity=opacity,
                marginal=marginal  # Add marginal plots like rug, box, or violin
            )
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Box Plot":
            # Box plot for numerical values with respect to churn
            fig = px.box(
                df,
                x="Churn",
                y=selected_num_col,
                color="Churn",
                title=f"Box Plot of {selected_num_col} by Churn",
                labels={"Churn": "Churn Status"}
            )
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Violin Plot":
            # Violin plot for numerical values with respect to churn
            fig = px.violin(
                df,
                x="Churn",
                y=selected_num_col,
                color="Churn",
                box=True,  # Add a box plot inside the violin plot
                title=f"Violin Plot of {selected_num_col} by Churn",
                labels={"Churn": "Churn Status"}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numerical columns found in the dataset.")



    # Collect user input in the same format as the dataset
    st.write("Enter values for prediction:")

    # Prepare the list of columns from the uploaded dataframe (excluding 'Churn')
    input_data = {}
    columns = [col for col in df.columns if col not in ['Churn', 'PhoneService']]  # Remove 'Churn' column from user input

    # Initialize LabelEncoder and store encoders for categorical columns
    label_encoders = {}

    for col in columns:
        if df[col].dtype == 'object':  # For categorical columns
            # Fit LabelEncoder on the training data (df[col]) and store the encoder
            le = LabelEncoder()
            le.fit(df[col].dropna())  # Fit on the original data
            label_encoders[col] = le
            input_data[col] = st.selectbox(f"Select {col}", options=df[col].dropna().unique())
        elif df[col].dtype == 'float64' or df[col].dtype == 'int64':  # For numerical columns
            input_data[col] = st.number_input(f"Enter value for {col}", value=0.0, step=1.0)

    # Create a DataFrame from user input
    user_df = pd.DataFrame([input_data])

    # Apply label encoding for categorical variables using stored encoders
    for col in user_df.columns:
        if col in label_encoders:  # Only encode columns that were previously encoded
            user_df[col] = label_encoders[col].transform(user_df[col])

    # Apply RobustScaler for numerical features
    scaler = RobustScaler()

    # Ensure that the numerical columns are the same as the model expects
    numerical_cols = user_df.select_dtypes(include=['float64', 'int64']).columns
    user_df[numerical_cols] = scaler.fit_transform(user_df[numerical_cols])

    # Prediction on user input
    if st.button("Predict Churn"):
        try:
            # Make prediction using the pre-trained model
            predictions = model.predict(user_df)
            user_df['Predicted_Churn'] = predictions
            predicted_churn = user_df['Predicted_Churn'].map({0: 'Not Churn', 1: 'Churn'}).iloc[0]
            # Display the user input and prediction result
            st.write("Prediction Results:")            
            st.write({predicted_churn})  # Display the table with input data and predicted churn

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

else:
    st.info("Please upload a file to get started.")
