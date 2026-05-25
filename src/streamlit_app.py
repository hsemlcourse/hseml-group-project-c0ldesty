import pandas as pd
import streamlit as st

from src.api import FINAL_THRESHOLD, PROCESSED_FE_PATH, _prepare_request_frame, get_model_bundle


TARGET_COLUMN = "FraudFound_P"


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    if not PROCESSED_FE_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset was not found: {PROCESSED_FE_PATH}. "
            "Run notebooks/01_eda.ipynb first."
        )
    return pd.read_csv(PROCESSED_FE_PATH)


@st.cache_resource
def load_model():
    return get_model_bundle()


def options_for(data: pd.DataFrame, column: str) -> list:
    return sorted(data[column].dropna().unique().tolist())


def select_value(data: pd.DataFrame, column: str, default):
    options = options_for(data, column)
    index = options.index(default) if default in options else 0
    return st.selectbox(column, options, index=index)


def build_record(data: pd.DataFrame) -> dict:
    with st.form("claim_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            month = select_value(data, "Month", "Dec")
            week_of_month = st.number_input("WeekOfMonth", min_value=1, max_value=5, value=5)
            day_of_week = select_value(data, "DayOfWeek", "Wednesday")
            make = select_value(data, "Make", "Honda")
            accident_area = select_value(data, "AccidentArea", "Urban")
            day_claimed = select_value(data, "DayOfWeekClaimed", "Tuesday")
            month_claimed = select_value(data, "MonthClaimed", "Jan")
            week_claimed = st.number_input("WeekOfMonthClaimed", min_value=1, max_value=5, value=1)
            sex = select_value(data, "Sex", "Female")
            marital_status = select_value(data, "MaritalStatus", "Single")

        with col2:
            age = st.number_input("Age", min_value=0, max_value=100, value=21)
            fault = select_value(data, "Fault", "Policy Holder")
            policy_type = select_value(data, "PolicyType", "Sport - Liability")
            vehicle_category = select_value(data, "VehicleCategory", "Sport")
            vehicle_price = select_value(data, "VehiclePrice", "more than 69000")
            deductible = st.number_input("Deductible", min_value=0, max_value=1000, value=300, step=100)
            driver_rating = st.number_input("DriverRating", min_value=1, max_value=4, value=1)
            days_policy_accident = select_value(data, "Days_Policy_Accident", "more than 30")
            days_policy_claim = select_value(data, "Days_Policy_Claim", "more than 30")
            past_claims = select_value(data, "PastNumberOfClaims", "none")

        with col3:
            age_vehicle = select_value(data, "AgeOfVehicle", "3 years")
            age_holder = select_value(data, "AgeOfPolicyHolder", "26 to 30")
            police_report = select_value(data, "PoliceReportFiled", "No")
            witness = select_value(data, "WitnessPresent", "No")
            agent_type = select_value(data, "AgentType", "External")
            supplements = select_value(data, "NumberOfSuppliments", "none")
            address_change = select_value(data, "AddressChange_Claim", "1 year")
            number_of_cars = select_value(data, "NumberOfCars", "3 to 4")
            year = st.number_input("Year", min_value=1994, max_value=1996, value=1994)
            base_policy = select_value(data, "BasePolicy", "Liability")

        submitted = st.form_submit_button("Predict fraud")

    record = {
        "Month": month,
        "WeekOfMonth": week_of_month,
        "DayOfWeek": day_of_week,
        "Make": make,
        "AccidentArea": accident_area,
        "DayOfWeekClaimed": day_claimed,
        "MonthClaimed": month_claimed,
        "WeekOfMonthClaimed": week_claimed,
        "Sex": sex,
        "MaritalStatus": marital_status,
        "Age": age,
        "Fault": fault,
        "PolicyType": policy_type,
        "VehicleCategory": vehicle_category,
        "VehiclePrice": vehicle_price,
        "Deductible": deductible,
        "DriverRating": driver_rating,
        "Days_Policy_Accident": days_policy_accident,
        "Days_Policy_Claim": days_policy_claim,
        "PastNumberOfClaims": past_claims,
        "AgeOfVehicle": age_vehicle,
        "AgeOfPolicyHolder": age_holder,
        "PoliceReportFiled": police_report,
        "WitnessPresent": witness,
        "AgentType": agent_type,
        "NumberOfSuppliments": supplements,
        "AddressChange_Claim": address_change,
        "NumberOfCars": number_of_cars,
        "Year": year,
        "BasePolicy": base_policy,
    }
    return record, submitted


def main() -> None:
    st.set_page_config(page_title="Fraud Detection", layout="wide")
    st.title("Vehicle Insurance Fraud Detection")
    st.caption("CatBoost model with validation threshold 0.33")

    try:
        data = load_reference_data()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    record, submitted = build_record(data)

    if submitted:
        try:
            bundle = load_model()
            request_frame = _prepare_request_frame([record], bundle.feature_columns)
            probability = float(bundle.pipeline.predict_proba(request_frame)[:, 1][0])
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            return

        prediction = int(probability >= FINAL_THRESHOLD)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Fraud probability", f"{probability:.3f}")
        metric_col2.metric("Threshold", f"{FINAL_THRESHOLD:.2f}")
        metric_col3.metric("Prediction", "Fraud" if prediction else "Not fraud")

        if prediction:
            st.warning("The claim is suspicious. It should be checked manually.")
        else:
            st.success("The claim is not marked as fraud by the model.")

        with st.expander("Request JSON"):
            st.json(record)


if __name__ == "__main__":
    main()
