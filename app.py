import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datetime
from datetime import date, timedelta
import random
import time
import hashlib  # For password hashing
import uuid  # For generating unique user IDs
import base64 # For encoding the local background image
import sqlite3  # Import the SQLite3 library

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Health AI Assistant", page_icon="⚕")


# --- Function to add background image from a local file ---
def add_bg_from_local(image_file):
    """
    Sets a local image as the background for the Streamlit app.

    Args:
        image_file (str): The path to the local image file.
    """
    try:
        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()
        st.markdown(
             f"""
             <style>
             [data-testid="stAppViewContainer"] {{
                 background-image: url("data:image.jpg;base64,{encoded_string}");
                 background-attachment: fixed;
                 background-size: cover;
             }}
             [data-testid="stHeader"] {{
                 background-color: rgba(0, 0, 0, 0);
             }}
             /* MODIFIED: Increased opacity (decreased transparency) from 0.85 to 0.95 */
             .st-emotion-cache-18ni7ap {{
                 background-color: rgba(255, 255, 255, 0.95); /* More opaque white for main content */
                 border-radius: 10px;
                 padding: 20px;
             }}
             /* MODIFIED: Increased opacity (decreased transparency) from 0.85 to 0.95 */
             [data-testid="stSidebar"] > div:first-child {{
                  background-color: rgba(240, 242, 246, 0.95); /* More opaque sidebar background */
             }}
             </style>
             """,
             unsafe_allow_html=True
         )
    except FileNotFoundError:
        st.error(f"Background image '{image_file}' not found. Please ensure the image is in the same directory as the script.")


# --- User Authentication System with SQLite ---
DB_FILE = "health_app.db"

def init_db():
    """Initializes the SQLite database and creates the users table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Create a table for users
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_uuid TEXT NOT NULL UNIQUE,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        email TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    """Register a new user in the SQLite database."""
    if not username or not password or not email:
        return False, "All fields are required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if '@' not in email or '.' not in email:
        return False, "Please enter a valid email address."

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return False, "Username already exists. Please choose another."

        user_id = str(uuid.uuid4())
        hashed_pass = hash_password(password)
        cursor.execute(
            "INSERT INTO users (user_uuid, username, password_hash, email) VALUES (?, ?, ?, ?)",
            (user_id, username, hashed_pass, email)
        )
        conn.commit()
        return True, "Registration successful! Please log in."
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False, f"An error occurred during registration: {e}"
    finally:
        conn.close()

def authenticate(username, password):
    """Authenticate a user against the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        result = cursor.fetchone() # Fetch one record

        if result:
            stored_hash = result[0]
            if hash_password(password) == stored_hash:
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.session_state.login_error = ""
                return True
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        st.session_state.login_error = "A database error occurred."
        return False
    finally:
        conn.close()

    st.session_state.authenticated = False
    st.session_state.login_error = "Invalid username or password."
    return False

def logout():
    """Logs the user out by resetting authentication state and clearing data."""
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.current_page = 'Home'
    st.session_state.login_error = ""
    st.session_state.appointments_page_message = None
    st.session_state.rescheduling_appointment_id = None
    st.session_state.pop("new_reschedule_date_input_tab_v6_value", None)
    _clear_symptoms_form_callback()
    st.session_state.health_topic_query_input_val_v5 = ""
    st.session_state.health_topic_overview_result_v5 = None
    st.session_state.health_topic_overview_query_processed_v5 = None
    st.session_state.appointment_reason = ""
    st.session_state.pop("appt_date_input_book_tab_v6_value", None)
    st.session_state.pop("appt_doc_select_book_tab_v6", None)
    st.session_state.pop("appt_time_select_book_tab_v6", None)
    st.session_state.pop("appt_reason_text_book_tab_v6", None)
    # Clear meal plan state on logout
    st.session_state.meal_plan_result = None
    st.session_state.meal_plan_details_submitted = None
    st.rerun()


# --- Global Variables and Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """Loads the AI model and tokenizer."""
    model_name = "ibm-granite/granite-3.3-2b-instruct"
    print(f"Attempting to load model: {model_name} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == 'cuda' else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)
        if tokenizer.pad_token_id is None:
             tokenizer.pad_token_id = tokenizer.eos_token_id
             if tokenizer.pad_token_id is None:
                 print("Warning: EOS token not found, cannot set pad_token_id.")
        print(f"Model '{model_name}' loaded successfully on {device}!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        st.error(f"Critical Error: AI model '{model_name}' could not be loaded. {e}. AI-dependent features will be unavailable.", icon="❌")
        return None, None

tokenizer, model = load_model()


# --- Helper Functions for AI Generation ---
def get_llm_symptom_analysis(symptoms_text):
    """
    Generates a longer, more detailed AI analysis of symptoms, including self-care,
    OTC suggestions, and when to see a doctor.
    """
    if model is None or tokenizer is None:
        return "AI analysis is currently unavailable because the required model could not be loaded."
    if tokenizer.pad_token_id is None:
        return "AI analysis unavailable: Model tokenizer is missing a pad token."
    
    # MODIFIED: Removed the mandatory disclaimer from the prompt
    prompt = f"""
Provide a detailed and structured analysis for the following patient symptoms. The response should be comprehensive and broken down into the following sections:

1.  General Symptom Analysis: Briefly interpret the cluster of symptoms provided.
2.  Possible Self-Care Measures: Suggest non-medicinal, general self-care actions (e.g., rest, hydration, etc.).
3.  Potential Over-the-Counter (OTC) Medication Options: List common OTC medications that could address the specific symptoms. For each, explain its primary use (e.g., Ibuprofen for pain relief and fever reduction). Group them by purpose if applicable.
4.  When to Consult a Doctor: Provide clear examples of conditions under which the user should seek professional medical help (e.g., if symptoms worsen, last more than a week, or if severe symptoms like difficulty breathing appear).

Symptoms: {symptoms_text}
Detailed Analysis:"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generate_kwargs = dict(max_new_tokens=1000, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], **generate_kwargs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        prompt_end_text = "Detailed Analysis:"
        prompt_start_index = generated_text.rfind(prompt_end_text)
        analysis = generated_text[prompt_start_index + len(prompt_end_text):].strip() if prompt_start_index != -1 else generated_text.strip()
        
        # MODIFIED: Removed the safety check for the disclaimer.
        return analysis if analysis.strip() else "No specific analysis or suggestions could be generated for these symptoms at this time."
    except Exception as e:
        print(f"An error occurred during AI generation (analysis): {e}")
        if device.type == 'cuda': torch.cuda.empty_cache()
        return "An error occurred while generating AI analysis. Please try again or check the logs."


def get_llm_health_topic_info(topic_query):
    if model is None or tokenizer is None: return "AI information is currently unavailable because the required model could not be loaded."
    if tokenizer.pad_token_id is None: return "AI information unavailable: Model tokenizer is missing a pad token."
    prompt = f"Provide a general overview of the health topic: {topic_query}. Focus on basic information.\nOverview:"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generate_kwargs = dict(max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
        with torch.no_grad(): outputs = model.generate(inputs["input_ids"], **generate_kwargs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_start_index = generated_text.rfind(prompt)
        info = generated_text[prompt_start_index + len(prompt):].strip() if prompt_start_index != -1 else generated_text.strip()
        if info.startswith("Overview:"): info = info[len("Overview:"):].strip()
        return info if info.strip() else "Could not generate information for this topic at this time."
    except Exception as e:
        print(f"An error occurred during AI generation for health topic: {e}")
        if device.type == 'cuda': torch.cuda.empty_cache()
        return "An error occurred while generating information for this topic. Please try again or check the logs."

def get_llm_meal_plan(details):
    """
    Generates a sample 1-day meal plan based on user inputs.
    """
    if model is None or tokenizer is None:
        return "Meal plan generation is currently unavailable because the AI model could not be loaded."
    if tokenizer.pad_token_id is None:
        return "Meal plan generation is unavailable: Model tokenizer is missing a pad token."

    # MODIFIED: Removed the mandatory disclaimer from the prompt
    prompt = f"""
Create a sample, one-day meal plan for a user with the following personal details and requirements.
The plan should be appropriate for the user's age, gender, and weight.
Structure the response with headings for "Breakfast", "Lunch", "Dinner", and "Snacks". For each meal, provide a simple meal idea.

User's Personal Details:
- Age: {details.get('age', 'Not specified')}
- Gender: {details.get('gender', 'Not specified')}
- Weight: {details.get('weight', 'Not specified')} kg

User's Dietary Requirements:
- Dietary Preference: {details.get('diet_pref', 'Not specified')}
- Primary Health Goal: {details.get('goal', 'Not specified')}
- Foods to Avoid (Allergies/Dislikes): {details.get('allergies', 'None specified')}
- Target Daily Calories (Approximate): {details.get('calories', 'Not specified')}
- Number of Meals: {details.get('num_meals', '3 meals, 1 snack')}
- Other Instructions: {details.get('other', 'None')}

Sample Meal Plan:
"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generate_kwargs = dict(max_new_tokens=1000, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], **generate_kwargs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        prompt_end_text = "Sample Meal Plan:"
        prompt_start_index = generated_text.rfind(prompt_end_text)
        plan = generated_text[prompt_start_index + len(prompt_end_text):].strip() if prompt_start_index != -1 else generated_text.strip()

        # MODIFIED: Removed the safety check for the disclaimer.
        return plan if plan.strip() else "No meal plan could be generated for these requirements at this time."
    except Exception as e:
        print(f"An error occurred during AI generation (meal plan): {e}")
        if device.type == 'cuda': torch.cuda.empty_cache()
        return "An error occurred while generating the meal plan. Please try again or check the logs."


# --- Placeholder/Example Data ---
COMMON_SYMPTOMS = {
    "General": ["Fever", "Fatigue", "Chills", "Sweating", "Body aches"],
    "Respiratory": ["Cough", "Shortness of breath", "Sore throat", "Runny nose", "Nasal congestion", "Difficulty breathing"],
    "Gastrointestinal": ["Nausea", "Vomiting", "Diarrhea", "Abdominal pain"],
    "Pain": ["Headache", "Chest pain", "Muscle pain", "Joint pain"],
    "Other": ["Rash", "Dizziness", "Loss of taste or smell"]
}

# --- Session State Management for Navigation and Data ---
def set_page(page_name):
    """Helper to change page and rerun."""
    st.session_state.current_page = page_name
    st.rerun()

def _clear_symptoms_form_callback():
    """Callback function for clearing symptoms form state."""
    for category, sym_list in COMMON_SYMPTOMS.items():
        for symptom in sym_list:
            checkbox_key = f"cb_symp_{category}{symptom.replace(' ', '').replace('-', '').replace('/', '').lower()}_v6"
            if checkbox_key in st.session_state:
                st.session_state[checkbox_key] = False
    st.session_state["ta_custom_symptoms"] = ""
    st.session_state["sel_duration"] = "Today"
    st.session_state["slider_severity"] = "Mild"
    st.session_state.symptom_analysis_done = False
    st.session_state.symptom_results = None
    st.session_state.current_symptoms_text = ""


# --- Initialize Session State (BEFORE any conditional logic based on auth) ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'current_user' not in st.session_state: st.session_state.current_user = None
if 'login_error' not in st.session_state: st.session_state.login_error = ""
if 'registration_status' not in st.session_state: st.session_state.registration_status = {"success": None, "message": ""}
if 'current_page' not in st.session_state: st.session_state.current_page = 'Home'
if 'current_symptoms_text' not in st.session_state: st.session_state.current_symptoms_text = ""
if 'symptom_analysis_done' not in st.session_state: st.session_state.symptom_analysis_done = False
if 'symptom_results' not in st.session_state: st.session_state.symptom_results = None
if 'symptom_analysis_history' not in st.session_state: st.session_state.symptom_analysis_history = []
if 'alerts' not in st.session_state: st.session_state.alerts = []
if 'appointment_reason' not in st.session_state: st.session_state.appointment_reason = ""
if 'appointments' not in st.session_state: st.session_state.appointments = []
if 'profile' not in st.session_state: st.session_state.profile = {}
if 'appointments_page_message' not in st.session_state: st.session_state.appointments_page_message = None
if 'rescheduling_appointment_id' not in st.session_state: st.session_state.rescheduling_appointment_id = None
if 'new_reschedule_date' not in st.session_state: st.session_state.new_reschedule_date = date.today() + timedelta(days=1)
for category, sym_list in COMMON_SYMPTOMS.items():
    for symptom in sym_list:
        checkbox_key = f"cb_symp_{category}{symptom.replace(' ', '').replace('-', '').replace('/', '').lower()}"
        if checkbox_key not in st.session_state: st.session_state[checkbox_key] = False
if "ta_custom_symptoms" not in st.session_state: st.session_state["ta_custom_symptoms"] = ""
if "sel_duration" not in st.session_state: st.session_state["sel_duration"] = "Today"
if "slider_severity" not in st.session_state: st.session_state["slider_severity"] = "Mild"
if "health_topic_query_input_val_v5" not in st.session_state: st.session_state.health_topic_query_input_val_v5 = ""
if "health_topic_overview_result_v5" not in st.session_state: st.session_state.health_topic_overview_result_v5 = None
if "health_topic_overview_query_processed_v5" not in st.session_state: st.session_state.health_topic_overview_query_processed_v5 = None
if 'meal_plan_result' not in st.session_state: st.session_state.meal_plan_result = None
if 'meal_plan_details_submitted' not in st.session_state: st.session_state.meal_plan_details_submitted = None


# --- Login Page Function ---
def login_page():
    st.title("⚕ Health AI Assistant")
    st.markdown("""<style>[data-testid="stMain"] h1, [data-testid="stMain"] h3, [data-testid="stMain"] p, [data-testid="stMain"] li {color: white;} [data-testid="stTabs"] .st-emotion-cache-1s4o3vj {color: white;} [data-testid="stTabs"] .st-emotion-cache-1s4o3vj[aria-selected="true"] {color: black;} [data-testid="stMain"] div[data-testid="stCaptionContainer"] {color: #f0f2f6 !important;} div[data-testid="stFormSubmitButton"] button p {color: black !important;} div[data-testid="stTextInput"] {max-width: 450px;} div[data-testid="stTextInput"] input::placeholder {font-size: 14px;color: #a9a9a9;}</style>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login to Your Account")
        with st.form("login_form_v2"):
            username = st.text_input("Username", placeholder="Enter your username", key="login_username_input_v2")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password_input_v2")
            login_button = st.form_submit_button("Login")

            if login_button:
                if authenticate(username, password):
                    st.success("Login successful! Redirecting to dashboard...", icon="✅")
                    time.sleep(0.5)
                    st.rerun()

        if st.session_state.login_error:
            st.error(st.session_state.login_error, icon="❌")

    with tab2:
        st.subheader("Create a New Account")
        with st.form("registration_form_v1"):
            new_username = st.text_input("Choose a Username", placeholder="e.g., healthuser123", key="reg_username_input_v1")
            new_email = st.text_input("Email Address", placeholder="your.email@example.com", key="reg_email_input_v1")
            new_password = st.text_input("Create Password", type="password", placeholder="Must be at least 8 characters", key="reg_password_input_v1")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password", key="reg_confirm_password_v1")

            register_button = st.form_submit_button("Register")

            if register_button:
                if new_password != confirm_password:
                    st.error("Passwords do not match!", icon="❌")
                else:
                    success, message = register_user(new_username, new_password, new_email)
                    if success:
                        st.success(message, icon="✅")
                        st.session_state.registration_status = {"success": True, "message": message}
                    else:
                        st.error(message, icon="❌")
                        st.session_state.registration_status = {"success": False, "message": message}

        if st.session_state.registration_status["success"] is not None:
            if st.session_state.registration_status["success"]:
                st.success(st.session_state.registration_status["message"], icon="✅")
            else:
                st.error(st.session_state.registration_status["message"], icon="❌")

            st.session_state.registration_status = {"success": None, "message": ""}

    st.markdown("---")
    st.markdown("### About Health AI Assistant")
    st.markdown("""
    This application provides AI-powered health information and symptom analysis.
    
    * Symptom analysis
    * Health resources
    * Appointment scheduling
    * Health profile management
    """)
    st.markdown("---")
    st.caption("© 2025 Health AI Assistant. All rights reserved.")


# --- Page Functions ---
def home_page():
    st.title("🏠 Health AI Assistant Home")
    st.write(f"Welcome back, {st.session_state.current_user}! Use the sidebar to navigate.")
    st.write("This application provides AI-powered tools to help you manage your health information and get general insights.")
    st.markdown("---")
    st.subheader("Explore Features:")
    st.markdown("- Symptom Analysis: Describe your symptoms and get general AI-based insights.\n- **Meal Plan Generator: Get a sample AI-generated meal plan based on your preferences.\n- **My Health Data: View your past symptom analyses and appointment history.\n- **Appointments: Schedule new appointments and see upcoming/past ones.\n- **Resources: Explore health topics using our AI-powered tool.")
    st.markdown("---")
    st.write("Select a page from the sidebar to get started.")

    if st.session_state.alerts:
        st.subheader("Active Alerts")
        unread_alerts = [alert for alert in st.session_state.alerts if not alert.get('read', False)]
        if unread_alerts:
            for i, alert_item in enumerate(unread_alerts):
                alert_id_key_part = alert_item.get('id', f"alert_{i}")
                alert_ts_key_part = alert_item.get('timestamp', datetime.datetime.now()).strftime('%Y%m%d%H%M%S%f')
                dismiss_key = f"dismiss_alert_home_v6_{alert_id_key_part}_{alert_ts_key_part}"

                col1_alert, col2_alert = st.columns([0.9, 0.1])
                with col1_alert:
                    if alert_item['type'] == 'critical': st.error(f"❗ {alert_item['message']}")
                    elif alert_item['type'] == 'warning': st.warning(f"⚠ {alert_item['message']}")
                    elif alert_item['type'] == 'success': st.success(f"✅ {alert_item['message']}")
                    elif alert_item['type'] == 'info': st.info(f"ℹ {alert_item['message']}")
                with col2_alert:
                    if st.button("Dismiss", key=dismiss_key, help="Mark this alert as read"):
                        for alert_in_session in st.session_state.alerts:
                             if alert_in_session.get('id') == alert_item.get('id') and \
                                (isinstance(alert_in_session.get('timestamp'), datetime.datetime) and isinstance(alert_item.get('timestamp'), datetime.datetime) and alert_in_session['timestamp'] == alert_item['timestamp']) and \
                                not alert_in_session.get('read', False):
                                 alert_in_session['read'] = True
                                 break
                        st.rerun()
        else: st.info("No new active alerts.")
    else: st.info("No active alerts.")


def symptom_checker_page():
    st.header("🩺 AI-Powered Symptom Checker")
    st.write("Describe your current symptoms for a general AI-based analysis.")
    # MODIFIED: Removed disclaimer warning.
    st.subheader("Describe Your Current Symptoms")

    with st.form("symptom_form_page_v6"): 
        selected_symptoms = []
        for category, sym_list in COMMON_SYMPTOMS.items():
            with st.expander(f"{category} Symptoms"):
                for symptom in sym_list:
                    checkbox_key = f"cb_symp_{category}{symptom.replace(' ', '').replace('-', '').replace('/', '').lower()}_v6"
                    if st.checkbox(symptom, key=checkbox_key, value=st.session_state.get(checkbox_key, False)):
                        selected_symptoms.append(symptom)

        custom_symptoms = st.text_area(
            "Describe additional symptoms (comma-separated):",
            height=100,
            key="ta_custom_symptoms_page_v6",
            value=st.session_state.get("ta_custom_symptoms", "")
        )

        duration_options = ["Today", "Yesterday", "2-3 days ago", "4-7 days ago", "More than a week ago", "Unsure"]
        symptom_duration_val = st.session_state.get("sel_duration", "Today")
        symptom_duration_index = duration_options.index(symptom_duration_val) if symptom_duration_val in duration_options else 0
        symptom_duration = st.selectbox(
            "Symptom duration:",
            duration_options,
            key="sel_duration_page_v6",
            index=symptom_duration_index
        )

        severity_options = ["Mild", "Moderate", "Severe", "Very Severe"]
        symptom_severity_val = st.session_state.get("slider_severity", "Mild")
        symptom_severity = st.select_slider(
            "Overall severity:",
            options=severity_options,
            key="slider_severity_page_v6",
            value=symptom_severity_val
        )
        submit_symptoms_button = st.form_submit_button("Analyze My Current Symptoms")

    if submit_symptoms_button:
        st.session_state.ta_custom_symptoms = custom_symptoms
        st.session_state.sel_duration = symptom_duration
        st.session_state.slider_severity = symptom_severity

        current_selected_symptoms = []
        for cat, s_list in COMMON_SYMPTOMS.items():
            for s in s_list:
                cb_k = f"cb_symp_{cat}{s.replace(' ', '').replace('-', '').replace('/', '').lower()}_v6"
                if st.session_state.get(cb_k, False):
                    current_selected_symptoms.append(s)

        final_symptoms_list = list(set(current_selected_symptoms + [s.strip() for s in custom_symptoms.split(',') if s.strip()]))
        full_symptom_description = f"Selected: {', '.join(final_symptoms_list) if final_symptoms_list else 'None'}. Custom: {custom_symptoms if custom_symptoms.strip() else 'None'}. Duration: {symptom_duration}. Severity: {symptom_severity}."

        st.session_state.current_symptoms_text = full_symptom_description
        st.session_state.symptom_analysis_done = True

        with st.spinner("HealthAI is analyzing your current symptoms..."):
            analysis = get_llm_symptom_analysis(full_symptom_description)

        current_urgency, current_potential_conditions = "Monitor symptoms. Consult a doctor if symptoms persist or worsen.", ["General considerations based on symptoms (based on rule)."]
        current_lower_desc, current_severity_lower = full_symptom_description.lower(), symptom_severity.lower()
        if "very severe" in current_severity_lower or "severe" in current_severity_lower:
            if any(phrase in current_lower_desc for phrase in ["chest pain", "shortness of breath", "difficulty breathing"]):
                current_urgency, current_potential_conditions = "CRITICAL: Seek immediate medical attention for severe chest pain or breathing issues.", ["Potential serious cardiac/respiratory issue (based on rule)."]
                alert_msg = "CRITICAL SYMPTOM: Severe chest pain/breathing difficulty reported. Seek emergency care."
                alert_id_crit = str(datetime.datetime.now().timestamp()) + "_symptom_critical_v6"
                if not any(a.get('id') == alert_id_crit and not a.get('read', False) for a in st.session_state.alerts):
                     st.session_state.alerts.append({"id": alert_id_crit, "type": "critical", "message": alert_msg, "read": False, "timestamp": datetime.datetime.now()})
            else:
                current_urgency, current_potential_conditions = f"High Urgency: Severe symptoms reported ({symptom_severity}). Seek medical attention soon.", ["Severe symptoms require medical attention (based on rule)."]
        elif "moderate" in current_severity_lower:
            current_urgency, current_potential_conditions = "Moderate Urgency: Consider consulting a doctor if symptoms persist or worsen.", ["Moderate symptoms warrant medical consideration (based on rule)."]

        st.session_state.symptom_results = {
            "ai_analysis": analysis,
            "urgency": current_urgency,
            "potential_conditions": current_potential_conditions,
            "symptoms_submitted": full_symptom_description,
            "timestamp": datetime.datetime.now()
        }
        st.session_state.symptom_analysis_history.append(dict(st.session_state.symptom_results))
        st.rerun()

    if st.session_state.symptom_analysis_done and st.session_state.symptom_results:
        results = st.session_state.symptom_results
        st.divider()
        st.subheader(f"Current Symptom Analysis Results ({results.get('timestamp', datetime.datetime.now()).strftime('%Y-%m-%d %H:%M')})")

        urgency_text = results.get('urgency', 'N/A')
        if "CRITICAL" in urgency_text: st.error(f"Urgency: {urgency_text}", icon="❗")
        elif "High Urgency" in urgency_text: st.warning(f"Urgency: {urgency_text}", icon="⚠")
        else: st.info(f"Urgency: {urgency_text}", icon="ℹ")

        st.write("Potential Considerations (Rule-Based):")
        for cond_item in results.get('potential_conditions', ['N/A']):
            st.markdown(f"- {cond_item}")

        st.subheader("Detailed AI Analysis")
        # MODIFIED: Removed disclaimer error message.

        analysis_text = results.get('ai_analysis', 'No AI analysis generated.')
        if any(err_msg in analysis_text for err_msg in ["AI analysis is currently unavailable", "An error occurred", "tokenizer is missing"]):
            st.error(f"{analysis_text}")
        elif "No specific analysis" in analysis_text:
            st.info(analysis_text)
        else:
            st.markdown(analysis_text)

        st.write(f"Symptoms Reported: {results.get('symptoms_submitted', 'N/A')}")
        st.subheader("Next Steps:")
        if "CRITICAL" not in urgency_text:
             if st.button("Book Appointment based on these symptoms", key="book_from_symptom_results_v6"):
                 st.session_state.appointment_reason = f"Symptoms: {results.get('symptoms_submitted', 'N/A')[:200]}..."
                 set_page("Appointments")
        else:
            st.write("Follow the CRITICAL urgency instructions and seek immediate medical care.")
        st.markdown("---"); st.caption("Share this information with your doctor.")

    st.button("Clear Symptoms & Analysis", key="clear_symptoms_button_page_bottom_v6", on_click=_clear_symptoms_form_callback, help="Clears the symptom form and the current analysis results.")


def meal_plan_generator_page():
    st.header("🥗 AI-Powered Meal Plan Generator")
    st.write("Provide your details to generate a sample daily meal plan.")
    # MODIFIED: Removed disclaimer warning.

    with st.form("meal_plan_form_v1"):
        st.subheader("Your Personal & Dietary Profile")
        st.markdown("##### Personal Details")
        colA, colB, colC = st.columns(3)
        with colA:
            age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1, key="meal_age")
        with colB:
            gender = st.selectbox("Gender", ("Female", "Male", "Prefer not to say"), key="meal_gender")
        with colC:
            weight = st.number_input("Weight (in kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.5, key="meal_weight")

        st.divider()
        st.markdown("##### Dietary Requirements")
        col1, col2 = st.columns(2)
        with col1:
            diet_pref = st.selectbox(
                "Dietary Preference",
                ("None (Balanced)", "Vegetarian", "Vegan", "Pescatarian", "Gluten-Free", "Keto"),
                key="meal_diet_pref"
            )
            health_goal = st.selectbox(
                "Primary Health Goal",
                ("General Wellness", "Weight Loss", "Weight Gain", "Muscle Building"),
                key="meal_health_goal"
            )
            num_meals = st.select_slider(
                "How many meals per day?",
                options=["3", "4", "5", "6"],
                value="4",
                key="meal_num_meals"
            )
        with col2:
            calories = st.number_input(
                "Approximate Daily Calorie Target (optional)",
                min_value=1000, max_value=5000, value=2000, step=100,
                key="meal_calories"
            )
            allergies = st.text_input(
                "List any allergies or foods to avoid (comma-separated)",
                placeholder="e.g., peanuts, shellfish, dairy",
                key="meal_allergies"
            )
            other_info = st.text_area(
                "Any other specific instructions?",
                placeholder="e.g., prefer quick meals, low-carb snacks...",
                key="meal_other_info"
            )

        generate_button = st.form_submit_button("Generate My Daily Meal Plan")

    if generate_button:
        user_details = {
            "age": age,
            "gender": gender,
            "weight": weight,
            "diet_pref": diet_pref,
            "goal": health_goal,
            "calories": calories,
            "allergies": allergies if allergies.strip() else "None specified",
            "num_meals": f"{num_meals} meals",
            "other": other_info if other_info.strip() else "None specified"
        }
        st.session_state.meal_plan_details_submitted = user_details

        with st.spinner("Our AI chef is crafting your personalized meal plan..."):
            meal_plan = get_llm_meal_plan(user_details)
            st.session_state.meal_plan_result = meal_plan
        st.rerun()

    if st.session_state.meal_plan_result:
        st.divider()
        st.subheader("Your AI-Generated Sample Daily Meal Plan")

        details = st.session_state.meal_plan_details_submitted
        if details:
            st.info(
                f"Generated for: {details.get('age')} years old, {details.get('gender')}, {details.get('weight')} kg. "
                f"Diet: {details['diet_pref']}, Goal: {details['goal']}, "
                f"at approx. {details['calories']} calories. "
                f"Avoiding: {details['allergies']}."
            )

        result_text = st.session_state.meal_plan_result
        if any(err_msg in result_text for err_msg in ["unavailable", "An error occurred"]):
            st.error(result_text)
        else:
            st.markdown(result_text)

        if st.button("Clear and Start Over", key="clear_meal_plan_btn"):
            st.session_state.meal_plan_result = None
            st.session_state.meal_plan_details_submitted = None
            st.rerun()


def my_health_data_page():
    st.header("📊 My Health Data"); st.write("View a history of your symptom analyses and appointments.")
    st.subheader("Past Symptom Analyses")
    if not st.session_state.symptom_analysis_history:
        st.info("No symptom analyses performed in this session yet. Use the 'Symptom Analysis' page.")
    else:
        history_to_display = list(reversed(st.session_state.symptom_analysis_history))
        for i, analysis in enumerate(history_to_display):
            timestamp_str = analysis.get('timestamp', datetime.datetime.now()).strftime('%Y-%m-%d %H:%M')
            symptoms_summary = analysis.get('symptoms_submitted', 'N/A')
            expander_title = f"Analysis #{len(history_to_display)-i}: {timestamp_str} - Symptoms: {symptoms_summary[:70]}{'...' if len(symptoms_summary) > 70 else ''}"

            with st.expander(expander_title):
                st.write(f"Timestamp: {timestamp_str}")
                st.write(f"Symptoms Reported: {symptoms_summary}")

                urgency_text = analysis.get('urgency', 'N/A')
                st.write(f"Urgency Assessment (Rule-Based): {urgency_text}")

                analysis_text = analysis.get('ai_analysis', 'No AI-generated information for this entry.')
                st.write("Detailed AI Analysis:")
                # MODIFIED: Removed disclaimer warning.
                st.markdown(analysis_text)

    st.divider(); st.subheader("Appointment History")

    today = date.today()
    all_appointments_current = list(st.session_state.get('appointments', []))

    made_status_changes = False
    updated_appointments_in_session = []
    for appt_data in all_appointments_current:
        temp_appt = appt_data.copy()
        appt_date_obj = temp_appt.get('date')
        if isinstance(appt_date_obj, date) and appt_date_obj < today and temp_appt.get('status') not in ["Cancelled", "Completed (Past due)"]:
            if temp_appt.get('status') in ["Upcoming", "Upcoming (Rescheduled)"]:
                temp_appt['status'] = "Completed (Past due)"
                made_status_changes = True
        updated_appointments_in_session.append(temp_appt)

    if made_status_changes:
        st.session_state.appointments = updated_appointments_in_session
        all_appointments_for_display = updated_appointments_in_session
        st.rerun()
    else:
        all_appointments_for_display = all_appointments_current


    upcoming_appointments_list = sorted([appt for appt in all_appointments_for_display if isinstance(appt.get('date'), date) and appt['date'] >= today and appt.get('status') not in ["Cancelled", "Completed (Past due)"]], key=lambda x: (x.get('date', date.max), x.get('time', '23:59')))
    past_completed_appointments_list = sorted([appt for appt in all_appointments_for_display if appt.get('status') == "Completed (Past due)"], key=lambda x: (x.get('date', date.min), x.get('time', '00:00')), reverse=True)
    cancelled_appointments_list = sorted([appt for appt in all_appointments_for_display if appt.get('status') == "Cancelled"], key=lambda x: (x.get('cancelled_timestamp', x.get('timestamp', datetime.datetime.min))), reverse=True)

    if not upcoming_appointments_list and not past_completed_appointments_list and not cancelled_appointments_list:
        st.info("No appointments recorded yet.")
    else:
        if upcoming_appointments_list:
            st.write("#### Upcoming Appointments")
            for i, appt in enumerate(upcoming_appointments_list):
                appt_date_obj = appt.get('date', date.min)
                appt_time_str = appt.get('time', 'N/A')
                appt_doctor = appt.get('doctor', 'N/A')
                exp_title = f"Upcoming: {appt_date_obj.strftime('%A, %B %d, %Y')} at {appt_time_str} - Dr. {appt_doctor.split('(')[0].strip() if '(' in appt_doctor else appt_doctor} (Ref: {appt.get('id', 'N/A')[:8]})"
                with st.expander(exp_title, expanded=(i == 0)):
                    st.markdown(f"Date & Time: {appt_date_obj.strftime('%A, %B %d, %Y')} at {appt_time_str}")
                    st.markdown(f"Doctor: {appt_doctor}")
                    st.markdown(f"Reason: {appt.get('reason', 'N/A')}")
                    st.caption(f"Status: {appt.get('status', 'N/A')}")
            st.markdown("---")

        if past_completed_appointments_list:
            st.write("#### Past (Completed) Appointments")
            for i, appt in enumerate(past_completed_appointments_list):
                appt_date_obj = appt.get('date', date.min)
                appt_time_str = appt.get('time', 'N/A')
                appt_doctor = appt.get('doctor', 'N/A')
                exp_title = f"Past: {appt_date_obj.strftime('%A, %B %d, %Y')} at {appt_time_str} - Dr. {appt_doctor.split('(')[0].strip() if '(' in appt_doctor else appt_doctor} (Ref: {appt.get('id', 'N/A')[:8]})"
                with st.expander(exp_title, expanded=(i == 0 and not upcoming_appointments_list)):
                    st.markdown(f"Date & Time: {appt_date_obj.strftime('%A, %B %d, %Y')} at {appt_time_str}")
                    st.markdown(f"Doctor: {appt_doctor}")
                    st.markdown(f"Reason: {appt.get('reason', 'N/A')}")
                    st.caption(f"Status: {appt.get('status', 'Completed (Past due)')}")
            st.markdown("---")

        if cancelled_appointments_list:
            st.write("#### Cancelled Appointments")
            for i, appt in enumerate(cancelled_appointments_list):
                appt_date_obj = appt.get('date', date.min)
                appt_time_str = appt.get('time', 'N/A')
                appt_doctor = appt.get('doctor', 'N/A')
                cancelled_ts = appt.get('cancelled_timestamp', appt.get('timestamp', datetime.datetime.min))
                exp_title = f"Cancelled: Originally {appt_date_obj.strftime('%Y-%m-%d')} with Dr. {appt_doctor.split('(')[0].strip() if '(' in appt_doctor else appt_doctor} (Ref: {appt.get('id', 'N/A')[:8]})"
                if isinstance(cancelled_ts, datetime.datetime):
                    exp_title += f" (Cancelled on {cancelled_ts.strftime('%Y-%m-%d %H:%M')})"
                with st.expander(exp_title, expanded=False):
                    st.markdown(f"Original Date & Time: {appt_date_obj.strftime('%A, %B %d, %Y')} at {appt_time_str}")
                    st.markdown(f"Doctor: {appt_doctor}")
                    st.markdown(f"Reason: {appt.get('reason', 'N/A')}")
                    st.caption(f"Status: {appt.get('status', 'Cancelled')}")
                    if isinstance(cancelled_ts, datetime.datetime):
                        st.caption(f"Cancelled on: {cancelled_ts.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown("---")
    st.divider()


def appointments_page():
    st.header("📅 My Appointments")
    if st.session_state.get('appointments_page_message'):
        msg_info = st.session_state.appointments_page_message
        if msg_info["type"] == "success": st.success(msg_info["message"], icon="✅")
        elif msg_info["type"] == "info": st.info(msg_info["message"], icon="ℹ")
        elif msg_info["type"] == "error": st.error(msg_info["message"], icon="❌")
        st.session_state.appointments_page_message = None

    today = date.today()
    made_status_changes_appt_page = False
    for i in range(len(st.session_state.appointments) - 1, -1, -1):
         appt = st.session_state.appointments[i]
         appt_date_obj = appt.get('date')
         if isinstance(appt_date_obj, date) and appt_date_obj < today and appt.get('status') not in ["Cancelled", "Completed (Past due)"]:
            if appt.get('status') in ["Upcoming", "Upcoming (Rescheduled)"]:
                st.session_state.appointments[i]['status'] = "Completed (Past due)"
                made_status_changes_appt_page = True

    if made_status_changes_appt_page:
        st.rerun()

    tab_book, tab_upcoming, tab_past_cancelled = st.tabs(["Book New Appointment", "Upcoming Appointments", "History (Past & Cancelled)"])

    with tab_book:
        st.subheader("Schedule a New Appointment")
        with st.form("appointment_form_booking_tab_v6"):
            doc_options = ["Any Available Doctor", "Dr. Smith (Cardiology)", "Dr. Jones (General)", "Dr. Lee (Pediatrics)", "Dr. Patel (Dermatology)"]
            selected_doc = st.selectbox("Select Doctor (Optional)", doc_options, key="appt_doc_select_book_tab_v6")

            default_appt_date_val = st.session_state.get("appt_date_input_book_tab_v6_value", date.today())
            if default_appt_date_val < date.today(): default_appt_date_val = date.today()

            appointment_date_val = st.date_input("Preferred Date", min_value=date.today(), value=default_appt_date_val, key="appt_date_input_book_tab_v6")
            st.session_state.appt_date_input_book_tab_v6_value = appointment_date_val

            base_times = ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]
            random.seed(appointment_date_val.strftime("%Y-%m-%d") + selected_doc)
            existing_slots = {a['time'] for a in st.session_state.appointments if isinstance(a.get('date'), date) and a.get('date') == appointment_date_val and (a.get('doctor') == selected_doc or selected_doc == "Any Available Doctor") and a.get('status') != "Cancelled"}
            available_slots = sorted(list(set(base_times) - existing_slots))

            if available_slots:
                appointment_time_val = st.selectbox("Preferred Time", available_slots, key="appt_time_select_book_tab_v6")
            else:
                st.info("No available times for this doctor on this date.")
                appointment_time_val = None

            appointment_reason_val = st.text_area("Reason for Visit", value=st.session_state.get('appointment_reason', ''), height=100, key="appt_reason_text_book_tab_v6")
            book_submit = st.form_submit_button("Request Appointment", disabled=(appointment_time_val is None))

            if book_submit:
                if not appointment_reason_val.strip():
                    st.warning("Please provide a reason for your visit.")
                else:
                    new_appt_data = {"id": f"appt_{time.time()}", "doctor": selected_doc, "date": appointment_date_val, "time": appointment_time_val, "reason": appointment_reason_val, "status": "Upcoming", "timestamp": datetime.datetime.now()}
                    st.session_state.appointments.append(new_appt_data)
                    st.session_state.appointment_reason = ''
                    st.session_state.appointments_page_message = {"type": "success", "message": f"Appointment successfully requested with {selected_doc} on {appointment_date_val.strftime('%Y-%m-%d')} at {appointment_time_val}."}
                    st.rerun()

    with tab_upcoming:
        st.subheader("Upcoming Appointments")
        upcoming_for_display = sorted([a for a in st.session_state.appointments if a.get('status', '').startswith("Upcoming")], key=lambda x: (x.get('date', date.max), x.get('time', '23:59')))
        if not upcoming_for_display:
            st.info("No upcoming appointments.")
        else:
            for i, appt in enumerate(upcoming_for_display):
                st.markdown(f"**{appt.get('date').strftime('%A, %B %d, %Y')} at {appt.get('time', 'N/A')}** - Dr. {appt.get('doctor', 'N/A')}")
                st.markdown(f"> {appt.get('reason', 'N/A')}")
                st.caption(f"Status: {appt.get('status')}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Reschedule", key=f"resched_btn_{appt['id']}"):
                        st.session_state.rescheduling_appointment_id = appt['id']
                        st.rerun()
                with col2:
                    if st.button("Cancel", key=f"cancel_btn_{appt['id']}"):
                        for idx, item in enumerate(st.session_state.appointments):
                            if item['id'] == appt['id']:
                                st.session_state.appointments[idx]['status'] = 'Cancelled'
                                st.session_state.appointments[idx]['cancelled_timestamp'] = datetime.datetime.now()
                                st.session_state.appointments_page_message = {"type": "info", "message": "Appointment cancelled."}
                                st.rerun()
                st.divider()

        if st.session_state.rescheduling_appointment_id:
            st.subheader("Reschedule Appointment")
            appt_to_reschedule = next((a for a in st.session_state.appointments if a['id'] == st.session_state.rescheduling_appointment_id), None)
            if appt_to_reschedule:
                new_date = st.date_input("New Date", min_value=date.today(), key=f"new_date_{appt_to_reschedule['id']}")
                # Simplified available times for rescheduling example
                new_time = st.selectbox("New Time", base_times, key=f"new_time_{appt_to_reschedule['id']}")
                if st.button("Confirm Reschedule", key=f"confirm_resched_{appt_to_reschedule['id']}"):
                    for idx, item in enumerate(st.session_state.appointments):
                        if item['id'] == appt_to_reschedule['id']:
                            st.session_state.appointments[idx]['date'] = new_date
                            st.session_state.appointments[idx]['time'] = new_time
                            st.session_state.appointments[idx]['status'] = 'Upcoming (Rescheduled)'
                            st.session_state.rescheduling_appointment_id = None
                            st.session_state.appointments_page_message = {"type": "success", "message": "Appointment successfully rescheduled."}
                            st.rerun()

    with tab_past_cancelled:
        st.subheader("Past (Completed) Appointments")
        past_for_display = sorted([a for a in st.session_state.appointments if a.get('status') == "Completed (Past due)"], key=lambda x: (x.get('date', date.min), x.get('time', '00:00')), reverse=True)
        if not past_for_display:
            st.info("No past appointments.")
        else:
            for appt in past_for_display:
                st.markdown(f"**{appt.get('date').strftime('%A, %B %d, %Y')} at {appt.get('time', 'N/A')}** - Dr. {appt.get('doctor', 'N/A')}")
                st.markdown(f"> {appt.get('reason', 'N/A')}")
                st.caption(f"Status: {appt.get('status')}")
                st.divider()

        st.subheader("Cancelled Appointments")
        cancelled_for_display = sorted([a for a in st.session_state.appointments if a.get('status') == "Cancelled"], key=lambda x: (x.get('cancelled_timestamp', datetime.datetime.min)), reverse=True)
        if not cancelled_for_display:
            st.info("No cancelled appointments.")
        else:
            for appt in cancelled_for_display:
                st.markdown(f"Originally for: {appt.get('date').strftime('%A, %B %d, %Y')} at {appt.get('time', 'N/A')} - Dr. {appt.get('doctor', 'N/A')}")
                st.caption(f"Cancelled on: {appt.get('cancelled_timestamp').strftime('%Y-%m-%d %H:%M')}")
                st.divider()


def resources_learning_page():
    st.header("📚 Resources & Learning")
    st.write("Explore health topics using our AI-powered tool.")

    st.subheader("Health Topic Explorer")
    st.write("Enter a health topic (e.g., 'Diabetes', 'Migraine', 'Vitamin D Deficiency') for a general overview by AI.")

    if "health_topic_query_input_val_v6" not in st.session_state: st.session_state.health_topic_query_input_val_v6 = ""
    if "health_topic_overview_result_v6" not in st.session_state: st.session_state.health_topic_overview_result_v6 = None
    if "health_topic_overview_query_processed_v6" not in st.session_state: st.session_state.health_topic_overview_query_processed_v6 = None

    topic_query_input_val = st.text_input("Enter health topic:", value=st.session_state.health_topic_query_input_val_v6, key="health_topic_query_input_resource_v6")
    st.session_state.health_topic_query_input_val_v6 = topic_query_input_val

    if st.button("Get Information", key="get_topic_info_btn_resource_v6"):
        if topic_query_input_val.strip():
            st.session_state.health_topic_overview_query_processed_v6 = topic_query_input_val
            with st.spinner(f"Fetching info on '{topic_query_input_val}'..."):
                overview_content = get_llm_health_topic_info(topic_query_input_val)
                st.session_state.health_topic_overview_result_v6 = overview_content
        else:
            st.warning("Please enter a health topic.")
            st.session_state.health_topic_overview_result_v6 = None
        st.rerun()

    if st.session_state.health_topic_overview_result_v6 and st.session_state.health_topic_overview_query_processed_v6 == st.session_state.health_topic_query_input_val_v6:
        query_title_display = st.session_state.health_topic_overview_query_processed_v6
        overview_content_display = st.session_state.health_topic_overview_result_v6
        if any(err in overview_content_display for err in ["error", "unavailable", "Could not generate"]):
            st.error(f"Could not fetch information for '{query_title_display}': {overview_content_display}")
        else:
            st.markdown(f"<div style='background-color: rgba(249, 249, 249, 0.9); padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin-top:10px; color: black;'><h4>Overview for: {query_title_display}</h4><p>{overview_content_display}</p></div>", unsafe_allow_html=True)

    # MODIFIED: Removed disclaimer caption.

# --- Main App Execution and Navigation ---
if __name__ == "__main__":
    init_db()

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'

    if st.session_state.authenticated:
        try:
            add_bg_from_local('background2.jpg')
        except Exception:
            # Fallback to a simpler style if image is not found
            st.markdown("""
            <style>
             [data-testid="stAppViewContainer"] {
                 background-color: #e5e5f7;
                 opacity: 0.8;
                 background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #e5e5f7 10px ), repeating-linear-gradient( #b8c6db, #f5f7fa );
             }
            </style>
            """, unsafe_allow_html=True)

        with st.sidebar:
            st.title("⚕ Health AI Nav")
            st.markdown("---")
            pages = {
                'Home': '🏠 Home',
                'SymptomChecker': '🩺 Symptom Analysis',
                'MealPlanner': '🥗 Meal Plan Generator',
                'MyHealthData': '📊 My Health Data',
                'Appointments': '📅 Appointments',
                'Resources': '📚 Resources'
            }
            for page_key, page_title in pages.items():
                if st.button(page_title, key=f"nav_btn_main_v7_{page_key}", type="primary" if st.session_state.current_page == page_key else "secondary", use_container_width=True):
                    set_page(page_key)
            st.markdown("---")
            if st.button("Logout", key="logout_button_v6", use_container_width=True):
                logout()
            st.markdown("---"); st.write("AI Model Status:")
            if model and tokenizer: st.success(f"Model Loaded ({str(device).upper()})", icon="✅")
            else: st.error("Model Failed to Load", icon="❌")

        page_to_render = st.session_state.current_page
        if page_to_render == 'SymptomChecker':
            symptom_checker_page()
        elif page_to_render == 'MealPlanner':
            meal_plan_generator_page()
        elif page_to_render == 'MyHealthData':
            my_health_data_page()
        elif page_to_render == 'Appointments':
            appointments_page()
        elif page_to_render == 'Resources':
            resources_learning_page()
        else:
            home_page()
    else:
        try:
            add_bg_from_local('background.jpg')
        except FileNotFoundError:
            pass

        login_page()