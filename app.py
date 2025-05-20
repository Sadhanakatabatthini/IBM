import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datetime
from datetime import date, timedelta
import random
import time # Import time for potentially clearing messages
import hashlib  # For password hashing
import uuid  # For generating unique user IDs


# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Health AI Assistant", page_icon="⚕")


# --- User Authentication System ---
# Database to store user credentials (in a real app, use a proper database)
if 'user_database' not in st.session_state:
    st.session_state.user_database = {}

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    """Register a new user."""
    if username in st.session_state.user_database:
        return False, "Username already exists. Please choose another."
    
    # Check for valid inputs
    if not username or not password or not email:
        return False, "All fields are required."
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    
    if '@' not in email or '.' not in email:
        return False, "Please enter a valid email address."
    
    # Store the user with hashed password
    user_id = str(uuid.uuid4())
    st.session_state.user_database[username] = {
        'id': user_id,
        'password_hash': hash_password(password),
        'email': email,
        'created_at': datetime.datetime.now(),
        'profile': {}  # Empty profile to be filled later
    }
    return True, "Registration successful! Please log in."

def authenticate(username, password):
    """Authenticate a user."""
    if username not in st.session_state.user_database:
        return False
    
    stored_hash = st.session_state.user_database[username]['password_hash']
    if hash_password(password) == stored_hash:
        st.session_state.authenticated = True
        st.session_state.current_user = username
        st.session_state.login_error = ""
        return True
    else:
        st.session_state.authenticated = False
        st.session_state.login_error = "Invalid username or password."
        return False

def logout():
    """Logs the user out by resetting authentication state and clearing data."""
    st.session_state.authenticated = False
    st.session_state.current_user = None
    # Optionally clear sensitive data on logout
    # st.session_state.symptom_analysis_history = []
    # st.session_state.appointments = []
    # st.session_state.profile = {}
    # st.session_state.alerts = [] # Decide if alerts should persist or clear
    st.session_state.current_page = 'Home' # Redirect to home or login page
    st.session_state.login_error = "" # Clear login error on logout
    st.session_state.appointments_page_message = None
    st.session_state.rescheduling_appointment_id = None
    st.session_state.pop("new_reschedule_date_input_tab_v6_value", None)

    # Clear form states related to specific pages to prevent data leakage or weird behavior
    _clear_symptoms_form_callback()
    # Clear resource page state
    st.session_state.health_topic_query_input_val_v5 = ""
    st.session_state.health_topic_overview_result_v5 = None
    st.session_state.health_topic_overview_query_processed_v5 = None
    # Clear appointment booking state
    st.session_state.appointment_reason = ""
    st.session_state.pop("appt_date_input_book_tab_v6_value", None) # Remove explicitly keyed date
    st.session_state.pop("appt_doc_select_book_tab_v6", None)
    st.session_state.pop("appt_time_select_book_tab_v6", None)
    st.session_state.pop("appt_reason_text_book_tab_v6", None)


    st.rerun()


# --- Global Variables and Model Loading ---
# Only load model if we *might* need it (i.e., not just sitting on the login page)
# We can lazy load, but @st.cache_resource is efficient, so loading it upfront is fine.
# However, we should gate the *use* of the model behind the authenticated check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """Loads the AI model and tokenizer."""
    model_name = "ibm-granite/granite-3.3-2b-instruct" # "google/gemma-2b-it" # "mistralai/Mistral-7B-Instruct-v0.2" #
    print(f"Attempting to load model: {model_name} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == 'cuda' else None, # float16 for GPU, None for CPU (to avoid errors)
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)
        if tokenizer.pad_token_id is None:
             tokenizer.pad_token_id = tokenizer.eos_token_id
             if tokenizer.pad_token_id is None: # Defensive check if eos_token_id is also None
                 print("Warning: EOS token not found, cannot set pad_token_id.")
        print(f"Model '{model_name}' loaded successfully on {device}!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        st.error(f"Critical Error: AI model '{model_name}' could not be loaded. {e}. AI-dependent features will be unavailable.", icon="❌")
        return None, None

# Load model outside the conditional logic so it's cached on first run
tokenizer, model = load_model()


# --- Helper Functions for AI Generation (UNCHANGED) ---
def get_llm_advice(symptoms_text):
    if model is None or tokenizer is None: return "AI analysis is currently unavailable because the required model could not be loaded."
    if tokenizer.pad_token_id is None: return "AI analysis unavailable: Model tokenizer is missing a pad token."
    prompt = f"Analyze the following patient symptoms and provide general advice and potential considerations (do not provide a medical diagnosis or specific treatment plans, just general insights based on symptoms):\nSymptoms: {symptoms_text}\nAnalysis:"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generate_kwargs = dict(max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
        with torch.no_grad(): outputs = model.generate(inputs["input_ids"], **generate_kwargs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # More robust way to get the advice part after the prompt
        prompt_start_index = generated_text.rfind(prompt) # find the last occurrence
        advice = generated_text[prompt_start_index + len(prompt):].strip() if prompt_start_index != -1 else generated_text.strip()
        if advice.startswith("Analysis:"): advice = advice[len("Analysis:"):].strip() # Remove "Analysis:" if present
        return advice if advice.strip() else "No specific advice could be generated for these symptoms at this time."
    except Exception as e:
        print(f"An error occurred during AI generation (advice): {e}")
        if device.type == 'cuda': torch.cuda.empty_cache()
        return "An error occurred while generating AI advice. Please try again or check the logs."

def get_llm_health_topic_info(topic_query):
    if model is None or tokenizer is None: return "AI information is currently unavailable because the required model could not be loaded."
    if tokenizer.pad_token_id is None: return "AI information unavailable: Model tokenizer is missing a pad token."
    prompt = f"Provide a general overview of the health topic: {topic_query}. (Do not provide medical advice or diagnosis. Focus on basic information.)\nOverview:"
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


# --- Placeholder/Example Data (UNCHANGED) ---
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
    # Clear page-specific states if necessary before navigating, e.g., symptom form
    # _clear_symptoms_form_callback() # Uncomment if you want to clear symptoms form on *any* page navigation
    st.rerun()

def _clear_symptoms_form_callback():
    """Callback function for clearing symptoms form state."""
    for category, sym_list in COMMON_SYMPTOMS.items():
        for symptom in sym_list:
            checkbox_key = f"cb_symp_{category}{symptom.replace(' ', '').replace('-', '').replace('/', '').lower()}"
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

# Initialize *other* session states only if they don't exist.
# These states are tied to the user session *after* login.
# Decide if you want to clear them on logout or keep them across logins in the same browser session.
# Keeping them is simpler for this demo, clearing offers more privacy.
# We'll clear them on logout in the logout function.
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
# Initialize session state for common symptom checkboxes and form fields
for category, sym_list in COMMON_SYMPTOMS.items():
    for symptom in sym_list:
        checkbox_key = f"cb_symp_{category}{symptom.replace(' ', '').replace('-', '').replace('/', '').lower()}"
        if checkbox_key not in st.session_state: st.session_state[checkbox_key] = False
if "ta_custom_symptoms" not in st.session_state: st.session_state["ta_custom_symptoms"] = ""
if "sel_duration" not in st.session_state: st.session_state["sel_duration"] = "Today"
if "slider_severity" not in st.session_state: st.session_state["slider_severity"] = "Mild"
# Initialize state for resources page
if "health_topic_query_input_val_v5" not in st.session_state: st.session_state.health_topic_query_input_val_v5 = ""
if "health_topic_overview_result_v5" not in st.session_state: st.session_state.health_topic_overview_result_v5 = None
if "health_topic_overview_query_processed_v5" not in st.session_state: st.session_state.health_topic_overview_query_processed_v5 = None

# --- Login Page Function ---
def login_page():
    st.title("⚕ Health AI Assistant")
    
    # Create tabs for login and registration
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    # Login Tab
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form_v2"):
            username = st.text_input("Username", key="login_username_input_v2")
            password = st.text_input("Password", type="password", key="login_password_input_v2")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if authenticate(username, password):
                    st.success("Login successful! Redirecting to dashboard...", icon="✅")
                    time.sleep(0.5)  # Small delay for success message visibility
                    st.rerun()
        
        # Display login error if any
        if st.session_state.login_error:
            st.error(st.session_state.login_error, icon="❌")
            
    # Registration Tab
    with tab2:
        st.subheader("Create a New Account")
        
        with st.form("registration_form_v1"):
            new_username = st.text_input("Choose a Username", key="reg_username_input_v1")
            new_email = st.text_input("Email Address", key="reg_email_input_v1")
            new_password = st.text_input("Create Password", type="password", key="reg_password_input_v1")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password_v1")
            
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
        
        # Display registration status message outside the form if set
        if st.session_state.registration_status["success"] is not None:
            if st.session_state.registration_status["success"]:
                st.success(st.session_state.registration_status["message"], icon="✅")
            else:
                st.error(st.session_state.registration_status["message"], icon="❌")
            
            # Reset registration status after displaying
            st.session_state.registration_status = {"success": None, "message": ""}
            
    # Add information about the application
    st.markdown("---")
    st.markdown("### About Health AI Assistant")
    st.markdown("""
    This application provides AI-powered health information and symptom analysis. 
    Please note that this is not a substitute for professional medical advice.
    
    * Symptom analysis
    * Health resources
    * Appointment scheduling
    * Health profile management
    """)
    
    # Footer
    st.markdown("---")
    st.caption("© 2025 Health AI Assistant. All rights reserved.")


# --- Page Functions (UNCHANGED, except they now assume authentication) ---
def home_page():
    st.title("🏠 Health AI Assistant Home"); st.write("Welcome to the Health AI Assistant. Use the sidebar to navigate.")
    st.write("This application provides AI-powered tools to help you manage your health information and get general insights. Please remember that AI outputs are for informational purposes only and are not a substitute for professional medical advice.")
    st.markdown("---"); st.subheader("Explore Features:")
    st.markdown("- Symptom Analysis: Describe your symptoms and get general AI-based insights.\n- My Health Data: View your past symptom analyses and appointment history.\n- Appointments: Schedule new appointments and see upcoming/past ones.\n- Resources: Explore health topics using our AI-powered tool.")
    st.markdown("---"); st.write("Select a page from the sidebar to get started.")

    if st.session_state.alerts:
        st.subheader("Active Alerts")
        unread_alerts = [alert for alert in st.session_state.alerts if not alert.get('read', False)]
        if unread_alerts:
            for i, alert_item in enumerate(unread_alerts):
                alert_message = f"{alert_item['message']}"
                # Ensure keys are unique across runs, combining ID and timestamp is robust
                alert_id_key_part = alert_item.get('id', f"alert_{i}")
                alert_ts_key_part = alert_item.get('timestamp', datetime.datetime.now()).strftime('%Y%m%d%H%M%S%f')
                dismiss_key = f"dismiss_alert_home_v6_{alert_id_key_part}_{alert_ts_key_part}" # Added v6

                col1_alert, col2_alert = st.columns([0.9, 0.1]) # Renamed columns to avoid conflict
                with col1_alert:
                    if alert_item['type'] == 'critical': st.error(f"❗ {alert_message}")
                    elif alert_item['type'] == 'warning': st.warning(f"⚠ {alert_message}")
                    elif alert_item['type'] == 'success': st.success(f"✅ {alert_message}")
                    elif alert_item['type'] == 'info': st.info(f"ℹ {alert_message}")
                with col2_alert:
                    if st.button("Dismiss", key=dismiss_key, help="Mark this alert as read"):
                        # Find the specific alert in the session state list and mark it as read
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
    st.warning("Disclaimer: This tool provides general information, NOT medical diagnosis or treatment. Always consult a healthcare professional for health concerns.", icon="⚠")
    st.subheader("Describe Your Current Symptoms")

    with st.form("symptom_form_page_v6"): # Unique form key
        selected_symptoms = []
        for category, sym_list in COMMON_SYMPTOMS.items():
            with st.expander(f"{category} Symptoms"):
                for symptom in sym_list:
                    checkbox_key = f"cb_symp_{category}{symptom.replace(' ', '').replace('-', '').replace('/', '').lower()}_v6" # Unique key
                    # Use session state to preserve checkbox values across reruns within the form
                    if st.checkbox(symptom, key=checkbox_key, value=st.session_state.get(checkbox_key, False)):
                        selected_symptoms.append(symptom)

        custom_symptoms = st.text_area(
            "Describe additional symptoms (comma-separated):",
            height=100,
            key="ta_custom_symptoms_page_v6", # Unique Key
            value=st.session_state.get("ta_custom_symptoms", "")
        )

        duration_options = ["Today", "Yesterday", "2-3 days ago", "4-7 days ago", "More than a week ago", "Unsure"]
        symptom_duration_val = st.session_state.get("sel_duration", "Today")
        symptom_duration_index = duration_options.index(symptom_duration_val) if symptom_duration_val in duration_options else 0
        symptom_duration = st.selectbox(
            "Symptom duration:",
            duration_options,
            key="sel_duration_page_v6", # Unique Key
            index=symptom_duration_index
        )

        severity_options = ["Mild", "Moderate", "Severe", "Very Severe"]
        symptom_severity_val = st.session_state.get("slider_severity", "Mild")
        if symptom_severity_val not in severity_options: symptom_severity_val = severity_options[0]

        symptom_severity = st.select_slider(
            "Overall severity:",
            options=severity_options,
            key="slider_severity_page_v6", # Unique Key
            value=symptom_severity_val
        )
        submit_symptoms_button = st.form_submit_button("Analyze My Current Symptoms")

    if submit_symptoms_button:
        # Update session state with current form values upon submission
        st.session_state.ta_custom_symptoms = custom_symptoms
        st.session_state.sel_duration = symptom_duration
        st.session_state.slider_severity = symptom_severity

        # Gather selected checkbox symptoms again (selected_symptoms list is from current run's checkbox states)
        # A more robust way is to iterate through session state keys for checkboxes if not relying on selected_symptoms list.
        current_selected_symptoms = []
        for cat, s_list in COMMON_SYMPTOMS.items():
            for s in s_list:
                cb_k = f"cb_symp_{cat}{s.replace(' ', '').replace('-', '').replace('/', '').lower()}_v6" # Use the unique key
                if st.session_state.get(cb_k, False):
                    current_selected_symptoms.append(s)

        final_symptoms_list = list(set(current_selected_symptoms + [s.strip() for s in custom_symptoms.split(',') if s.strip()]))
        full_symptom_description = f"Selected: {', '.join(final_symptoms_list) if final_symptoms_list else 'None'}. Custom: {custom_symptoms if custom_symptoms.strip() else 'None'}. Duration: {symptom_duration}. Severity: {symptom_severity}."

        st.session_state.current_symptoms_text = full_symptom_description
        st.session_state.symptom_analysis_done = True

        with st.spinner("HealthAI is analyzing your current symptoms..."):
            if model is None or tokenizer is None:
                advice = "AI analysis is currently unavailable because the required model could not be loaded."
            elif tokenizer.pad_token_id is None:
                advice = "AI analysis unavailable: Model tokenizer is missing a pad token."
            else:
                advice = get_llm_advice(full_symptom_description)

        # Rule-based urgency (example)
        current_urgency, current_potential_conditions = "Monitor symptoms. Consult a doctor if symptoms persist or worsen.", ["General considerations based on symptoms (based on rule)."]
        current_lower_desc, current_severity_lower = full_symptom_description.lower(), symptom_severity.lower()

        if "very severe" in current_severity_lower or "severe" in current_severity_lower:
            if any(phrase in current_lower_desc for phrase in ["chest pain", "shortness of breath", "difficulty breathing"]):
                current_urgency, current_potential_conditions = "CRITICAL: Seek immediate medical attention for severe chest pain or breathing issues.", ["Potential serious cardiac/respiratory issue (based on rule)."]
                alert_msg = "CRITICAL SYMPTOM: Severe chest pain/breathing difficulty reported. Seek emergency care."
                alert_id_crit = str(datetime.datetime.now().timestamp()) + "_symptom_critical_v6" # Unique ID
                if not any(a.get('id') == alert_id_crit and not a.get('read', False) for a in st.session_state.alerts):
                     st.session_state.alerts.append({"id": alert_id_crit, "type": "critical", "message": alert_msg, "read": False, "timestamp": datetime.datetime.now()})
            elif "fever" in current_lower_desc:
                current_urgency, current_potential_conditions = "High Urgency: High fever with severe symptoms. Seek medical attention soon.", ["Possible significant infection (based on rule)."]
            else:
                current_urgency, current_potential_conditions = f"High Urgency: Severe symptoms reported ({symptom_severity}). Seek medical attention soon.", ["Severe symptoms require medical attention (based on rule)."]
        elif "moderate" in current_severity_lower:
            current_urgency, current_potential_conditions = "Moderate Urgency: Consider consulting a doctor if symptoms persist or worsen.", ["Moderate symptoms warrant medical consideration (based on rule)."]
        elif "mild" in current_severity_lower and \
             any(phrase in current_lower_desc for phrase in ["runny nose", "sore throat"]) and \
             "fever" not in current_lower_desc:
            current_urgency, current_potential_conditions = "Low Urgency: Monitor symptoms. Consider self-care. Consult a doctor if symptoms worsen/persist.", ["Possible common cold/mild allergy (based on rule)."]

        st.session_state.symptom_results = {
            "advice": advice,
            "urgency": current_urgency,
            "potential_conditions": current_potential_conditions,
            "symptoms_submitted": full_symptom_description,
            "timestamp": datetime.datetime.now()
        }
        st.session_state.symptom_analysis_history.append(dict(st.session_state.symptom_results)) # Store a copy
        st.rerun() # Rerun to display the results from session state


    if st.session_state.symptom_analysis_done and st.session_state.symptom_results:
        results = st.session_state.symptom_results
        st.divider()
        st.subheader(f"Current Symptom Analysis Results ({results.get('timestamp', datetime.datetime.now()).strftime('%Y-%m-%d %H:%M')})")

        urgency_text = results.get('urgency', 'N/A')
        if "CRITICAL" in urgency_text: st.error(f"Urgency: {urgency_text}", icon="❗")
        elif "High Urgency" in urgency_text: st.warning(f"Urgency: {urgency_text}", icon="⚠")
        elif "Moderate Urgency" in urgency_text: st.warning(f"Urgency: {urgency_text}", icon="⚠")
        else: st.info(f"Urgency: {urgency_text}", icon="ℹ")

        st.write("Potential Considerations (Rule-Based):")
        potential_cond = results.get('potential_conditions', ['N/A'])
        if isinstance(potential_cond, list):
            for cond_item in potential_cond:
                st.markdown(f"- {cond_item}")
        else:
            st.markdown(f"- {potential_cond}")

        st.subheader("General AI Advice:")
        advice_text = results.get('advice', 'No AI advice generated.')
        if any(err_msg in advice_text for err_msg in ["AI analysis is currently unavailable", "An error occurred while generating AI advice", "tokenizer is missing a pad token"]):
            st.error(f"{advice_text}")
        elif advice_text.strip() and advice_text != "No specific advice could be generated for these symptoms at this time.":
            st.markdown(advice_text)
        else:
            st.info("No specific AI advice generated based on the input. Please consult a doctor for medical concerns.")

        st.write(f"Symptoms Reported: {results.get('symptoms_submitted', 'N/A')}")
        st.subheader("Next Steps:")
        if "CRITICAL" not in urgency_text:
             if st.button("Book Appointment based on these symptoms", key="book_from_symptom_results_v6"): # Unique key
                 st.session_state.appointment_reason = f"Symptoms: {results.get('symptoms_submitted', 'N/A')[:200]}..."
                 set_page("Appointments")
        else:
            st.write("Follow the CRITICAL urgency instructions and seek immediate medical care.")
        st.markdown("---"); st.caption("Share this information with your doctor.")

    st.button("Clear Symptoms & Analysis", key="clear_symptoms_button_page_bottom_v6", on_click=_clear_symptoms_form_callback, help="Clears the symptom form and the current analysis results.")


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

            with st.expander(expander_title): # Unique key is not strictly necessary here as expander content is dynamic
                st.write(f"*Timestamp:* {timestamp_str}")
                st.write(f"*Symptoms Reported:* {symptoms_summary}")

                urgency_text = analysis.get('urgency', 'N/A')
                if "CRITICAL" in urgency_text: st.error(f"*Urgency Assessment:* {urgency_text}", icon="❗")
                elif "High Urgency" in urgency_text: st.warning(f"*Urgency Assessment:* {urgency_text}", icon="⚠")
                elif "Moderate Urgency" in urgency_text: st.warning(f"*Urgency Assessment:* {urgency_text}", icon="⚠")
                else: st.info(f"*Urgency Assessment:* {urgency_text}", icon="ℹ")

                potential_cond = analysis.get('potential_conditions', ['N/A'])
                st.write("*Potential Considerations (Rule-Based):*")
                if isinstance(potential_cond, list):
                    for p_cond_item in potential_cond:
                        st.markdown(f"- {p_cond_item}")
                else:
                     st.markdown(f"- {potential_cond}")

                advice_text = analysis.get('advice', 'No AI advice generated.')
                st.write("*General AI Advice:*")
                if any(err_msg in advice_text for err_msg in ["AI analysis is currently unavailable", "An error occurred while generating AI advice", "tokenizer is missing a pad token"]):
                    st.error(f"{advice_text}")
                elif advice_text.strip() and advice_text != "No specific advice could be generated for these symptoms at this time.":
                    st.markdown(advice_text)
                else:
                    st.info("No specific AI advice generated for this entry.")
    st.divider(); st.subheader("Appointment History")

    today = date.today()
    all_appointments_current = list(st.session_state.get('appointments', []))

    made_status_changes = False
    updated_appointments_in_session = []
    for appt_data in all_appointments_current:
        temp_appt = appt_data.copy()
        # Ensure 'date' is a date object if it exists and isn't None
        appt_date_obj = temp_appt.get('date')
        if isinstance(appt_date_obj, date) and appt_date_obj < today and temp_appt.get('status') not in ["Cancelled", "Completed (Past due)"]:
            if temp_appt.get('status') in ["Upcoming", "Upcoming (Rescheduled)"]:
                temp_appt['status'] = "Completed (Past due)"
                made_status_changes = True
        updated_appointments_in_session.append(temp_appt)

    if made_status_changes:
        st.session_state.appointments = updated_appointments_in_session
        all_appointments_for_display = updated_appointments_in_session
        st.rerun() # Rerun to show updated statuses immediately on My Health Data page
    else:
        all_appointments_for_display = all_appointments_current


    upcoming_appointments_list = sorted([
        appt for appt in all_appointments_for_display
        if isinstance(appt.get('date'), date) and appt['date'] >= today and appt.get('status') not in ["Cancelled", "Completed (Past due)"]
    ], key=lambda x: (x.get('date', date.max), x.get('time', '23:59')))

    past_completed_appointments_list = sorted([
        appt for appt in all_appointments_for_display
        if appt.get('status') == "Completed (Past due)"
    ], key=lambda x: (x.get('date', date.min), x.get('time', '00:00')), reverse=True)

    cancelled_appointments_list = sorted([
        appt for appt in all_appointments_for_display if appt.get('status') == "Cancelled"
    ], key=lambda x: (x.get('cancelled_timestamp', x.get('timestamp', datetime.datetime.min))), reverse=True)


    if not upcoming_appointments_list and not past_completed_appointments_list and not cancelled_appointments_list:
        st.info("No appointments recorded yet.")
    else:
        if upcoming_appointments_list:
            st.write("#### Upcoming Appointments")
            for i, appt in enumerate(upcoming_appointments_list):
                appt_date_obj = appt.get('date', date.min)
                appt_time_str = appt.get('time', 'N/A')
                appt_doctor = appt.get('doctor', 'N/A')
                exp_title = f"Upcoming: {appt_date_obj.strftime('%A, %B %d, %Y')} at {appt_time_str} - Dr. {appt_doctor.split('(')[0].strip() if '(' in appt_doctor else appt_doctor} (Ref: {appt.get('id', 'N/A')[:8]})" # Used get('id','N/A')
                with st.expander(exp_title, expanded=(i == 0)):
                    st.markdown(f"*Date & Time:* {appt_date_obj.strftime('%A, %B %d, %Y')} at {appt_time_str}")
                    st.markdown(f"*Doctor:* {appt_doctor}")
                    st.markdown(f"*Reason:* {appt.get('reason', 'N/A')}")
                    st.caption(f"*Status:* {appt.get('status', 'N/A')}")
            st.markdown("---")

        if past_completed_appointments_list:
            st.write("#### Past (Completed) Appointments")
            for i, appt in enumerate(past_completed_appointments_list):
                appt_date_obj = appt.get('date', date.min)
                appt_time_str = appt.get('time', 'N/A')
                appt_doctor = appt.get('doctor', 'N/A')
                exp_title = f"Past: {appt_date_obj.strftime('%A, %B %d, %Y')} at {appt_time_str} - Dr. {appt_doctor.split('(')[0].strip() if '(' in appt_doctor else appt_doctor} (Ref: {appt.get('id', 'N/A')[:8]})" # Used get('id','N/A')
                with st.expander(exp_title, expanded=(i == 0 and not upcoming_appointments_list)):
                    st.markdown(f"*Date & Time:* {appt_date_obj.strftime('%A, %B %d, %Y')} at {appt_time_str}")
                    st.markdown(f"*Doctor:* {appt_doctor}")
                    st.markdown(f"*Reason:* {appt.get('reason', 'N/A')}")
                    st.caption(f"*Status:* {appt.get('status', 'Completed (Past due)')}")
            st.markdown("---")

        if cancelled_appointments_list:
            st.write("#### Cancelled Appointments")
            for i, appt in enumerate(cancelled_appointments_list):
                appt_date_obj = appt.get('date', date.min)
                appt_time_str = appt.get('time', 'N/A')
                appt_doctor = appt.get('doctor', 'N/A')
                cancelled_ts = appt.get('cancelled_timestamp', appt.get('timestamp', datetime.datetime.min))
                exp_title = f"Cancelled: Originally {appt_date_obj.strftime('%Y-%m-%d')} with Dr. {appt_doctor.split('(')[0].strip() if '(' in appt_doctor else appt_doctor} (Ref: {appt.get('id', 'N/A')[:8]})" # Used get('id','N/A')
                if isinstance(cancelled_ts, datetime.datetime):
                    exp_title += f" (Cancelled on {cancelled_ts.strftime('%Y-%m-%d %H:%M')})"
                with st.expander(exp_title, expanded=False):
                    st.markdown(f"*Original Date & Time:* {appt_date_obj.strftime('%A, %B %d, %Y')} at {appt_time_str}")
                    st.markdown(f"*Doctor:* {appt_doctor}")
                    st.markdown(f"*Reason:* {appt.get('reason', 'N/A')}")
                    st.caption(f"*Status:* {appt.get('status', 'Cancelled')}")
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
        # Clear message after displaying
        st.session_state.appointments_page_message = None


    # Update status of past appointments first, regardless of which tab is open
    today = date.today()
    made_status_changes_appt_page = False
    # Iterate backwards if potentially removing/modifying list
    for i in range(len(st.session_state.appointments) - 1, -1, -1):
         appt = st.session_state.appointments[i]
         appt_date_obj = appt.get('date')
         # Check if it's a date object and in the past, and not already completed/cancelled
         if isinstance(appt_date_obj, date) and appt_date_obj < today and appt.get('status') not in ["Cancelled", "Completed (Past due)"]:
            if appt.get('status') in ["Upcoming", "Upcoming (Rescheduled)"]:
                st.session_state.appointments[i]['status'] = "Completed (Past due)"
                made_status_changes_appt_page = True

    # Rerun if any status was changed
    if made_status_changes_appt_page:
        st.rerun()


    tab_book, tab_upcoming, tab_past_cancelled = st.tabs(["Book New Appointment", "Upcoming Appointments", "History (Past & Cancelled)"])

    with tab_book:
        st.subheader("Schedule a New Appointment")
        with st.form("appointment_form_booking_tab_v6"):
            doc_options = ["Any Available Doctor", "Dr. Smith (Cardiology)", "Dr. Jones (General)", "Dr. Lee (Pediatrics)", "Dr. Patel (Dermatology)"]
            selected_doc = st.selectbox("Select Doctor (Optional)", doc_options, key="appt_doc_select_book_tab_v6")

            default_appt_date_val = st.session_state.get("appt_date_input_book_tab_v6_value", date.today())
            if default_appt_date_val < date.today(): default_appt_date_val = date.today() # Ensure default is not in past

            appointment_date_val = st.date_input(
                "Preferred Date",
                min_value=date.today(),
                value=default_appt_date_val,
                key="appt_date_input_book_tab_v6"
            )
            st.session_state.appt_date_input_book_tab_v6_value = appointment_date_val # Store selected value

            urgency_factor = 0
            if st.session_state.symptom_analysis_history:
                last_symptom_result = st.session_state.symptom_analysis_history[-1]
                if last_symptom_result and last_symptom_result.get('urgency'):
                    urgency_text_lower = last_symptom_result['urgency'].lower()
                    if "high urgency" in urgency_text_lower or "critical" in urgency_text_lower : urgency_factor = 1
                    st.caption(f"Based on recent symptom analysis urgency: {last_symptom_result['urgency']}")

            base_times = ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]
            random.seed(appointment_date_val.strftime("%Y-%m-%d") + selected_doc) # Seed for consistency per day/doc

            existing_slots_for_day_doc = {
                appt['time'] for appt in st.session_state.appointments
                if isinstance(appt.get('date'), date) and appt.get('date') == appointment_date_val and appt.get('doctor') == selected_doc and appt.get('status') not in ["Cancelled", "Completed (Past due)"]
            }

            available_base_slots = [t for t in base_times if t not in existing_slots_for_day_doc]

            num_to_sample = len(available_base_slots)
            if selected_doc == "Any Available Doctor":
                 num_to_sample = random.randint(max(0, num_to_sample // 2), num_to_sample)
            else:
                 num_to_sample = random.randint(max(0, num_to_sample - random.randint(1,3)), num_to_sample)

            simulated_available_times_final = sorted(random.sample(available_base_slots, k=max(0, num_to_sample)))


            if simulated_available_times_final:
                suggestion_text = "General Suggestion"
                suggested_times_display = []
                if urgency_factor == 1:
                    suggested_times_display = simulated_available_times_final[:min(4, len(simulated_available_times_final))]
                    suggestion_text = "Higher Urgency Suggestion (earlier slots if available)"
                else:
                    mid_idx = len(simulated_available_times_final)//2
                    suggested_times_display = simulated_available_times_final[max(0,mid_idx-2):min(mid_idx+2,len(simulated_available_times_final))]
                    if not suggested_times_display: suggested_times_display = simulated_available_times_final[:min(4, len(simulated_available_times_final))] # Fallback if slice is empty

                st.info(f"{suggestion_text} for {appointment_date_val.strftime('%Y-%m-%d')}: Consider these available slots: {', '.join(suggested_times_display)}", icon="ℹ")
            else: st.info("No time slots seem available for this doctor on this date based on simulation. Try another date or doctor.")

            appointment_time_val = st.selectbox(
                "Preferred Time",
                simulated_available_times_final if simulated_available_times_final else ["No times available"],
                key="appt_time_select_book_tab_v6",
                disabled=not simulated_available_times_final
            )
            appointment_reason_val = st.text_area(
                "Reason for Visit",
                value=st.session_state.get('appointment_reason', ''),
                height=100,
                key="appt_reason_text_book_tab_v6"
            )
            book_submit = st.form_submit_button("Request Appointment", disabled=(not simulated_available_times_final or appointment_time_val == "No times available"))

            if book_submit:
                if not appointment_reason_val.strip(): st.warning("Please provide a reason for your visit.")
                else:
                    appt_id = f"appt_{datetime.datetime.now().timestamp()}_{random.randint(1000,9999)}"
                    new_appt_data = {
                        "id": appt_id, "doctor": selected_doc, "date": appointment_date_val,
                        "time": appointment_time_val, "reason": appointment_reason_val,
                        "status": "Upcoming", "timestamp": datetime.datetime.now()
                    }
                    st.session_state.appointments.append(new_appt_data)
                    st.session_state.appointment_reason = '' # Clear pre-fill
                    st.session_state.appointments_page_message = {"type": "success", "message": f"Appointment successfully requested with {selected_doc} on {appointment_date_val.strftime('%Y-%m-%d')} at {appointment_time_val}."}
                    alert_id_book = str(datetime.datetime.now().timestamp()) + "_appt_book_success_v6" # Unique ID
                    st.session_state.alerts.append({
                        "id": alert_id_book, "type": "success",
                        "message": f"Appointment requested: {selected_doc} on {appointment_date_val.strftime('%Y-%m-%d')} at {appointment_time_val}.",
                        "read": False, "timestamp": datetime.datetime.now()
                    })
                    st.rerun()

    with tab_upcoming:
        st.subheader("Upcoming Appointments")
        if st.session_state.rescheduling_appointment_id:
            # Find the specific appointment being rescheduled
            appt_to_reschedule_data = next((appt for appt in st.session_state.appointments if appt.get('id') == st.session_state.rescheduling_appointment_id), None)

            if appt_to_reschedule_data:
                st.markdown(f"#### Rescheduling Appointment for: {appt_to_reschedule_data.get('doctor')}")
                current_appt_dt_obj = appt_to_reschedule_data.get('date', date.today())
                st.markdown(f"Current Date: {current_appt_dt_obj.strftime('%A, %B %d, %Y')} at {appt_to_reschedule_data.get('time')}")
                st.markdown(f"Reason: {appt_to_reschedule_data.get('reason')}")

                # Calculate min allowed date: must be after today AND after original appointment date
                min_allowed_reschedule_date = max(date.today(), current_appt_dt_obj + timedelta(days=1)) # At least tomorrow, and after original date

                # Get default value for date input from session state, ensuring it's not before min allowed
                default_new_reschedule_date_val = st.session_state.get("new_reschedule_date_input_tab_v6_value", min_allowed_reschedule_date)
                if default_new_reschedule_date_val < min_allowed_reschedule_date:
                    default_new_reschedule_date_val = min_allowed_reschedule_date


                new_date_for_reschedule = st.date_input(
                    "Select New Date:",
                    value=default_new_reschedule_date_val, # Use default value
                    min_value=min_allowed_reschedule_date,
                    key="new_reschedule_date_input_tab_v6"
                )
                # Store the selected date value back into session state
                st.session_state.new_reschedule_date_input_tab_v6_value = new_date_for_reschedule


                col_confirm, col_cancel_reschedule = st.columns(2)
                with col_confirm:
                    if st.button("Confirm Reschedule", key="confirm_reschedule_btn_tab_v6"):
                        original_appt_dt_str = appt_to_reschedule_data['date'].strftime('%Y-%m-%d')
                        updated_in_list = False
                        for i, appt_loop_item in enumerate(st.session_state.appointments):
                            if appt_loop_item.get('id') == st.session_state.rescheduling_appointment_id:
                                st.session_state.appointments[i]['date'] = new_date_for_reschedule
                                st.session_state.appointments[i]['status'] = "Upcoming (Rescheduled)"
                                updated_in_list = True
                                break

                        if updated_in_list:
                            st.session_state.appointments_page_message = {"type": "success", "message": f"Appointment on {original_appt_dt_str} successfully rescheduled to {new_date_for_reschedule.strftime('%Y-%m-%d')}."}
                            alert_id_resched_success = str(datetime.datetime.now().timestamp()) + "_appt_resched_ok_v6" # Unique ID
                            st.session_state.alerts.append({
                                "id": alert_id_resched_success, "type": "success",
                                "message": f"Appointment with {appt_to_reschedule_data['doctor']} rescheduled to {new_date_for_reschedule.strftime('%Y-%m-%d')}.",
                                "read": False, "timestamp": datetime.datetime.now()
                            })
                        else:
                            st.session_state.appointments_page_message = {"type": "error", "message": "Failed to update appointment in list."}

                        st.session_state.rescheduling_appointment_id = None
                        st.session_state.pop("new_reschedule_date_input_tab_v6_value", None)
                        st.rerun()
                with col_cancel_reschedule:
                    if st.button("Cancel Reschedule Process", key="cancel_reschedule_btn_tab_v6"):
                        st.session_state.rescheduling_appointment_id = None
                        st.session_state.pop("new_reschedule_date_input_tab_v6_value", None)
                        st.session_state.appointments_page_message = {"type": "info", "message": "Reschedule process cancelled."}
                        st.rerun()
            else:
                st.error("Error: Could not find appointment to reschedule. Displaying list.");
                st.session_state.rescheduling_appointment_id = None # Reset
                st.session_state.pop("new_reschedule_date_input_tab_v6_value", None)
                st.rerun()
        else:
            # Filter for display, excluding any currently being rescheduled
            upcoming_for_display_now = sorted([
                appt for appt in st.session_state.appointments
                if isinstance(appt.get('date'), date) and appt['date'] >= date.today()
                and appt.get('status') not in ["Cancelled", "Completed (Past due)"]
            ], key=lambda x: (x.get('date', date.max), x.get('time', '23:59')))

            if not upcoming_for_display_now: st.info("No upcoming appointments to display.")
            else:
                for i, appt_disp_item in enumerate(upcoming_for_display_now):
                    appt_unique_id = appt_disp_item.get('id', f"no_id_tab_upcoming_v6_{i}")
                    appt_dt_obj = appt_disp_item.get('date', date.min)
                    appt_tm_str = appt_disp_item.get('time', 'N/A')
                    st.markdown(f"**{appt_dt_obj.strftime('%A, %B %d, %Y')}** at {appt_tm_str}") # Bold date
                    st.markdown(f"- Doctor: {appt_disp_item.get('doctor', 'N/A')}")
                    st.markdown(f"- Reason: {appt_disp_item.get('reason', 'N/A')}")
                    st.caption(f"Status: {appt_disp_item.get('status', 'N/A')}")
                    col_resched, col_cancel_appt = st.columns(2)
                    with col_resched:
                        # Only show reschedule if not currently in reschedule mode
                        if st.button("Reschedule", key=f"reschedule_btn_tab_upcoming_v6_{appt_unique_id}"):
                            st.session_state.rescheduling_appointment_id = appt_unique_id
                            current_appt_date_for_resched_init = appt_disp_item.get('date', date.today())
                            st.session_state.new_reschedule_date_input_tab_v6_value = max(date.today(), current_appt_date_for_resched_init + timedelta(days=1)) # Initialize default date input value
                            st.rerun()
                    with col_cancel_appt:
                         if st.button("Cancel Appointment", key=f"cancel_actual_btn_tab_upcoming_v6_{appt_unique_id}"):
                            cancelled_successfully = False
                            for j, appt_to_cancel_from_list in enumerate(st.session_state.appointments):
                                if appt_to_cancel_from_list.get('id') == appt_unique_id:
                                    st.session_state.appointments[j]['status'] = "Cancelled"
                                    st.session_state.appointments[j]['cancelled_timestamp'] = datetime.datetime.now()
                                    cancelled_successfully = True
                                    st.session_state.appointments_page_message = {"type": "success", "message": f"Appointment on {appt_dt_obj.strftime('%Y-%m-%d')} with {appt_to_cancel_from_list.get('doctor')} has been successfully cancelled."}
                                    alert_id_appt_cancelled_user = str(datetime.datetime.now().timestamp()) + "_user_cancel_appt_v6" # Unique ID
                                    st.session_state.alerts.append({
                                        "id": alert_id_appt_cancelled_user, "type": "info",
                                        "message": f"Appointment with {appt_to_cancel_from_list.get('doctor')} on {appt_dt_obj.strftime('%Y-%m-%d')} cancelled.",
                                        "read": False, "timestamp": datetime.datetime.now()
                                    })
                                    break
                            if not cancelled_successfully:
                                st.session_state.appointments_page_message = {"type": "error", "message": "Could not find the appointment to cancel."}
                            st.rerun()
                    st.divider()

    with tab_past_cancelled:
        st.subheader("Past (Completed) Appointments")

        past_completed_for_display_tab = sorted(
            [appt for appt in st.session_state.appointments if appt.get('status', '').startswith("Completed (Past due)")],
            key=lambda x: (x.get('date', date.min), x.get('time', '00:00')),
            reverse=True
        )

        if not past_completed_for_display_tab: st.info("No past (completed) appointments.")
        else:
            for i, appt_data_item in enumerate(past_completed_for_display_tab):
                appt_unique_id_past = appt_data_item.get('id', f"past_no_id_tab_v6_{i}")
                appt_dt_obj_past = appt_data_item.get('date', date.min)
                appt_tm_str_past = appt_data_item.get('time', 'N/A')
                status_past = appt_data_item.get('status', 'Completed (Past due)')
                st.markdown(f"**{appt_dt_obj_past.strftime('%A, %B %d, %Y')}** at {appt_tm_str_past}") # Bold date
                st.markdown(f"- Doctor: {appt_data_item.get('doctor', 'N/A')}")
                st.markdown(f"- Reason: {appt_data_item.get('reason', 'N/A')}")
                st.caption(f"Status: {status_past}")
                if st.button("View Summary (Simulated)", key=f"past_summary_btn_tab_v6_{appt_unique_id_past}"):
                    st.session_state.appointments_page_message = {"type": "info", "message": "Viewing appointment summaries is a simulated feature."}
                    st.rerun()
                st.divider()

        st.subheader("Cancelled Appointments History")
        cancelled_for_display_tab = sorted([
                appt for appt in st.session_state.appointments if appt.get('status') == "Cancelled"
            ],
            key=lambda x: (x.get('cancelled_timestamp', x.get('timestamp', datetime.datetime.min))),
            reverse=True
        )

        if not cancelled_for_display_tab: st.info("No cancelled appointments.")
        else:
            for i, appt_data_item_cancelled in enumerate(cancelled_for_display_tab):
                appt_unique_id_cancelled = appt_data_item_cancelled.get('id', f"cancelled_no_id_tab_v6_{i}")
                appt_dt_obj_cancelled = appt_data_item_cancelled.get('date', date.min)
                appt_tm_str_cancelled = appt_data_item_cancelled.get('time', 'N/A')
                cancelled_timestamp_obj = appt_data_item_cancelled.get('cancelled_timestamp', appt_data_item_cancelled.get('timestamp'))

                cancelled_timestamp_str_display = "N/A"
                if isinstance(cancelled_timestamp_obj, datetime.datetime):
                    cancelled_timestamp_str_display = cancelled_timestamp_obj.strftime('%Y-%m-%d %H:%M')

                st.markdown(f"Originally for: *{appt_dt_obj_cancelled.strftime('%A, %B %d, %Y')}* at {appt_tm_str_cancelled}")
                st.markdown(f"- Doctor: {appt_data_item_cancelled.get('doctor', 'N/A')}")
                st.markdown(f"- Reason: {appt_data_item_cancelled.get('reason', 'N/A')}")
                st.caption(f"Status: Cancelled (on {cancelled_timestamp_str_display})")
                st.divider()


def resources_learning_page():
    st.header("📚 Resources & Learning")
    st.write("Explore health topics using our AI-powered tool.")

    st.subheader("Health Topic Explorer")
    st.write("Enter a health topic (e.g., 'Diabetes', 'Migraine', 'Vitamin D Deficiency') for a general overview by AI.")

    if "health_topic_query_input_val_v6" not in st.session_state: st.session_state.health_topic_query_input_val_v6 = ""
    if "health_topic_overview_result_v6" not in st.session_state: st.session_state.health_topic_overview_result_v6 = None
    if "health_topic_overview_query_processed_v6" not in st.session_state: st.session_state.health_topic_overview_query_processed_v6 = None

    topic_query_input_val = st.text_input(
        "Enter health topic:",
        value=st.session_state.health_topic_query_input_val_v6,
        key="health_topic_query_input_resource_v6"
    )
    st.session_state.health_topic_query_input_val_v6 = topic_query_input_val # Update session state with current input

    if st.button("Get Information", key="get_topic_info_btn_resource_v6"):
        if topic_query_input_val.strip():
            st.session_state.health_topic_overview_query_processed_v6 = topic_query_input_val
            if model and tokenizer and tokenizer.pad_token_id is not None:
                with st.spinner(f"Fetching info on '{topic_query_input_val}'..."):
                    overview_content = get_llm_health_topic_info(topic_query_input_val)
                    st.session_state.health_topic_overview_result_v6 = overview_content
            elif not model or not tokenizer:
                st.session_state.health_topic_overview_result_v6 = "AI model not available to fetch health topic information."
            else:
                st.session_state.health_topic_overview_result_v6 = "AI information unavailable: Model tokenizer is missing a pad token."
        else:
            st.warning("Please enter a health topic.")
            st.session_state.health_topic_overview_result_v6 = None
            st.session_state.health_topic_overview_query_processed_v6 = None
        st.rerun()

    if st.session_state.health_topic_overview_result_v6 and \
       st.session_state.health_topic_overview_query_processed_v6 == st.session_state.health_topic_query_input_val_v6: # Only display if query matches input field
        query_title_display = st.session_state.health_topic_overview_query_processed_v6
        overview_content_display = st.session_state.health_topic_overview_result_v6

        error_messages = ["AI model not available", "tokenizer is missing a pad token", "An error occurred"]
        if any(err_msg in overview_content_display for err_msg in error_messages):
            st.error(f"Could not fetch information for '{query_title_display}': {overview_content_display}")
        elif "Could not generate information for this topic" in overview_content_display:
            st.info(f"For '{query_title_display}': {overview_content_display}")
        else:
            st.markdown(
                f"<div style='background-color: #f9f9f9; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin-top:10px;'><h4>Overview for: {query_title_display}</h4><p>{overview_content_display}</p></div>",
                unsafe_allow_html=True
            )

    st.caption("AI-powered educational information. Always consult your doctor for medical advice, diagnosis, or treatment.")


# --- Main App Execution and Navigation ---
if __name__ == "__main__":
    if st.session_state.authenticated:
        # User is logged in, show the main app interface
        with st.sidebar:
            st.title("⚕ Health AI Nav")
            st.markdown("---")
            pages = {
                'Home': '🏠 Home',
                'SymptomChecker': '🩺 Symptom Analysis',
                'MyHealthData': '📊 My Health Data',
                'Appointments': '📅 Appointments',
                'Resources': '📚 Resources'
            }
            for page_key_nav, page_title_nav in pages.items():
                button_type = "primary" if st.session_state.current_page == page_key_nav else "secondary"
                if st.button(page_title_nav, key=f"nav_btn_main_v6_{page_key_nav}", type=button_type, use_container_width=True):
                    set_page(page_key_nav)
            st.markdown("---")
            if st.button("Logout", key="logout_button_v6", use_container_width=True):
                logout() # Call logout function and rerun
            st.markdown("---"); st.write("AI Model Status:")
            if model and tokenizer: st.success(f"Model Loaded ({str(device).upper()})", icon="✅")
            else: st.error("Model Failed to Load", icon="❌"); st.caption("AI features may be unavailable.")


        # Page rendering logic based on session state
        page_to_render_main = st.session_state.current_page
        if   page_to_render_main == 'Home': home_page()
        elif page_to_render_main == 'SymptomChecker': symptom_checker_page()
        elif page_to_render_main == 'MyHealthData': my_health_data_page()
        elif page_to_render_main == 'Appointments': appointments_page()
        elif page_to_render_main == 'Resources': resources_learning_page()
        else:
            # Fallback if session state is invalid
            st.session_state.current_page = 'Home'
            home_page()

    else:
        # User is not logged in, show the login page
        login_page()
        