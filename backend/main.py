import os
import sqlite3
import json
import faiss
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from passlib.context import CryptContext
from jose import JWTError, jwt

# Load environment configuration
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_NAME = 'gp_database.db'

# Security Configuration
SECRET_KEY = "YOUR_SUPER_SECRET_KEY_HERE"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Attempt to load the RAG knowledge base (Vector Store + JSON)
# If this fails, the application will degrade gracefully but triage will be limited.
try:
    faiss_index = faiss.read_index("nhs_index.faiss")
    with open("nhs_data.json", "r") as f:
        nhs_data = json.load(f)
    print("Knowledge base loaded successfully.")
except Exception as e:
    print(f"WARNING: Could not load knowledge base: {e}")
    faiss_index = None
    nhs_data = []

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialise_database():
    """Sets up the SQLite schema if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    
    # Users table: Stores registered patient credentials and personal information.
    # We use a unique constraint on username to prevent duplicate registrations.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT,
        dob TEXT,
        address TEXT
    )
    ''')
    
    # Appointments table: Stores the core slot data.
    # We use a unique constraint on slot_time to prevent double-booking at the schema level.
    # Now includes a foreign key relationship to link slots to authenticated users.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY,
        slot_time TEXT NOT NULL UNIQUE,
        is_booked BOOLEAN NOT NULL DEFAULT 0,
        user_id INTEGER,
        patient_name TEXT,
        booking_reference TEXT UNIQUE,
        booked_at TEXT,
        appointment_type TEXT DEFAULT 'standard',
        notes TEXT,
        status TEXT DEFAULT 'available',
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Audit trail: Tracks all changes (booking, cancellation) for compliance.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS booking_history (
        id INTEGER PRIMARY KEY,
        appointment_id INTEGER,
        action TEXT NOT NULL,
        patient_name TEXT,
        booking_reference TEXT,
        timestamp TEXT NOT NULL,
        reason TEXT,
        FOREIGN KEY (appointment_id) REFERENCES appointments (id)
    )
    ''')

    # Messages table: Stores chat history to prevent context loss.
    # Indexed by user_id implicitly via the foreign key to speed up retrieval.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def generate_appointment_slots():
    """
    Populates the database with empty slots for the next 5 business days.
    Logic: 9am-12pm and 2pm-5pm, every 20 minutes.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    current_date = datetime.now() + timedelta(days=1)
    current_date = current_date.replace(hour=9, minute=0, second=0, microsecond=0)
    slots_generated = 0
    days_checked = 0
    
    # Generate a rolling window of 5 days
    while slots_generated < 5 and days_checked < 10:
        if current_date.weekday() < 5: # Monday (0) to Friday (4)
            # Define clinic hours
            hours = [9, 10, 11, 14, 15, 16]
            minutes = [0, 20, 40]
            
            for hour in hours:
                for minute in minutes:
                    slot_datetime = current_date.replace(hour=hour, minute=minute)
                    slot_time = slot_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    try:
                        cursor.execute(
                            "INSERT INTO appointments (slot_time, appointment_type) VALUES (?, ?)", 
                            (slot_time, 'standard')
                        )
                    except sqlite3.IntegrityError:
                        # Slot already exists; ignore
                        pass
            slots_generated += 1
            
        current_date += timedelta(days=1)
        days_checked += 1
    
    conn.commit()
    conn.close()
    print(f"Generated appointment slots for the next 5 business days.")

# Perform initial setup on module load
initialise_database()
generate_appointment_slots()

def verify_password(plain_password, hashed_password):
    """Verifies a plaintext password against its bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Generates a secure bcrypt hash for a new password."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a signed JWT token for stateless API authentication."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency function to extract and validate the current user from the JWT.
    Returns the user database record as a dictionary if successful.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    
    if user is None:
        raise credentials_exception
    return dict(user)

def get_available_appointments():
    """
    Retrieves and formats available slots for the LLM.
    Returns a grouped natural language string to help the model present data clearly.
    """
    conn = get_db_connection()
    # Fetch only future, unbooked slots. Limit to 60 to prevent token overflow.
    appointments = conn.execute("""
        SELECT id, slot_time, appointment_type 
        FROM appointments 
        WHERE is_booked = 0 
        AND datetime(slot_time) > datetime('now')
        ORDER BY slot_time
        LIMIT 60
    """).fetchall()
    conn.close()
    
    if not appointments:
        return json.dumps({"status": "none_available", "message": "There are no available appointments at this time."})
    
    grouped_for_display = {}
    
    # Format data for the LLM consumption
    for appt in appointments:
        slot_datetime = datetime.strptime(appt['slot_time'], '%Y-%m-%d %H:%M:%S')
        date_key = slot_datetime.strftime('%A, %B %d, %Y')
        time_12hr = slot_datetime.strftime('%I:%M %p').lstrip('0')
        
        if date_key not in grouped_for_display:
            grouped_for_display[date_key] = {'date': date_key, 'times': []}
        
        grouped_for_display[date_key]['times'].append(time_12hr)
    
    # Construct the display string
    display_text = []
    for date, info in list(grouped_for_display.items())[:5]:
        display_text.append(f"### {date}")
        for time in info['times'][:18]:
            display_text.append(f"- {time}")
    
    return json.dumps({
        "status": "available",
        "formatted_display": "\n".join(display_text),
        "appointments_grouped": grouped_for_display
    })

def book_appointment_by_datetime(user_id: int, patient_name: str, requested_day: str, requested_time: str):
    """
    Attempts to match a fuzzy natural language date request to a specific database slot.
    Booking logic securely links the confirmed slot to the authenticated user ID.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, slot_time 
            FROM appointments 
            WHERE is_booked = 0 
            AND datetime(slot_time) > datetime('now')
            ORDER BY slot_time
        """)
        available_slots = cursor.fetchall()
        
        if not available_slots:
            conn.close()
            return json.dumps({"status": "error", "message": "No available appointments found."})
        
        # Normalise input to uppercase for comparison
        requested_time_clean = requested_time.strip().upper()
        
        for slot in available_slots:
            slot_datetime = datetime.strptime(slot['slot_time'], '%Y-%m-%d %H:%M:%S')
            slot_day = slot_datetime.strftime('%A')
            slot_time_12hr = slot_datetime.strftime('%I:%M %p').lstrip('0').upper()
            slot_date_full = slot_datetime.strftime('%A, %B %d, %Y')
            
            # Flexible matching: accepts "Monday" or "Monday, November 17"
            day_match = (requested_day.lower() in slot_day.lower() or requested_day.lower() in slot_date_full.lower())
            time_match = requested_time_clean in slot_time_12hr
            
            if day_match and time_match:
                # Generate a pseudo-random booking reference
                booking_ref = f"GP{slot['id']:04d}{datetime.now().strftime('%H%M')}"
                booked_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Update the main record with the authenticated user ID
                cursor.execute("""
                    UPDATE appointments 
                    SET is_booked = 1, 
                        user_id = ?,
                        patient_name = ?, 
                        booking_reference = ?,
                        booked_at = ?,
                        status = 'confirmed'
                    WHERE id = ?
                """, (user_id, patient_name, booking_ref, booked_at, slot['id']))
                
                # Write to audit log
                cursor.execute("""
                    INSERT INTO booking_history (appointment_id, action, patient_name, booking_reference, timestamp)
                    VALUES (?, 'booked', ?, ?, ?)
                """, (slot['id'], patient_name, booking_ref, booked_at))
                
                conn.commit()
                conn.close()
                
                return json.dumps({
                    "status": "success",
                    "confirmation": f"Appointment Confirmed!\nRef: {booking_ref}\nTime: {slot_datetime.strftime('%A, %B %d at %I:%M %p')}"
                })
        
        conn.close()
        return json.dumps({"status": "error", "message": f"Couldn't find slot for {requested_day} at {requested_time}."})
        
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Booking error: {str(e)}"})

def cancel_appointment(booking_reference: str = None, user_id: int = None):
    """
    Cancels an existing appointment.
    Ensures that users can only cancel appointments tied to their specific user account.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Only allow cancellation if the appointment belongs to the authenticated user
        cursor.execute("""
            SELECT id, slot_time, patient_name, booking_reference 
            FROM appointments 
            WHERE booking_reference = ? AND user_id = ? AND is_booked = 1
        """, (booking_reference, user_id))
        
        appointment = cursor.fetchone()
        
        if not appointment:
            conn.close()
            return json.dumps({"status": "error", "message": "No booking found with that reference for your account."})
        
        # Reset the slot to available and remove user bindings
        cursor.execute("""
            UPDATE appointments 
            SET is_booked = 0, 
                user_id = NULL,
                patient_name = NULL, 
                booking_reference = NULL,
                booked_at = NULL,
                status = 'available'
            WHERE id = ?
        """, (appointment['id'],))
        
        # Log the cancellation
        cursor.execute("""
            INSERT INTO booking_history (appointment_id, action, patient_name, booking_reference, timestamp)
            VALUES (?, 'cancelled', ?, ?, ?)
        """, (appointment['id'], appointment['patient_name'], appointment['booking_reference'], datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        conn.commit()
        conn.close()
        return json.dumps({"status": "success", "message": "Appointment cancelled successfully."})
        
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Cancellation error: {str(e)}"})

def get_triage_recommendation_from_kb(symptom_description: str):
    """
    RAG Implementation:
    1. Vectorises the symptom description.
    2. Searches the FAISS index for relevant NHS content chunks.
    3. Evaluates the distance score of the best match.
    4. If the distance is too high (weak match), it forces the LLM to trigger a human fallback.

    """
    if not faiss_index or not nhs_data:
        return json.dumps({"status": "error", "message": "Knowledge base not loaded."})

    try:
        query_embedding = client.embeddings.create(
            input=[symptom_description], model="text-embedding-3-small"
        ).data[0].embedding
        query_vector = np.array([query_embedding]).astype("float32")

        distances, indices = faiss_index.search(query_vector, 3)

        # Extract the distance of the absolute best (closest) match.
        best_match_distance = float(distances[0][0])

        print(f"\n---> FAISS DISTANCE SCORE: {best_match_distance} <---", flush=True)

        # Define the threshold
        DISTANCE_THRESHOLD = 1.15

        if best_match_distance > DISTANCE_THRESHOLD:
            # The match is too weak. Intercept the data before the LLM can guess.
            return json.dumps({
                "status": "low_confidence", 
                "message": "The system could not find a confident match for these symptoms. You MUST immediately trigger the HUMAN FALLBACK and ask the user to call the surgery on 01632 960000.",
                "best_match_distance": best_match_distance
            })



        relevant_chunks = [nhs_data[i] for i in indices[0]]
        return json.dumps({"status": "found", "context": relevant_chunks})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

# FastAPI Configuration
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserCreate(BaseModel):
    username: str
    password: str
    full_name: str
    dob: str
    address: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    message: str
    history: list

@app.post("/register", response_model=Token)
async def register(user: UserCreate):
    """Endpoint for new patients to create an account in the system."""
    conn = get_db_connection()
    try:
        hashed_password = get_password_hash(user.password)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password, full_name, dob, address) VALUES (?, ?, ?, ?, ?)",
            (user.username, hashed_password, user.full_name, user.dob, user.address)
        )
        conn.commit()
        access_token = create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already registered")
    finally:
        conn.close()

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Endpoint to exchange valid user credentials for a JWT access token."""
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (form_data.username,)).fetchone()
    conn.close()
    
    if not user or not verify_password(form_data.password, user['password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['username']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """Returns profile data for the currently authenticated user."""
    return {
        "username": current_user['username'], 
        "full_name": current_user['full_name'],
        "dob": current_user['dob']
    }

@app.get("/chat/history")
async def fetch_history(current_user: dict = Depends(get_current_user)):
    """Retrieves the last 50 messages for the authenticated user to restore chat context."""
    conn = get_db_connection()
    
    # Use a subquery to grab the newest 50 (DESC), then re-sort them chronologically (ASC)
    messages = conn.execute("""
        SELECT role, content, timestamp 
        FROM (
            SELECT id, role, content, timestamp 
            FROM messages 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 50
        )
        ORDER BY timestamp ASC 
    """, (current_user['id'],)).fetchall()
    
    conn.close()
    
    # Convert the SQLite rows into standard dictionaries for JSON response
    return [dict(msg) for msg in messages]

@app.get("/appointments")
async def list_appointments(current_user: dict = Depends(get_current_user)):
    """Endpoint for the user dashboard. Returns ONLY the logged-in user's appointments."""
    conn = get_db_connection()
    appointments = conn.execute("""
        SELECT id, slot_time, booking_reference, appointment_type, status, patient_name
        FROM appointments
        WHERE user_id = ?
        ORDER BY slot_time
    """, (current_user['id'],)).fetchall()
    conn.close()
    return [dict(appt) for appt in appointments]

@app.post("/appointments/refresh")
async def refresh_appointments():
    """Admin/Dev endpoint to reset the appointment database."""
    try:
        generate_appointment_slots()
        return {"status": "success", "message": "Appointment slots have been refreshed for the next 5 business days."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to refresh appointments: {str(e)}"}
    


@app.post("/chat")
async def handle_chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    user_message = request.message
    history = request.history

    # save the user's incoming message
    conn = get_db_connection()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn.execute(
        "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (current_user['id'], 'user', user_message, current_time)
    )
    conn.commit()

    
    # System prompt defines the persona, STRICT safety guardrails, and injects authenticated user context.
    # Note: Moving this to a config file would be a future improvement.
    system_prompt = {
        "role": "system",
        "content": f"""You are a helpful GP surgery assistant.
You are speaking with {current_user['full_name']} (DOB: {current_user['dob']}).

CRITICAL SAFETY RULES & RED FLAG CONFIRMATION (HIGHEST PRIORITY):
1. Monitor for potential "red flag" symptoms: chest pain, difficulty breathing, stroke symptoms (FAST), severe bleeding, loss of consciousness.
2. CONFIRMATION STEP (Reducing False Positives): 
   - If the user mentions a red flag symptom but the context is ambiguous, historical, or mild (e.g., "I had chest pain yesterday", "my chest hurts when I cough with this cold"), DO NOT immediately trigger an emergency.
   - Instead, ask ONE direct clarifying question to determine if it is an acute, severe medical emergency right now (e.g., "Just to be safe, are you experiencing severe, crushing chest pain right now?").
3. TRIGGERING THE ALARM: 
   - ONLY if the user CONFIRMS the symptom is currently severe/life-threatening, OR if their initial message is unambiguously an active emergency (e.g., "I think I'm having a heart attack"), you MUST respond ONLY with the exact phrase: "EMERGENCY_DETECTED". 
   - Do not provide any other text when triggering the alarm.

OUT-OF-BOUNDS QUERIES (STRICT GUARDRAIL):
1. You are strictly a GP surgery assistant. You must ONLY discuss medical symptoms, NHS triage advice, and appointment management.
2. If the user asks about ANYTHING unrelated to these topics (e.g., programming, general knowledge, politics, creative writing, or financial advice), you MUST politely refuse to answer.
3. Example refusal: "I am a medical assistant, so I can only help you with health-related concerns and booking GP appointments. How can I help you with your health today?"

HUMAN FALLBACK & UNCERTAINTY :
1. If the `get_triage_recommendation_from_kb` tool returns results that do not confidently match the user's symptoms, or if the symptoms are too complex and require multiple conflicting possibilities, DO NOT guess.
2. You must immediately trigger a human fallback response to prioritize patient safety.
3. Example fallback: "Based on what you've told me, my system isn't completely certain about the best advice for your specific symptoms. To ensure you get the safest care, please call the GP surgery directly on 01632 960000 to speak with our reception team or a clinician."

DYNAMIC TRIAGE WORKFLOW (STRICT TOOL USAGE):
1. ASSESS THE INPUT: Read the user's message. Does it contain specific physical or mental symptoms (e.g., "sore throat, cough, back pain") or is it vague/missing (e.g., "I feel unwell", "I need an appointment")?
2. GATHER (If necessary): If symptoms are missing or too vague to run a search, ask ONE direct clarifying question to gather them. Do NOT ask redundant follow-up questions if the user has already provided specific symptoms.
3. SEARCH (When ready): As soon as you have specific symptoms, you MUST IMMEDIATELY call `get_triage_recommendation_from_kb`. Do not delay or ask for further confirmation.
4. NO GUESSING: DO NOT rely on your internal knowledge to offer medical advice, guess conditions, or suggest treatments. You must only provide advice based on the output of the tool.
5. RESOLUTION: If the tool returns a `low_confidence` status, immediately trigger the human fallback. If it returns a confident match (e.g., advice is 'GP' or 'Pharmacist'), present that advice to the patient and transition to booking if 'GP' is advised.

BOOKING WORKFLOW (STRICTLY AFTER TRIAGE):
1. PREREQUISITE: You MUST NOT offer or book an appointment unless you have gathered symptoms AND the `get_triage_recommendation_from_kb` tool has specifically advised seeing a GP. 
2. If a user asks for an appointment but hasn't given symptoms, politely explain that you need to do a quick symptom check first.
3. Call `get_available_appointments`.
4. Ask user to pick a time (e.g., "Monday at 9am").
5. Call `book_appointment_by_datetime`. 
   IMPORTANT: You already know their name is {current_user['full_name']}, so DO NOT ask for it.

CANCELLATION:
1. Ask for the booking reference.
2. Call `cancel_appointment`.

Be warm, professional, and concise."""
    }
    
    messages = [system_prompt] + history + [{"role": "user", "content": user_message}]
    
    # Wrapper functions to securely inject the authenticated user_id without relying on the LLM
    def book_appt_wrapper(requested_day, requested_time, patient_name=None):
        return book_appointment_by_datetime(current_user['id'], current_user['full_name'], requested_day, requested_time)
        
    def cancel_appt_wrapper(booking_reference):
        return cancel_appointment(booking_reference, current_user['id'])

    # Define available tools (Functions) for the OpenAI model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_triage_recommendation_from_kb",
                "description": "CRITICAL: You MUST call this tool whenever the user mentions ANY physical symptoms, mental symptoms, or feeling unwell, no matter how vague or minor.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symptom_description": {"type": "string", "description": "Patient symptoms."}
                    },
                    "required": ["symptom_description"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_available_appointments",
                "description": "Retrieves available appointment slots. CRITICAL: ONLY call this AFTER completing triage. Do NOT call this tool if the user has not provided symptoms yet, even if they explicitly ask for an appointment."
            }
        },
        {
            "type": "function",
            "function": {
                "name": "book_appointment_by_datetime",
                "description": "Books a GP appointment. Do NOT ask for patient name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requested_day": {"type": "string", "description": "Day wanted (e.g. Monday)."},
                        "requested_time": {"type": "string", "description": "Time wanted (e.g. 9:00 AM)."},
                    },
                    "required": ["requested_day", "requested_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_appointment",
                "description": "Cancels a booking.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "booking_reference": {"type": "string", "description": "Booking reference code."}
                    },
                    "required": ["booking_reference"],
                },
            },
        },
    ]
    
    available_functions = {
        "get_triage_recommendation_from_kb": get_triage_recommendation_from_kb,
        "get_available_appointments": get_available_appointments,
        "book_appointment_by_datetime": book_appt_wrapper,
        "cancel_appointment": cancel_appt_wrapper,
    }

    try:
        # Tool-calling loop with safety limit to prevent infinite recursion
        iteration = 0
        while iteration < 5:
            iteration += 1
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if not tool_calls:
                # LLM has decided on a final text response
                agent_response = response_message.content
                break
            
            messages.append(response_message)
            
            # Execute tool logic
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                
                if not function_to_call:
                    function_response = json.dumps({"error": f"Unknown function: {function_name}"})
                else:
                    try:
                        if tool_call.function.arguments:
                            function_args = json.loads(tool_call.function.arguments)
                            function_response = function_to_call(**function_args)
                        else:
                            function_response = function_to_call()
                        
                    except TypeError as e:
                         # Handle cases where LLM sends extra/wrong args
                         function_response = json.dumps({"status": "error", "message": f"Argument error: {str(e)}"})

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
        
        else:
            agent_response = "I'm having trouble processing that right now. Could you try again?"

    except Exception as e:
        print(f"Chat Error: {e}")
        agent_response = "I encountered an error. Please try again."

    # Save the AI's final response
    response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn.execute(
        "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (current_user['id'], 'assistant', agent_response, response_time)
    )
    conn.commit()
    conn.close()

    # emergency intercept 

    if "EMERGENCY_DETECTED" in agent_response:
        return {
            "response": "⚠️ URGENT: Your symptoms require immediate medical assessment.",
            "status": "emergency"
        }

    return {
        "response": agent_response,
        "status": "normal"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)