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


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_NAME = 'gp_database.db'

# Security 
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Attempt to load the RAG knowledge base (Vector Store + JSON)
# If this fails, the application will fallback but triage will be limited.
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
    # Sets up the SQLite schema if it doesn't exist.
    conn = get_db_connection()
    cursor = conn.cursor()

    
    # Users table, Stores registered patient credentials and personal information.
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

    # Messages table, Stores chat history to prevent context loss.
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

    #add new columns for the profile if they don't exist yet
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN medical_notes TEXT DEFAULT 'No known allergies. Routine health checks up to date.'")
        cursor.execute("ALTER TABLE users ADD COLUMN prescriptions TEXT DEFAULT 'None active.'")
    except sqlite3.OperationalError:
        pass # Columns already exist
        
    #  add the source column to messages
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN source TEXT")
    except sqlite3.OperationalError:
        pass 
        
    conn.commit()
    conn.close()

def generate_appointment_slots():

    # Populates the database with empty slots for the next 5 business days.


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
                        # Slot already exists, ignore
                        pass
            slots_generated += 1
            
        current_date += timedelta(days=1)
        days_checked += 1
    
    conn.commit()
    conn.close()
    print(f"Generated appointment slots for the next 5 business days.")


initialise_database()
generate_appointment_slots()

def verify_password(plain_password, hashed_password):
    # Verifies a plaintext password against its bcrypt hash.
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
     # Generates a secure bcrypt hash for a new password.
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
     #Creates a signed JWT token for stateless API authentication.
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):

    # Dependency function to extract and validate the current user from the JWT.
    # Returns the user database record as a dictionary if successful.

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

    # Retrieves and formats available slots for the LLM.

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

def book_appointment_by_datetime(user_id: int, patient_name: str, requested_day: str, requested_time: str, notes: str = None):

    # Booking logic securely links the confirmed slot to the authenticated user ID.

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
                        status = 'confirmed',
                        notes = ?
                    WHERE id = ?
                """, (user_id, patient_name, booking_ref, booked_at, notes, slot['id']))
                
                # Write to audit log
                cursor.execute("""
                    INSERT INTO booking_history (appointment_id, action, patient_name, booking_reference, timestamp)
                    VALUES (?, 'booked', ?, ?, ?)
                """, (slot['id'], patient_name, booking_ref, booked_at))
                
                conn.commit()
                conn.close()
                
                return json.dumps({
                    "status": "success",
                    "confirmation": f"Appointment Confirmed!\nTime: {slot_datetime.strftime('%A, %B %d at %I:%M %p')}"
                })
        
        conn.close()
        return json.dumps({"status": "error", "message": f"Couldn't find slot for {requested_day} at {requested_time}."})
        
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Booking error: {str(e)}"})

def cancel_appointment(user_id: int, requested_day: str, requested_time: str):


   # Ensures that users can only cancel appointments tied to their specific user account.

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Fetch user's currently booked appointments
        cursor.execute("""
            SELECT id, slot_time, patient_name, booking_reference 
            FROM appointments 
            WHERE user_id = ? AND is_booked = 1
        """, (user_id,))
        
        user_appointments = cursor.fetchall()
        
        if not user_appointments:
            conn.close()
            return json.dumps({"status": "error", "message": "You currently have no booked appointments."})
        
        # Normalise input for comparison
        requested_time_clean = requested_time.strip().upper()
        appointment_to_cancel = None

        for appt in user_appointments:
            slot_datetime = datetime.strptime(appt['slot_time'], '%Y-%m-%d %H:%M:%S')
            slot_day = slot_datetime.strftime('%A')
            slot_time_12hr = slot_datetime.strftime('%I:%M %p').lstrip('0').upper()
            slot_date_full = slot_datetime.strftime('%A, %B %d, %Y')
            
            day_match = (requested_day.lower() in slot_day.lower() or requested_day.lower() in slot_date_full.lower())
            time_match = requested_time_clean in slot_time_12hr
            
            if day_match and time_match:
                appointment_to_cancel = appt
                break
                
        if not appointment_to_cancel:
            conn.close()
            return json.dumps({"status": "error", "message": f"Could not find an appointment on {requested_day} at {requested_time} for your account."})
        
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
        """, (appointment_to_cancel['id'],))
        
        # Log the cancellation
        cursor.execute("""
            INSERT INTO booking_history (appointment_id, action, patient_name, booking_reference, timestamp)
            VALUES (?, 'cancelled', ?, ?, ?)
        """, (appointment_to_cancel['id'], appointment_to_cancel['patient_name'], appointment_to_cancel['booking_reference'], datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        conn.commit()
        conn.close()
        return json.dumps({"status": "success", "message": f"Your appointment on {requested_day} at {requested_time} has been cancelled successfully."})
        
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

        # Extract the distance of the absolute best match.
        best_match_distance = float(distances[0][0])

        print(f"\n---> FAISS DISTANCE SCORE: {best_match_distance} <---", flush=True)

        # Define the threshold
        DISTANCE_THRESHOLD = 1.15

        if best_match_distance > DISTANCE_THRESHOLD:
            # The match is too weak. Intercept the data before the LLM can guess.
            return json.dumps({
                "status": "low_confidence", 
                "message": "The NHS knowledge base could not find a confident match. You MUST still directly answer the user's question using general medical knowledge, following the HUMAN FALLBACK & UNCERTAINTY rules. An advisory card with a verified source will appear below your message automatically.",
                "best_match_distance": best_match_distance
            })



        relevant_chunks = [nhs_data[i] for i in indices[0]]
        return json.dumps({"status": "found", "context": relevant_chunks})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
    
def get_general_medical_advisory(symptom_description: str) -> Optional[dict]:
    """
    Multi-Agent Advisory Pipeline (triggered ONLY on RAG low_confidence):
    
    Agent 1 (Advisory): Generates general medical advice from LLM knowledge,
                        citing a specific, verifiable medical source URL.
    Agent 2 (Verifier): Cross-checks the advice and source for legitimacy,
                        rejecting hallucinated or unreliable sources.
    
    Returns a dict with advice, source_name, source_url, and condition or None if rejected.
    """
    try:
        # Agent 1: Advisory Agent
        advisory_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a medical information assistant. The user has asked a specific medical question. You MUST directly answer their question.

Given the user's question, provide:
1. A DIRECT answer to their specific question (2-3 sentences max). Do NOT give generic information about the condition — answer exactly what was asked.
   - If they ask "can I take antibiotics?" → explain whether antibiotics are appropriate for this situation and why/why not.
   - If they ask "should I use ice or heat?" → explain which is recommended and when.
   - If they ask "can I exercise?" → explain what activity is safe and what to avoid.
   - If they describe symptoms → explain what common causes may be and what to do.
2. A SPECIFIC source URL from a trusted medical website.

CRITICAL RULES:
- ANSWER THE QUESTION FIRST. Do not deflect or give vague advice.
- ONLY cite sources you are highly confident exist. Use well-known condition pages.
- You may cite ANY reputable medical institution including but not limited to: Mayo Clinic, Cleveland Clinic, NHS.uk, WHO, BMJ, MedlinePlus, WebMD, Patient.info, American Academy of Family Physicians.
- Prefer URLs in the format: https://www.mayoclinic.org/diseases-conditions/[condition]/symptoms-causes/syc-[id]
  or https://www.nhs.uk/conditions/[condition]/
  or https://www.who.int/news-room/fact-sheets/detail/[condition]
  or https://medlineplus.gov/[condition].html
- Do NOT invent or guess URLs. If unsure, use a general trusted page.
- Do NOT diagnose. But DO answer the specific question asked.
- Always recommend seeing a healthcare professional for confirmation.

Respond in EXACTLY this JSON format and nothing else:
{
  "advice": "Your direct answer to the question here.",
  "source_name": "Source Name",
  "source_url": "https://...",
  "condition_topic": "Brief topic name"
}"""
                },
                {
                    "role": "user",
                    "content": f"User question: {symptom_description}"
                }
            ],
            temperature=0.2,
        )

        advisory_raw = advisory_response.choices[0].message.content.strip()
        # Clean markdown fences if present
        advisory_raw = advisory_raw.replace("```json", "").replace("```", "").strip()
        advisory_data = json.loads(advisory_raw)

        # Validate required fields exist
        if not all(k in advisory_data for k in ["advice", "source_name", "source_url", "condition_topic"]):
            print("Advisory agent returned incomplete data.")
            return None

        # Agent 2: Verification Agent 
        verification_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a medical source verification agent. You will receive:
- A piece of medical advice
- A source name and URL
- The original user question

Your job is to verify:
1. Is the source a REAL, trusted medical institution? (Mayo Clinic, Cleveland Clinic, NHS, WHO, BMJ, MedlinePlus, WebMD, Patient.info, American Academy of Family Physicians, etc.)
2. Is the URL plausible and well-formed for that institution? (correct domain, reasonable path structure)
3. Is the advice safe, non-diagnostic, and generally aligned with mainstream medical guidance?
4. Does the advice DIRECTLY answer the question asked? (not just generic info about the condition)

Respond in EXACTLY this JSON format and nothing else:
{
  "is_verified": true or false,
  "rejection_reason": "null if verified, otherwise explain why"
}"""
                },
                {
                    "role": "user",
                    "content": f"""ADVICE: {advisory_data['advice']}
SOURCE: {advisory_data['source_name']} — {advisory_data['source_url']}
ORIGINAL QUESTION: {symptom_description}"""
                }
            ],
            temperature=0.0,
        )

        verify_raw = verification_response.choices[0].message.content.strip()
        verify_raw = verify_raw.replace("```json", "").replace("```", "").strip()
        verify_data = json.loads(verify_raw)

        if not verify_data.get("is_verified"):
            print(f"Verification agent REJECTED advisory: {verify_data.get('rejection_reason')}")
            return None

        print(f"✓ Advisory verified: {advisory_data['source_name']} — {advisory_data['source_url']}")
        return advisory_data

    except json.JSONDecodeError as e:
        print(f"Advisory pipeline JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"Advisory pipeline error: {e}")
        return None

def get_user_bookings(user_id: int):
    """Returns the user's current bookings formatted for the cancellation picker."""
    conn = get_db_connection()
    bookings = conn.execute("""
        SELECT id, slot_time, booking_reference
        FROM appointments
        WHERE user_id = ? AND is_booked = 1
        AND datetime(slot_time) > datetime('now')
        ORDER BY slot_time
    """, (user_id,)).fetchall()
    conn.close()

    if not bookings:
        return json.dumps({"status": "none", "message": "You have no current bookings."})

    formatted = []
    for b in bookings:
        slot_dt = datetime.strptime(b['slot_time'], '%Y-%m-%d %H:%M:%S')
        formatted.append({
            "id": b['id'],
            "slot_time": b['slot_time'],
            "booking_reference": b['booking_reference'],
            "display_date": slot_dt.strftime('%A, %d %b'),
            "display_time": slot_dt.strftime('%I:%M %p').lstrip('0'),
        })

    return json.dumps({"status": "found", "bookings": formatted})


def get_reschedule_data(user_id: int):
    # Returns current bookings + available slots in one payload for the reschedule widget.
    bookings_response = json.loads(get_user_bookings(user_id))
    slots_response = json.loads(get_available_appointments())

    if bookings_response.get("status") == "none":
        return json.dumps({"status": "no_bookings", "message": "You have no current bookings to reschedule."})

    if slots_response.get("status") != "available":
        return json.dumps({"status": "no_slots", "message": "There are no available slots to reschedule to."})

    return json.dumps({
        "status": "ready",
        "current_bookings": bookings_response["bookings"],
        "available_slots": slots_response["appointments_grouped"]
    })


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
    # Endpoint for new patients to create an account in the system.
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
    # Endpoint to exchange valid user credentials for a JWT access token.
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
    # Returns profile data for the currently authenticated user.
    return {
        "username": current_user['username'], 
        "full_name": current_user['full_name'],
        "dob": current_user['dob'],
        "address": current_user.get('address', ''),
        "medical_notes": current_user.get('medical_notes', 'No known allergies. Routine health checks up to date.'),
        "prescriptions": current_user.get('prescriptions', 'None active.')
    }

class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    full_name: Optional[str] = None
    address: Optional[str] = None

@app.put("/users/me")
async def update_user_profile(update_data: UserUpdate, current_user: dict = Depends(get_current_user)):
    # Endpoint for patients to update their personal details.
    conn = get_db_connection()
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if update_data.username:
        updates.append("username = ?")
        params.append(update_data.username)
    if update_data.password:
        updates.append("password = ?")
        params.append(get_password_hash(update_data.password))
    if update_data.full_name:
        updates.append("full_name = ?")
        params.append(update_data.full_name)
    if update_data.address:
        updates.append("address = ?")
        params.append(update_data.address)
        
    if not updates:
        return {"status": "no_changes"}
        
    params.append(current_user['id'])
    query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
    
    try:
        cursor.execute(query, params)
        conn.commit()
        return {"status": "success", "message": "Profile updated successfully."}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already taken.")
    finally:
        conn.close()

@app.get("/chat/history")
async def fetch_history(current_user: dict = Depends(get_current_user)):
    # Retrieves the last 50 messages for the authenticated user to restore chat context.
    conn = get_db_connection()
    
    messages = conn.execute("""
        SELECT role, content, timestamp, source 
        FROM (
            SELECT id, role, content, timestamp, source 
            FROM messages 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 50
        )
        ORDER BY timestamp ASC 
    """, (current_user['id'],)).fetchall()
    
    conn.close()
    
    # Parse the source JSON back into a dictionary for the frontend
    formatted_messages = []
    for msg in messages:
        msg_dict = dict(msg)
        if msg_dict.get('source'):
            msg_dict['source'] = json.loads(msg_dict['source'])
        formatted_messages.append(msg_dict)
        
    return formatted_messages

@app.get("/appointments")
async def list_appointments(current_user: dict = Depends(get_current_user)):
    # Endpoint for the user dashboard. Returns only the logged-in user's appointments.
    conn = get_db_connection()
    appointments = conn.execute("""
        SELECT id, slot_time, booking_reference, appointment_type, status, patient_name, notes
        FROM appointments
        WHERE user_id = ?
        ORDER BY slot_time
    """, (current_user['id'],)).fetchall()
    conn.close()
    return [dict(appt) for appt in appointments]

@app.post("/appointments/refresh")
async def refresh_appointments():
    # Admin/Dev endpoint to reset the appointment database.
    try:
        generate_appointment_slots()
        return {"status": "success", "message": "Appointment slots have been refreshed for the next 5 business days."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to refresh appointments: {str(e)}"}
    


class CancelByIdRequest(BaseModel):
    appointment_id: int

@app.post("/appointments/cancel-by-id")
async def cancel_by_id(request: CancelByIdRequest, current_user: dict = Depends(get_current_user)):
    """Directly cancels a specific appointment by ID. Used by the cancellation picker widget."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        appt = cursor.execute("""
            SELECT id, patient_name, booking_reference, slot_time
            FROM appointments
            WHERE id = ? AND user_id = ? AND is_booked = 1
        """, (request.appointment_id, current_user['id'])).fetchone()

        if not appt:
            raise HTTPException(status_code=404, detail="Appointment not found or does not belong to you.")

        cursor.execute("""
            UPDATE appointments
            SET is_booked = 0, user_id = NULL, patient_name = NULL,
                booking_reference = NULL, booked_at = NULL, status = 'available'
            WHERE id = ?
        """, (appt['id'],))

        cursor.execute("""
            INSERT INTO booking_history (appointment_id, action, patient_name, booking_reference, timestamp)
            VALUES (?, 'cancelled', ?, ?, ?)
        """, (appt['id'], appt['patient_name'], appt['booking_reference'],
              datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        conn.commit()
        slot_dt = datetime.strptime(appt['slot_time'], '%Y-%m-%d %H:%M:%S')
        return {
            "status": "success",
            "message": f"Your appointment on {slot_dt.strftime('%A, %d %b at %I:%M %p').lstrip('0')} has been cancelled."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


class RescheduleRequest(BaseModel):
    old_appointment_id: int
    requested_day: str
    requested_time: str

@app.post("/appointments/reschedule")
async def reschedule_appointment(request: RescheduleRequest, current_user: dict = Depends(get_current_user)):
    # cancels an existing appointment and books a new one in a single transaction.
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        old_appt = cursor.execute("""
            SELECT id, patient_name, booking_reference, slot_time, notes
            FROM appointments
            WHERE id = ? AND user_id = ? AND is_booked = 1
        """, (request.old_appointment_id, current_user['id'])).fetchone()

        if not old_appt:
            raise HTTPException(status_code=404, detail="Original appointment not found or does not belong to you.")

        available = cursor.execute("""
            SELECT id, slot_time FROM appointments
            WHERE is_booked = 0 AND datetime(slot_time) > datetime('now')
            ORDER BY slot_time
        """).fetchall()

        requested_time_clean = request.requested_time.strip().upper()
        new_slot = None

        for slot in available:
            slot_dt = datetime.strptime(slot['slot_time'], '%Y-%m-%d %H:%M:%S')
            slot_day = slot_dt.strftime('%A')
            slot_time_12hr = slot_dt.strftime('%I:%M %p').lstrip('0').upper()
            slot_date_full = slot_dt.strftime('%A, %B %d, %Y')

            day_match = (request.requested_day.lower() in slot_day.lower() or
                         request.requested_day.lower() in slot_date_full.lower())
            time_match = requested_time_clean in slot_time_12hr

            if day_match and time_match:
                new_slot = slot
                break

        if not new_slot:
            raise HTTPException(
                status_code=409,
                detail="That slot is no longer available. Please choose another time."
            )

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_booking_ref = f"GP{new_slot['id']:04d}{datetime.now().strftime('%H%M')}"

        # Cancel old
        cursor.execute("""
            UPDATE appointments
            SET is_booked = 0, user_id = NULL, patient_name = NULL,
                booking_reference = NULL, booked_at = NULL, status = 'available'
            WHERE id = ?
        """, (old_appt['id'],))

        cursor.execute("""
            INSERT INTO booking_history (appointment_id, action, patient_name, booking_reference, timestamp, reason)
            VALUES (?, 'cancelled', ?, ?, ?, 'rescheduled')
        """, (old_appt['id'], old_appt['patient_name'], old_appt['booking_reference'], now))

        # Book new
        cursor.execute("""
            UPDATE appointments
            SET is_booked = 1, user_id = ?, patient_name = ?,
                booking_reference = ?, booked_at = ?, status = 'confirmed',
                notes = ?
            WHERE id = ?
        """, (current_user['id'], current_user['full_name'], new_booking_ref, now, old_appt['notes'], new_slot['id']))
        cursor.execute("""
            INSERT INTO booking_history (appointment_id, action, patient_name, booking_reference, timestamp)
            VALUES (?, 'booked', ?, ?, ?)
        """, (new_slot['id'], current_user['full_name'], new_booking_ref, now))

        conn.commit()
        new_slot_dt = datetime.strptime(new_slot['slot_time'], '%Y-%m-%d %H:%M:%S')
        return {
            "status": "success",
            "booking_reference": new_booking_ref,
            "message": f"Rescheduled to {new_slot_dt.strftime('%A, %d %b at %I:%M %p').lstrip('0')}",
        }

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Reschedule failed: {str(e)}")
    finally:
        conn.close()


class VerifyIdentityRequest(BaseModel):
    username: str
    dob: str

class ResetPasswordRequest(BaseModel):
    username: str
    dob: str
    new_password: str

@app.post("/auth/verify-identity")
async def verify_identity(request: VerifyIdentityRequest):
    """Verifies username and DOB match before allowing a password reset."""
    conn = get_db_connection()
    user = conn.execute(
        "SELECT id FROM users WHERE username = ? AND dob = ?",
        (request.username, request.dob)
    ).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=404, detail="No account found with those details.")
    return {"status": "verified"}

@app.post("/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Resets the password after identity has been verified via DOB."""
    conn = get_db_connection()
    user = conn.execute(
        "SELECT id FROM users WHERE username = ? AND dob = ?",
        (request.username, request.dob)
    ).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="Identity verification failed.")
    conn.execute(
        "UPDATE users SET password = ? WHERE id = ?",
        (get_password_hash(request.new_password), user['id'])
    )
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Password reset successfully."}


@app.post("/chat")
async def handle_chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    user_message = request.message
    history = request.history
    
    # Variable to capture the official NHS source if the tool is used
    medical_source = None
    # Variable to capture structured slots for the frontend interactive picker
    available_slots = None
    # Variables for cancellation and reschedule widgets
    cancellation_slots = None
    reschedule_data = None
    # Variable to capture verified general advisory when RAG returns low_confidence
    general_source = None
    
    # save the user's incoming message
    conn = get_db_connection()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn.execute(
        "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (current_user['id'], 'user', user_message, current_time)
    )
    conn.commit()

    
    
    system_prompt = {
        "role": "system",
        "content": f"""You are a helpful GP surgery assistant.
You are speaking with {current_user['full_name']} (DOB: {current_user['dob']}).

CRITICAL SAFETY RULES & RED FLAG CONFIRMATION (HIGHEST PRIORITY):
1. Monitor for potential "red flag" symptoms: chest pain, difficulty breathing, stroke symptoms (FAST), severe bleeding, loss of consciousness.
2. CONFIRMATION STEP (Reducing False Positives): 
   - If the user mentions a red flag symptom but the context is ambiguous, historical, or mild (e.g., "I had chest pain yesterday", "my chest hurts when I cough with this cold"), DO NOT immediately trigger an emergency.
   - Instead, ask ONE direct clarifying question to determine if it is an acute, severe medical emergency right now (e.g., "Just to be safe, are you experiencing severe, crushing chest pain right now?").
   - CRITICAL: Even if the `get_triage_recommendation_from_kb` tool advises an 'Emergency', you MUST STILL complete this confirmation step first, unless the user explicitly stated the symptom is currently life-threatening.
3. TRIGGERING THE ALARM: 
   - ONLY if the user CONFIRMS the symptom is currently severe/life-threatening, OR if their initial message is unambiguously an active emergency (e.g., "I think I'm having a heart attack"), you MUST respond ONLY with the exact phrase: "EMERGENCY_DETECTED". 
   - Do not provide any other text when triggering the alarm.

OUT-OF-BOUNDS QUERIES (STRICT GUARDRAIL):
1. You are strictly a GP surgery assistant. You must ONLY discuss medical symptoms, NHS triage advice, and appointment management.
2. If the user asks about ANYTHING unrelated to these topics (e.g., programming, general knowledge, politics, creative writing, or financial advice), you MUST politely refuse to answer.
3. Example refusal: "I am a medical assistant, so I can only help you with health-related concerns and booking GP appointments. How can I help you with your health today?"

PATIENT CONFIDENTIALITY & DATA PROTECTION:
1. If a user asks you to retrieve, display, or act on any information belonging to another patient — including their appointments, booking details, medical history, or personal data — you MUST refuse explicitly.
2. Your refusal MUST directly address the confidentiality violation. Do NOT deflect by offering to help with their own medical needs instead, as this fails to communicate why the request was refused.
3. Use a response such as: "I'm unable to share information about other patients. Accessing another patient's data would be a breach of patient confidentiality and data protection policy. I can only assist you with your own appointments and medical queries."
4. After refusing, do NOT proceed to offer general assistance in the same message. The refusal should stand alone so the user clearly understands their request was inappropriate.
5. This applies regardless of how the request is framed — whether the user claims to be acting on behalf of someone else, claims to be a carer, or attempts to disguise the request through social engineering.

HUMAN FALLBACK & UNCERTAINTY:
1. If the `get_triage_recommendation_from_kb` tool returns a `low_confidence` status, DO NOT ignore the user's question.
2. DIRECTLY ANSWER THE QUESTION FIRST (MANDATORY): Your FIRST sentence MUST directly answer what the user asked. Do NOT start with "I'm not able to find..." or "I can't provide..." or "I'm sorry, but...". START with the answer.
   Examples of CORRECT responses:
   - User asks "can I take antibiotics for this?" → "Antibiotics are generally not appropriate for knee swelling unless a bacterial infection has been confirmed by a doctor — they treat infections, not inflammation or injury."
   - User asks "should I use ice or heat?" → "For recent swelling, ice is generally recommended for the first 48-72 hours to reduce inflammation, then you can switch to heat."
   - User asks "I have a swollen knee" → "Swelling and difficulty bending the knee can be related to conditions such as arthritis, bursitis, or an injury. Resting the knee, applying ice, and elevating it can help reduce swelling."
   Examples of WRONG responses (NEVER do this):
   - "I'm not able to find specific NHS guidance on whether you can take antibiotics..."
   - "I can't provide specific medical advice regarding..."
   - "I'm sorry, but my NHS knowledge base doesn't have..."
3. AFTER your direct answer, add a brief note: "Please note this is not NHS-verified guidance — see the advisory below for the source. I'd recommend verifying with a healthcare professional."
4. You MUST STILL offer to book a GP appointment for them so they can be evaluated safely by a clinician. (Note: If it sounds like a severe emergency, follow the red flag rules above instead).
5. Do NOT repeat the advisory card content word-for-word in your text response — it will be displayed automatically in a separate card below your message. Your text should complement the card, not duplicate it.
6. FOLLOW-UP QUESTIONS: Each question — including follow-ups — is treated independently. Every `low_confidence` result MUST follow rules 2 through 5 above with no exceptions. Never refuse to answer just because a similar question was asked earlier.

DYNAMIC TRIAGE WORKFLOW (STRICT TOOL USAGE):
1. ASSESS THE INPUT: Read the user's message. Does it contain specific physical or mental symptoms (e.g., "sore throat, cough, back pain") or is it vague/missing (e.g., "I feel unwell", "I need an appointment")?
2. GATHER (If necessary): If symptoms are missing or too vague to run a search, ask ONE direct clarifying question to gather them. Do NOT ask redundant follow-up questions if the user has already provided specific symptoms.
3. SEARCH (When ready): As soon as you have specific symptoms, you MUST IMMEDIATELY call `get_triage_recommendation_from_kb`. Do not delay or ask for further confirmation.
4. NO GUESSING WHEN NHS DATA EXISTS: When the tool returns a confident NHS match (`found` status), you MUST ONLY provide advice based on the NHS data returned by the tool. DO NOT supplement with your own knowledge.
   EXCEPTION — LOW CONFIDENCE: When the tool returns `low_confidence`, you ARE PERMITTED AND REQUIRED to directly answer the user's question using general medical knowledge. Follow the HUMAN FALLBACK & UNCERTAINTY rules above. A verified advisory card will appear below your message to provide the source. Your text response MUST directly address what the user asked — start with the answer, not a disclaimer.
5. FOLLOW-UP QUESTIONS: This rule applies equally to ALL follow-up medical questions, not just the initial symptom report. If the user asks ANY question that requires medical knowledge — including questions about medications (e.g. "can I take antibiotics"), suitability for activities (e.g. "can my child go to school"), home management, or anything else clinical in nature — you MUST call `get_triage_recommendation_from_kb` with the follow-up question as the symptom description. If the tool returns `low_confidence`, follow the HUMAN FALLBACK & UNCERTAINTY rules: directly answer the question, reference the advisory card below your message, and offer a GP appointment.
6. CRITICAL — NO SKIPPING BASED ON HISTORY: You MUST call the required tool for EVERY relevant request in the CURRENT turn, without exception. This applies to ALL tools:
   - `get_triage_recommendation_from_kb` must be called every time symptoms are mentioned, even if identical symptoms appear in history.
   - `get_available_appointments` must be called every time the user wants to book, even if slots were fetched earlier in the conversation.
   - `get_user_bookings` must be called every time the user wants to cancel, even if their bookings were fetched earlier.
   - `get_reschedule_data` must be called every time the user wants to reschedule, even if this data was fetched earlier.
   Prior conversation context NEVER substitutes for a fresh tool call. The interactive widgets are only shown when the tool is called in the current turn — if you skip the tool call, the user will see no widget and the feature will be broken.
7. RESOLUTION: 
   - If the tool returns `low_confidence`, follow the HUMAN FALLBACK & UNCERTAINTY rules: directly answer the question, reference the advisory card, and offer a GP appointment.
   - If it advises 'GP' or 'Pharmacist', present the advice. 
   - If it advises an 'Emergency', you MUST ask a clarifying question to confirm severity before triggering the alarm.

BOOKING WORKFLOW (STRICTLY AFTER TRIAGE):
1. PREREQUISITE: You MUST NOT offer or book an appointment unless you have gathered symptoms AND run the `get_triage_recommendation_from_kb` tool. You are permitted to book an appointment if the tool advises 'GP' OR if it returns 'low_confidence'.
2. If a user asks for an appointment but hasn't given symptoms, politely explain that you need to do a quick symptom check first.
3. Call `get_available_appointments`.
4. After calling the tool, respond with ONLY a brief single sentence such as "Here are the available slots — please select a time below." Do NOT list the individual dates or times in your text response, as an interactive picker will be shown to the user automatically.
5. Call `book_appointment_by_datetime`. 
   IMPORTANT: You already know their name is {current_user['full_name']}, so DO NOT ask for it.
   NOTES: Always populate the `notes` field with a concise clinical summary, e.g. "Likely common cold. Triage: Pharmacist. Symptoms: sore throat, runny nose for 3 days." If no symptoms were discussed (routine checkup), set notes to "Routine checkup."

CANCELLATION:
1. If a user asks to cancel an appointment, immediately call `get_user_bookings`. Do NOT ask for the day, time, or booking reference first.
2. A cancellation picker will be shown to the user automatically. Respond with only: "Here are your current appointments — please select the one you'd like to cancel."

RESCHEDULING:
1. If a user asks to reschedule, immediately call `get_reschedule_data`. Do NOT treat this as a two-step process or ask for any details first.
2. A reschedule widget will be shown to the user automatically. Respond with only: "Please select the appointment you'd like to move and your preferred new time below."
Be warm, professional, and concise."""
    }
    
    messages = [system_prompt] + history + [{"role": "user", "content": user_message}]

    def book_appt_wrapper(requested_day, requested_time, patient_name=None, notes=None):
        return book_appointment_by_datetime(current_user['id'], current_user['full_name'], requested_day, requested_time, notes)
        
    def cancel_appt_wrapper(requested_day, requested_time):
        return cancel_appointment(current_user['id'], requested_day, requested_time)

    def get_user_bookings_wrapper():
        return get_user_bookings(current_user['id'])

    def get_reschedule_data_wrapper():
        return get_reschedule_data(current_user['id'])

    # Define available tools for the OpenAI model
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
                        "notes": {"type": "string", "description": "A brief clinical summary for the appointment, including the triage recommendation and the likely condition based on symptoms discussed. Leave empty if booking is for a routine checkup with no symptoms discussed."},
                    },
                    "required": ["requested_day", "requested_time"],
                },
            },
        },
        
        {
            "type": "function",
            "function": {
                "name": "cancel_appointment",
                "description": "Cancels a booking based on the natural language day and time. Do NOT ask for a booking reference.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requested_day": {"type": "string", "description": "Day of the appointment to cancel (e.g. Monday)."},
                        "requested_time": {"type": "string", "description": "Time of the appointment to cancel (e.g. 9:00 AM)."}
                    },
                    "required": ["requested_day", "requested_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_bookings",
                "description": "Retrieves the user's current booked appointments. Call this immediately when the user wants to cancel. Do NOT ask for day or time first — a picker will be shown automatically.",
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_reschedule_data",
                "description": "Retrieves the user's current bookings AND available slots in one call. Call this immediately when the user wants to reschedule. Do NOT manage this as a two-step flow — a widget will be shown automatically.",
            }
        },
    ]

    available_functions = {
        "get_triage_recommendation_from_kb": get_triage_recommendation_from_kb,
        "get_available_appointments": get_available_appointments,
        "book_appointment_by_datetime": book_appt_wrapper,
        "cancel_appointment": cancel_appt_wrapper,
        "get_user_bookings": get_user_bookings_wrapper,
        "get_reschedule_data": get_reschedule_data_wrapper,
    }

    # Detect clear intent and force the correct tool on the first iteration.
    # This bypasses LLM judgment entirely for widget-triggering scenarios,
    # preventing the model from skipping tool calls based on conversation history.
    msg_lower = user_message.lower()
    
    forced_tool_first_pass = "auto"
    if any(w in msg_lower for w in ['reschedule', 'move my appointment', 'change my appointment', 'swap my appointment']):
        forced_tool_first_pass = {"type": "function", "function": {"name": "get_reschedule_data"}}
    elif any(w in msg_lower for w in ['cancel', 'remove my appointment', 'delete my appointment']):
        forced_tool_first_pass = {"type": "function", "function": {"name": "get_user_bookings"}}
    elif any(w in msg_lower for w in ['book an appointment', 'make an appointment', 'schedule an appointment', 'get an appointment', 'see a gp', 'see the gp']):
        forced_tool_first_pass = {"type": "function", "function": {"name": "get_available_appointments"}}

    try:
        # Tool-calling loop with safety limit to prevent infinite recursion
        iteration = 0
        while iteration < 5:
            iteration += 1
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice=forced_tool_first_pass if iteration == 1 else "auto",
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
                        if function_name == "get_triage_recommendation_from_kb":
                            try:
                                symptom = json.loads(tool_call.function.arguments).get("symptom_description", "")
                                function_response = function_to_call(symptom_description=symptom)
                            except Exception:
                                function_response = json.dumps({"status": "error", "message": "Could not parse triage arguments."})
                        else:
                            function_response = json.dumps({"status": "error", "message": f"Argument error: {str(e)}"})
                    except Exception as e:
                        function_response = json.dumps({"status": "error", "message": f"Error: {str(e)}"})

                    # Capture bookings for the cancellation picker
                    if function_name == "get_user_bookings":
                        try:
                            cancel_data = json.loads(function_response)
                            if cancel_data.get("status") == "found":
                                cancellation_slots = cancel_data.get("bookings")
                        except (json.JSONDecodeError, KeyError):
                            pass

                    # Capture data for the reschedule widget
                    if function_name == "get_reschedule_data":
                        try:
                            r_data = json.loads(function_response)
                            if r_data.get("status") == "ready":
                                reschedule_data = {
                                    "current_bookings": r_data.get("current_bookings"),
                                    "available_slots": r_data.get("available_slots")
                                }
                        except (json.JSONDecodeError, KeyError):
                            pass

                    # Capture structured slots for the interactive frontend picker
                    if function_name == "get_available_appointments":
                        try:
                            appt_data = json.loads(function_response)
                            if appt_data.get("status") == "available":
                                available_slots = appt_data.get("appointments_grouped")
                        except (json.JSONDecodeError, KeyError):
                            pass

                    # Capture source data here
                    # Moved outside try/except so it always runs on the final function_response
                    if function_name == "get_triage_recommendation_from_kb":
                        try:
                            triage_data = json.loads(function_response)
                            if triage_data.get("status") == "found" and triage_data.get("context"):
                                best_match = triage_data["context"][0]
                                medical_source = {
                                    "condition": best_match.get("condition_name", "Medical Condition"),
                                    "text": best_match.get("source_explanation", ""),
                                    "url": best_match.get("source_url", "")
                                }
                            elif triage_data.get("status") == "low_confidence":
                                # Trigger the multi-agent advisory pipeline with the user's actual question
                                advisory_input = user_message
                                # Append symptom context from tool arguments if different from the user message
                                try:
                                    tool_symptom = json.loads(tool_call.function.arguments).get("symptom_description", "")
                                    if tool_symptom and tool_symptom.lower().strip() != user_message.lower().strip():
                                        advisory_input = f"Question: {user_message} (Context: patient has {tool_symptom})"
                                except Exception:
                                    pass
                                if advisory_input:
                                    advisory_result = get_general_medical_advisory(advisory_input)
                                    if advisory_result:
                                        general_source = {
                                            "advice": advisory_result["advice"],
                                            "source_name": advisory_result["source_name"],
                                            "source_url": advisory_result["source_url"],
                                            "condition_topic": advisory_result["condition_topic"]
                                        }
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass

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
    
    # Convert the dictionary to a JSON string for SQLite storage
    source_json = json.dumps(medical_source) if medical_source else None
    general_source_json = json.dumps(general_source) if general_source else None
    
    conn.execute(
        "INSERT INTO messages (user_id, role, content, timestamp, source) VALUES (?, ?, ?, ?, ?)",
        (current_user['id'], 'assistant', agent_response, response_time, source_json)
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
        "status": "normal",
        "source": medical_source,
        "general_source": general_source,
        "available_slots": available_slots,
        "cancellation_slots": cancellation_slots,
        "reschedule_data": reschedule_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)