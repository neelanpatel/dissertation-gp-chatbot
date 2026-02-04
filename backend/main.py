import os
import sqlite3
import json
import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment configuration
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_NAME = 'gp_database.db'

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
    
    # Appointments table: Stores the core slot data.
    # We use a unique constraint on slot_time to prevent double-booking at the schema level.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY,
        slot_time TEXT NOT NULL UNIQUE,
        is_booked BOOLEAN NOT NULL DEFAULT 0,
        patient_name TEXT,
        booking_reference TEXT UNIQUE,
        booked_at TEXT,
        appointment_type TEXT DEFAULT 'standard',
        notes TEXT,
        status TEXT DEFAULT 'available'
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
    
    conn.commit()
    conn.close()

def generate_appointment_slots():
    """
    Populates the database with empty slots for the next 5 business days.
    Logic: 9am-12pm and 2pm-5pm, every 20 minutes.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Resetting the schedule for the demo/dev environment
    cursor.execute("DELETE FROM appointments")
    
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
            grouped_for_display[date_key] = {
                'date': date_key,
                'day_name': slot_datetime.strftime('%A'),
                'times': []
            }
        
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
        "appointments_grouped": grouped_for_display,
        "total_available": len(appointments),
        "instruction": "Please let me know which time works best for you by specifying the day and time (e.g., 'Monday at 9:20 AM')."
    })

def book_appointment_by_datetime(patient_name: str, requested_day: str, requested_time: str):
    """
    Attempts to match a fuzzy natural language date request to a specific database slot.
    """
    import re
    from datetime import datetime
    
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
            day_match = (requested_day.lower() in slot_day.lower() or 
                        requested_day.lower() in slot_date_full.lower())
            time_match = requested_time_clean in slot_time_12hr
            
            if day_match and time_match:
                # Generate a pseudo-random booking reference
                booking_ref = f"GP{slot['id']:04d}{datetime.now().strftime('%H%M')}"
                booked_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Update the main record
                cursor.execute("""
                    UPDATE appointments 
                    SET is_booked = 1, 
                        patient_name = ?, 
                        booking_reference = ?,
                        booked_at = ?,
                        status = 'confirmed'
                    WHERE id = ?
                """, (patient_name, booking_ref, booked_at, slot['id']))
                
                # Write to audit log
                cursor.execute("""
                    INSERT INTO booking_history (appointment_id, action, patient_name, booking_reference, timestamp)
                    VALUES (?, 'booked', ?, ?, ?)
                """, (slot['id'], patient_name, booking_ref, booked_at))
                
                conn.commit()
                conn.close()
                
                return json.dumps({
                    "status": "success",
                    "confirmation": f"""тЬЕ Appointment Confirmed!

ЁЯУЛ Booking Reference: {booking_ref}
ЁЯУЕ Date & Time: {slot_datetime.strftime('%A, %B %d, %Y at %I:%M %p').replace(' 0', ' ')}
ЁЯСд Patient: {patient_name}

Your appointment has been successfully booked. Please arrive 5 minutes early."""
                })
        
        conn.close()
        return json.dumps({
            "status": "error", 
            "message": f"I couldn't find an available appointment for {requested_day} at {requested_time}. Please choose from the available times I showed you."
        })
        
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Booking error: {str(e)}"})

def cancel_appointment(booking_reference: str = None, patient_name: str = None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Support cancellation by either Ref ID (preferred) or Name (fallback)
        if booking_reference:
            cursor.execute("""
                SELECT id, slot_time, patient_name, booking_reference 
                FROM appointments 
                WHERE booking_reference = ? AND is_booked = 1
            """, (booking_reference,))
        elif patient_name:
            cursor.execute("""
                SELECT id, slot_time, patient_name, booking_reference 
                FROM appointments 
                WHERE patient_name = ? AND is_booked = 1
                ORDER BY slot_time DESC
                LIMIT 1
            """, (patient_name,))
        else:
            conn.close()
            return json.dumps({"status": "error", "message": "Please provide either a booking reference or patient name."})
        
        appointment = cursor.fetchone()
        
        if not appointment:
            conn.close()
            return json.dumps({"status": "error", "message": "No booking found with that information."})
        
        # Reset the slot to available
        cursor.execute("""
            UPDATE appointments 
            SET is_booked = 0, 
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
        
        slot_datetime = datetime.strptime(appointment['slot_time'], '%Y-%m-%d %H:%M:%S')
        
        return json.dumps({
            "status": "success",
            "message": f"""Appointment cancelled successfully.

Reference: {appointment['booking_reference']}
Date: {slot_datetime.strftime('%A, %B %d, %Y at %I:%M %p').replace(' 0', ' ')}
Patient: {appointment['patient_name']}

The slot is now available for rebooking."""
        })
        
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Cancellation error: {str(e)}"})

def get_triage_recommendation_from_kb(symptom_description: str):
    """
    RAG Implementation:
    1. Vectorises the symptom description.
    2. Searches the FAISS index for relevant NHS content chunks.
    3. Returns the top 3 matches for the LLM to synthesise.
    """
    if not faiss_index or not nhs_data:
        return json.dumps({"status": "error", "message": "Knowledge base is not loaded."})

    try:
        query_embedding = client.embeddings.create(
            input=[symptom_description],
            model="text-embedding-3-small"
        ).data[0].embedding
        query_vector = np.array([query_embedding]).astype("float32")

        k = 3
        distances, indices = faiss_index.search(query_vector, k)
        
        relevant_chunks = [nhs_data[i] for i in indices[0]]
        
        return json.dumps({"status": "found", "context": relevant_chunks})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Knowledge base search error: {str(e)}"})

# FastAPI Configuration
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list

@app.get("/appointments")
async def list_all_appointments():
    """Endpoint for the admin/debug sidebar."""
    conn = get_db_connection()
    appointments = conn.execute("""
        SELECT id, slot_time, is_booked, patient_name, booking_reference, appointment_type, status
        FROM appointments
        ORDER BY slot_time
    """).fetchall()
    conn.close()
    return [dict(appt) for appt in appointments]

@app.post("/appointments/refresh")
async def refresh_appointments():
    try:
        generate_appointment_slots()
        return {"status": "success", "message": "Appointment slots have been refreshed for the next 5 business days."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to refresh appointments: {str(e)}"}

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    user_message = request.message
    history = request.history
    
    # System prompt defines the persona and STRICT safety guardrails.
    # Note: Moving this to a config file would be a future improvement.
    system_prompt = {
        "role": "system",
        "content": """You are a helpful, professional, and empathetic GP surgery assistant with a friendly personality.

CRITICAL SAFETY RULES - HIGHEST PRIORITY:
1. ALWAYS check for "red flag" emergency symptoms first: chest pain, difficulty breathing, severe bleeding, stroke symptoms (FAST), sudden confusion, seizures, loss of consciousness, severe allergic reactions.
2. If ANY red flag is detected, IMMEDIATELY respond: "ЁЯЪи Based on your symptoms, you need urgent medical attention. Please call 999 immediately or go to your nearest A&E. This is a medical emergency that I cannot help with."
3. DO NOT proceed with triage or booking for emergency symptoms.

TRIAGE WORKFLOW (only for non-emergency symptoms):
1. When user requests an appointment or mentions feeling unwell:
   - First response: Ask for symptoms naturally: "I'd be happy to help you book an appointment. Could you briefly tell me what symptoms you're experiencing? This helps ensure you get the right level of care."
   
2. After receiving symptoms:
   - ALWAYS call `get_triage_recommendation_from_kb` tool
   - The tool returns JSON with 'status' and 'context' (array of relevant medical info)
   
3. Analyze the triage results:
   - If context shows 'Pharmacist' recommendation: Inform user kindly, cite the condition_name and source_url from context
   - If context shows 'Emergency': Follow CRITICAL SAFETY RULES above
   - If context shows 'GP' or is unclear: Proceed to call `get_available_appointments`
   
4. If user insists on GP appointment after pharmacist recommendation: Be helpful and call `get_available_appointments`

BOOKING WORKFLOW:
1. When showing appointments:
   - Present them in a clean, organized format by date
   - Use the formatted_display from the tool response
   - Ask user to specify day and time (e.g., "Monday at 9:00 AM")

2. When user chooses a time:
   - Check conversation history for their name
   - If no name found, ask: "Great choice! Could I have your full name for the booking?"
   - Once you have both name and time, call `book_appointment_by_datetime`

3. Confirmation:
   - Display the formatted confirmation message from the tool
   - Be warm and professional

CANCELLATION WORKFLOW:
1. When user wants to cancel an appointment:
   - First call `get_available_appointments` to show all appointments (including booked ones)
   - Ask user to provide the appointment ID number
   - Once they provide the ID, call `cancel_appointment` with that appointment_id
   - DO NOT ask for name or booking reference - appointment ID is sufficient
2. Confirmation:
   - Display the confirmation message warmly
   - Let them know the slot is now available again

CONVERSATION STYLE:
- Be warm, professional, and efficient
- Use natural language, not robotic
- Show empathy for health concerns
- Keep responses concise but friendly
- Use emojis sparingly for clarity (тЬЕ for confirmations, ЁЯУЕ for dates, etc.)

ERROR HANDLING:
- If a tool returns an error, rephrase it naturally and helpfully
- Never show raw JSON or technical errors to the user
- Always offer an alternative or next step"""
    }
    
    messages = [system_prompt] + history + [{"role": "user", "content": user_message}]
    
    # Define available tools (Functions) for the OpenAI model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_triage_recommendation_from_kb",
                "description": "Searches the NHS knowledge base for triage advice based on symptoms. Returns medical guidance including whether to see a pharmacist, GP, or seek emergency care.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symptom_description": {
                            "type": "string", 
                            "description": "The patient's description of their symptoms, e.g., 'sore throat and fever', 'persistent headache'."
                        },
                    },
                    "required": ["symptom_description"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_available_appointments",
                "description": "Retrieves a formatted list of available appointment slots grouped by date. Returns up to 5 days of availability."
            }
        },
        {
            "type": "function",
            "function": {
                "name": "book_appointment_by_datetime",
                "description": "Books a GP appointment for a patient using natural language date and time (e.g., 'Monday at 9:00 AM'). Returns a formatted confirmation with booking reference.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_name": {
                            "type": "string", 
                            "description": "The patient's full name."
                        },
                        "requested_day": {
                            "type": "string", 
                            "description": "The day the patient wants (e.g., 'Monday', 'Tuesday')."
                        },
                        "requested_time": {
                            "type": "string", 
                            "description": "The time the patient wants in 12-hour format (e.g., '9:00 AM', '2:20 PM')."
                        },
                    },
                    "required": ["patient_name", "requested_day", "requested_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_appointment",
                "description": "Cancels a booked appointment using either a booking reference or patient name. Returns confirmation of cancellation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "booking_reference": {
                            "type": "string", 
                            "description": "The booking reference code (e.g., 'GP00011430')."
                        },
                        "patient_name": {
                            "type": "string", 
                            "description": "The patient's full name (used if booking reference not available)."
                        },
                    },
                },
            },
        },
    ]
    
    available_functions = {
        "get_triage_recommendation_from_kb": get_triage_recommendation_from_kb,
        "get_available_appointments": get_available_appointments,
        "book_appointment_by_datetime": book_appointment_by_datetime,
        "cancel_appointment": cancel_appointment,
    }

    try:
        # Tool-calling loop with safety limit to prevent infinite recursion
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
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
                    function_response = json.dumps({"status": "error", "message": f"Unknown function: {function_name}"})
                else:
                    if tool_call.function.arguments:
                        function_args = json.loads(tool_call.function.arguments)
                        function_response = function_to_call(**function_args)
                    else:
                        function_response = function_to_call()
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
        
        if iteration >= max_iterations:
            agent_response = "I apologise, but I'm having trouble processing your request. Could you please try again or rephrase your question?"
                
    except Exception as e:
        print(f"Error handling chat request: {e}")
        agent_response = "I'm sorry, I encountered an error. Please try again or contact the surgery directly if this persists."

    return {"response": agent_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)