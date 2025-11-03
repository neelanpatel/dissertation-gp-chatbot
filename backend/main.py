# backend/main.py
import os
import sqlite3
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATABASE_NAME = 'gp_database.db'

# --- Database Helper Functions ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Initializes the database, creating all necessary tables and mock data."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create appointments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY,
        slot_time TEXT NOT NULL UNIQUE,
        is_booked BOOLEAN NOT NULL DEFAULT 0,
        patient_name TEXT
    )
    ''')
    
    # Create table for triage rules
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS triage_rules (
        id INTEGER PRIMARY KEY,
        condition_keyword TEXT NOT NULL UNIQUE,
        recommended_service TEXT NOT NULL,
        explanation TEXT
    )
    ''')

    # Add mock appointment data if table is empty
    cursor.execute("SELECT COUNT(*) FROM appointments")
    if cursor.fetchone()[0] == 0:
        mock_slots = [
            ('2025-10-21 10:00:00',),
            ('2025-10-21 11:00:00',),
            ('2025-10-22 14:00:00',),
        ]
        cursor.executemany("INSERT INTO appointments (slot_time) VALUES (?)", mock_slots)

    # Add mock triage data if table is empty
    cursor.execute("SELECT COUNT(*) FROM triage_rules")
    if cursor.fetchone()[0] == 0:
        mock_rules = [
            ('hay fever', 'Pharmacist', 'Pharmacists can provide over-the-counter antihistamines and advice for hay fever.'),
            ('cold', 'Pharmacist', 'A pharmacist can recommend remedies for a common cold, like decongestants.'),
            ('persistent headache', 'GP', 'A persistent headache should be discussed with a GP to rule out any serious issues.'),
        ]
        cursor.executemany(
            "INSERT INTO triage_rules (condition_keyword, recommended_service, explanation) VALUES (?, ?, ?)",
            mock_rules
        )
    
    conn.commit()
    conn.close()

# Run DB initialization on startup
initialize_database()

# --- Tool Functions for the LLM ---
def get_available_appointments():
    """Gets a list of all available appointment slots from the database."""
    conn = get_db_connection()
    appointments = conn.execute("SELECT id, slot_time FROM appointments WHERE is_booked = 0").fetchall()
    conn.close()
    if not appointments:
        return "There are no available appointments."
    return json.dumps([dict(ix) for ix in appointments])

def book_appointment(appointment_id: int, patient_name: str):
    """Books an appointment for a given patient by its ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT is_booked FROM appointments WHERE id = ?", (appointment_id,))
        slot = cursor.fetchone()
        if slot is None: return f"Error: Appointment ID {appointment_id} not found."
        if slot['is_booked']: return f"Error: Appointment ID {appointment_id} is already booked."
        
        cursor.execute("UPDATE appointments SET is_booked = 1, patient_name = ? WHERE id = ?", (patient_name, appointment_id))
        conn.commit()
        conn.close()
        return json.dumps({"status": "success", "message": f"Appointment ID {appointment_id} has been successfully booked for {patient_name}."})
    except Exception as e:
        return f"Database Error: {e}"

def cancel_appointment(appointment_id: int):
    """Cancels a previously booked appointment by its ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT is_booked, patient_name FROM appointments WHERE id = ?", (appointment_id,))
        slot = cursor.fetchone()
        if slot is None: return f"Error: Appointment ID {appointment_id} not found."
        if not slot['is_booked']: return f"Error: Appointment ID {appointment_id} is not currently booked."
        
        patient_name = slot['patient_name']
        cursor.execute("UPDATE appointments SET is_booked = 0, patient_name = NULL WHERE id = ?", (appointment_id,))
        conn.commit()
        conn.close()
        return json.dumps({"status": "success", "message": f"Appointment ID {appointment_id} for {patient_name} has been successfully cancelled."})
    except Exception as e:
        return f"Database Error: {e}"

def get_triage_recommendation(symptom: str):
    """Looks up a symptom to find the recommended service (e.g., GP or Pharmacist) and an explanation."""
    try:
        conn = get_db_connection()
        query_symptom = f"%{symptom.lower()}%"
        rule = conn.execute(
            "SELECT recommended_service, explanation FROM triage_rules WHERE ? LIKE '%' || condition_keyword || '%'",
            (symptom.lower(),)
        ).fetchone()
        conn.close()
        
        if rule:
            return json.dumps({"status": "found", "service": rule['recommended_service'], "explanation": rule['explanation']})
        else:
            return json.dumps({"status": "not_found", "message": "No specific rule for this symptom. Proceed with GP booking."})
    except Exception as e:
        return f"Database Error: {e}"

# --- FastAPI App Setup ---
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

# --- API Endpoints ---
@app.get("/appointments")
async def list_all_appointments():
    conn = get_db_connection()
    appointments = conn.execute("SELECT id, slot_time, is_booked, patient_name FROM appointments").fetchall()
    conn.close()
    return [dict(ix) for ix in appointments]

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    user_message = request.message
    history = request.history
    
    # ### UPDATED SYSTEM PROMPT ###
    system_prompt = {
        "role": "system",
        "content": """You are a helpful and empathetic GP surgery assistant. Your purpose is to triage patients and book or cancel appointments.
        You have four tools: 'get_triage_recommendation', 'get_available_appointments', 'book_appointment', and 'cancel_appointment'.

        CRITICAL TRIAGE WORKFLOW:
        1. When a user describes a medical symptom (e.g., "I have a cough", "my head hurts"), your *first* action MUST be to use the `get_triage_recommendation` tool.
        2. If the tool returns 'Pharmacist', you MUST inform the user of this recommendation and provide the explanation. DO NOT offer to book a GP appointment.
        3. If the tool returns 'GP' or 'not_found', you should then offer to book a GP appointment by using the `get_available_appointments` tool.

        BOOKING WORKFLOW:
        1. When you list available appointments, you MUST include the `id` for each slot and ask the user to provide the `id`.
        2. The 'book_appointment' tool requires both an 'appointment_id' and a 'patient_name'.
        3. When a user asks to book an appointment, you MUST *first check the conversation history* to see if they have already provided their name (e.g., "Neelan Patel").
        4. If the name is already in the history, USE IT. Do not ask for it again.
        5. If the name is not in the history and the user is booking, you MUST ask for the patient's full name before calling the tool.
        
        CRITICAL ERROR HANDLING RULES:
        1. If a tool call returns an error (e.g., "Error: Appointment ID 1 is already booked" or "Error: Appointment ID 99 not found"), you MUST NOT just read the error message.
        2. You MUST re-phrase the error in a helpful, conversational, and apologetic way.
        3. You MUST suggest a solution or a next step.
        4. Example 1: If the error is 'already booked', say: "I'm so sorry, it looks like that appointment slot was just taken. Can I check for other available slots for you?"
        5. Example 2: If the error is 'not found', say: "I'm sorry, I couldn't find an appointment with that ID. Could you please double-check the ID from the list I gave you?"

        Always be clear and confirm when an action (booking or cancelling) is complete."""
    }
    
    messages = [system_prompt] + history + [{"role": "user", "content": user_message}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_triage_recommendation",
                "description": "Checks a user's symptom against a knowledge base to find the correct service (GP or Pharmacist).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symptom": {"type": "string", "description": "The medical symptom described by the user, e.g., 'cough' or 'hay fever'."},
                    },
                    "required": ["symptom"],
                },
            },
        },
        {"type": "function", "function": {"name": "get_available_appointments", "description": "Get a list of available appointment slots."}},
        {
            "type": "function",
            "function": {
                "name": "book_appointment",
                "description": "Book a specific appointment slot for a patient using its unique ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_id": {"type": "integer", "description": "The unique ID of the appointment slot to book."},
                        "patient_name": {"type": "string", "description": "The full name of the patient."},
                    },
                    "required": ["appointment_id", "patient_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_appointment",
                "description": "Cancel a previously booked appointment using its unique ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_id": {"type": "integer", "description": "The unique ID of the appointment to cancel."},
                    },
                    "required": ["appointment_id"],
                },
            },
        },
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            available_functions = {
                "get_triage_recommendation": get_triage_recommendation,
                "get_available_appointments": get_available_appointments,
                "book_appointment": book_appointment,
                "cancel_appointment": cancel_appointment,
            }
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "get_triage_recommendation" and "symptom" not in function_args:
                    function_response = json.dumps({"status": "error", "message": "Missing symptom argument."})
                else:
                    function_response = function_to_call(**function_args)
                
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            
            second_response = client.chat.completions.create(model="gpt-4o", messages=messages)
            agent_response = second_response.choices[0].message.content
        else:
            agent_response = response_message.content
                
    except Exception as e:
        print(f"An error occurred: {e}")
        agent_response = "Sorry, I encountered an error. Please try again."

    return {"response": agent_response}