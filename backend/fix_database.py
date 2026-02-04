import os
import sqlite3
import shutil
from datetime import datetime, timedelta

DATABASE_NAME = 'gp_database.db'

# Backup existing database if it exists
if os.path.exists(DATABASE_NAME):
    backup_name = f'gp_database_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
    shutil.copy2(DATABASE_NAME, backup_name)
    print(f"✅ Created backup: {backup_name}")
    os.remove(DATABASE_NAME)
    print(f"✅ Removed old database")

# Create new database with correct schema
conn = sqlite3.connect(DATABASE_NAME)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE appointments (
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

cursor.execute('''
CREATE TABLE booking_history (
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

# Add appointment slots
base_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
slots = []
for day in range(7):
    current = base_date + timedelta(days=day)
    if current.weekday() < 5:
        for hour in [9, 10, 11, 14, 15, 16]:
            for minute in [0, 20, 40]:
                slot_time = current.replace(hour=hour, minute=minute)
                slots.append((slot_time.strftime('%Y-%m-%d %H:%M:%S'), 'standard'))

cursor.executemany("INSERT INTO appointments (slot_time, appointment_type) VALUES (?, ?)", slots)
conn.commit()
conn.close()

print("✅ Database fixed! You can now restart your server.")
