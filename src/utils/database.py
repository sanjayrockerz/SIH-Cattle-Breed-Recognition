"""
Database utilities for animal registration and management
"""

import sqlite3
import streamlit as st
from datetime import datetime


@st.cache_resource
def setup_database():
    """Setup SQLite database for animal registration"""
    conn = sqlite3.connect("vaccination.db", check_same_thread=False)
    c = conn.cursor()
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS animals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            breed TEXT NOT NULL,
            last_vaccination_date TEXT,
            registration_date TEXT DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
    """)
    
    conn.commit()
    return conn, c


def get_animals_from_db(cursor):
    """Get all registered animals from database"""
    try:
        cursor.execute("SELECT name, breed, last_vaccination_date FROM animals ORDER BY id DESC")
        return cursor.fetchall()
    except:
        return []


def register_animal(cursor, conn, name, breed, last_vaccination_date, notes=""):
    """Register a new animal in the database"""
    try:
        cursor.execute(
            "INSERT INTO animals (name, breed, last_vaccination_date, notes) VALUES (?,?,?,?)",
            (name, breed, last_vaccination_date, notes)
        )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False


def get_vaccination_status(last_vaccination_date):
    """Calculate vaccination status based on last vaccination date"""
    try:
        last_date = datetime.strptime(last_vaccination_date, "%Y-%m-%d").date()
        today = datetime.now().date()
        days_since = (today - last_date).days
        
        if days_since > 180:
            return "🔴 Overdue", days_since
        elif days_since > 150:
            return "🟡 Due Soon", days_since
        else:
            return "🟢 Current", days_since
    except:
        return "❓ Unknown", 0