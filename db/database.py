import aiosqlite

class DatabaseManager:
    def __init__(self, db_path='cattle_management.db'):
        self.db_path = db_path

    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS animals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    breed TEXT NOT NULL,
                    last_vaccination_date TEXT,
                    registration_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    location TEXT,
                    farmer_id TEXT,
                    notes TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    animal_id INTEGER,
                    breed_predicted TEXT,
                    confidence REAL,
                    processing_time_ms REAL,
                    prediction_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (animal_id) REFERENCES animals (id)
                )
            """)
            await db.commit()
