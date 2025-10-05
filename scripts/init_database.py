#!/usr/bin/env python3
"""
Initialize database schema by calling init_database().

This creates all tables defined in src.services.db using SQLAlchemy's Base.metadata.create_all().
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from src.services import db

def main():
    """Initialize database schema."""
    print("Initializing database schema...")
    
    try:
        db.init_database()
        print("✅ Database schema initialized successfully!")
        print("\nCreated tables:")
        print("  - experiments")
        print("  - optimization_runs")
        print("  - ai_queries")
        print("  - experiment_runs")
        print("  - instrument_runs")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)
    
    finally:
        db.close_database()

if __name__ == "__main__":
    main()
