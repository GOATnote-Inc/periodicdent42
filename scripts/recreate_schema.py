#!/usr/bin/env python3
"""
Drop and recreate database schema with latest models.
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from src.services.db import Base, get_database_url
from sqlalchemy import create_engine, text

def main():
    """Drop and recreate all tables."""
    print("🔄 Recreating database schema...")
    
    try:
        # Create engine
        url = get_database_url()
        engine = create_engine(url)
        
        # Drop all tables
        print("🗑️  Dropping existing tables...")
        Base.metadata.drop_all(bind=engine)
        print("✅ Tables dropped")
        
        # Recreate with current schema
        print("🔨 Creating tables with updated schema...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created successfully!")
        
        # Verify tables
        with engine.connect() as conn:
            result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
            tables = [row[0] for row in result]
            print(f"\n📋 Tables in database: {', '.join(tables)}")
            
            # Check optimization_runs schema
            result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='optimization_runs' ORDER BY ordinal_position"))
            cols = [f"{row[0]} ({row[1]})" for row in result]
            print(f"\n📊 optimization_runs columns:")
            for col in cols:
                print(f"   - {col}")
        
        engine.dispose()
        print("\n✅ Schema recreation complete!")
        
    except Exception as e:
        print(f"❌ Error recreating schema: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
