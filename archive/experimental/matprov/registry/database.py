"""
Database manager for prediction registry
"""

from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from contextlib import contextmanager

from matprov.registry.models import Base


class Database:
    """
    Database connection manager for prediction registry.
    
    Supports SQLite for prototyping with migration path to PostgreSQL.
    """
    
    def __init__(self, database_url: Optional[str] = None, echo: bool = False):
        """
        Initialize database connection.
        
        Args:
            database_url: SQLAlchemy database URL. Defaults to SQLite in .matprov/predictions.db
            echo: If True, log all SQL statements
        """
        if database_url is None:
            # Default to SQLite in .matprov directory
            matprov_dir = Path.cwd() / ".matprov"
            matprov_dir.mkdir(parents=True, exist_ok=True)
            db_path = matprov_dir / "predictions.db"
            database_url = f"sqlite:///{db_path}"
        
        self.engine = create_engine(database_url, echo=echo)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._database_url = database_url
    
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)
    
    def drop_tables(self):
        """Drop all tables (DANGEROUS - use only for testing)."""
        Base.metadata.drop_all(self.engine)
    
    @contextmanager
    def session(self):
        """
        Provide a transactional scope for database operations.
        
        Usage:
            with db.session() as session:
                session.add(obj)
                session.commit()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """Get a new session (caller responsible for closing)."""
        return self.SessionLocal()
    
    @property
    def url(self) -> str:
        """Get the database URL."""
        return self._database_url
    
    def __repr__(self) -> str:
        return f"<Database(url={self._database_url})>"


# Example usage
if __name__ == "__main__":
    import tempfile
    from matprov.registry.models import Model, Prediction
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    db = Database(f"sqlite:///{db_path}", echo=False)
    db.create_tables()
    
    print(f"✅ Created database: {db}")
    
    # Add test data
    with db.session() as session:
        model = Model(
            model_id="test_model_v1",
            version="1.0",
            checkpoint_hash="test_hash_123",
            training_dataset_hash="test_dataset_456",
            architecture="TestNet"
        )
        session.add(model)
        session.flush()
        
        prediction = Prediction(
            prediction_id="PRED-TEST-001",
            model_id=model.id,
            material_formula="TestMaterial",
            predicted_tc=100.0,
            uncertainty=10.0
        )
        session.add(prediction)
    
    print(f"✅ Added test data")
    
    # Query
    with db.session() as session:
        models = session.query(Model).all()
        predictions = session.query(Prediction).all()
        print(f"✅ Models: {len(models)}")
        print(f"✅ Predictions: {len(predictions)}")
    
    # Cleanup
    import os
    os.unlink(db_path)
    print(f"✅ Database test complete")

