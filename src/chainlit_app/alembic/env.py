from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Configuration du chemin - méthode robuste
current_file = os.path.abspath(__file__)
alembic_dir = os.path.dirname(current_file)
chainlit_app_dir = os.path.dirname(alembic_dir)  
src_dir = os.path.dirname(chainlit_app_dir)
project_root = os.path.dirname(src_dir)

# Ajouter les chemins nécessaires
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# Imports après configuration du chemin
try:
    from src.chainlit_app.models import *
    from src.chainlit_app.database import DATABASE_URL
    print("✅ Imports réussis")
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    # Fallback - import direct
    import sys
    sys.path.append(chainlit_app_dir)
    from models import *
    from database import DATABASE_URL

# Configuration Alembic
config = context.config

# Configuration de l'URL de base de données
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Métadonnées pour autogenerate
target_metadata = SQLModel.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Utilisation de l'URL de configuration
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
