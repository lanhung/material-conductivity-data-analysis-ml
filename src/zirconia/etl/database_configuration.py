import os
from pathlib import Path

class DatabaseConfiguration:
    """
    Central configuration for database connections, file paths, and schema definitions.
    """
    # MySQL Connection Details
    DATABASE_HOST = 'localhost'
    DATABASE_USER = 'root'
    DATABASE_PASSWORD = '123456root'
    DATABASE_PORT = '3306'
    DATABASE_NAME = 'zirconia_conductivity'

    # Local DuckDB File Path
    # Resolves to: project_root/data/zirconia_snapshot.duckdb
    LOCAL_DATABASE_FILE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "zirconia_snapshot.duckdb"

    # SQL Schema Definition (Data Definition Language)
    DUCKDB_SCHEMA_DEFINITION = """
                               -- Dictionary table for crystal structures
                               CREATE TABLE crystal_structure_dict (
                                                                       id          INTEGER PRIMARY KEY,
                                                                       code        VARCHAR NOT NULL UNIQUE,
                                                                       full_name   VARCHAR
                               );

                               -- Main table for material samples
                               CREATE TABLE material_samples (
                                                                 sample_id                  INTEGER PRIMARY KEY,
                                                                 reference                  VARCHAR,
                                                                 material_source_and_purity VARCHAR,
                                                                 synthesis_method           VARCHAR,
                                                                 processing_route           VARCHAR,
                                                                 operating_temperature      FLOAT,
                                                                 conductivity               DOUBLE
                               );

                               -- Detail table for dopants (Depends on material_samples)
                               CREATE TABLE sample_dopants (
                                                               id                    INTEGER PRIMARY KEY,
                                                               sample_id             INTEGER NOT NULL,
                                                               dopant_element        VARCHAR,
                                                               dopant_ionic_radius   FLOAT,
                                                               dopant_valence        INTEGER,
                                                               dopant_molar_fraction FLOAT,
                                                               FOREIGN KEY (sample_id) REFERENCES material_samples (sample_id)
                               );

                               -- Table for sintering steps (Depends on material_samples)
                               CREATE TABLE sintering_steps (
                                                                id                    INTEGER PRIMARY KEY,
                                                                sample_id             INTEGER NOT NULL,
                                                                step_order            INTEGER,
                                                                sintering_temperature FLOAT,
                                                                sintering_duration    FLOAT,
                                                                FOREIGN KEY (sample_id) REFERENCES material_samples (sample_id)
                               );

                               -- Association table for sample and crystal phases
                               CREATE TABLE sample_crystal_phases (
                                                                      sample_id      INTEGER NOT NULL,
                                                                      crystal_id     INTEGER NOT NULL,
                                                                      is_major_phase BOOLEAN DEFAULT TRUE,
                                                                      PRIMARY KEY (sample_id, crystal_id),
                                                                      FOREIGN KEY (sample_id) REFERENCES material_samples (sample_id),
                                                                      FOREIGN KEY (crystal_id) REFERENCES crystal_structure_dict (id)
                               ); \
                               """