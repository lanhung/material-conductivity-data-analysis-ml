import duckdb
import os
from pathlib import Path
from .database_configuration import DatabaseConfiguration

class MySQLToDuckDBSynchronizer:
    """
    Handles the synchronization of data from a remote MySQL database
    to a local DuckDB snapshot file.
    """

    def __init__(self, configuration: DatabaseConfiguration = DatabaseConfiguration):
        self.configuration = configuration
        self.database_file_path = str(self.configuration.LOCAL_DATABASE_FILE_PATH)
        self.connection = None

    def synchronize_data(self):
        """
        Main execution method to perform the full synchronization process.
        """
        self._remove_existing_snapshot()
        self._initialize_connection()
        self._create_tables()
        self._attach_remote_database()
        self._copy_data_from_remote()
        self._detach_remote_database()

        print("-" * 40)
        print(f">>> [Success] Synchronization complete.")
        print(f">>> File saved at: {self.database_file_path}")

    def _remove_existing_snapshot(self):
        """
        Removes the old database file to ensure a fresh full synchronization.
        """
        if os.path.exists(self.database_file_path):
            print(f">>> [Initialization] Removing old snapshot: {self.database_file_path}")
            os.remove(self.database_file_path)

    def _initialize_connection(self):
        """
        Creates a new DuckDB connection.
        """
        print(f">>> [Initialization] Creating new DuckDB snapshot...")
        # Ensure the directory exists
        Path(self.database_file_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(self.database_file_path)

    def _create_tables(self):
        """
        Executes the SQL schema definition to create empty tables in DuckDB.
        """
        print(">>> [Step 1] Creating Tables (Schema Generation)...")
        try:
            self.connection.execute(self.configuration.DUCKDB_SCHEMA_DEFINITION)
            print("    -> Tables created successfully.")
        except Exception as error_message:
            print(f"    [Error] Schema execution failed: {error_message}")
            raise error_message

    def _attach_remote_database(self):
        """
        Installs the MySQL extension and attaches the remote database as a virtual source.
        """
        print(">>> [Step 2] Attaching Remote MySQL Database...")
        self.connection.execute("INSTALL mysql; LOAD mysql;")

        connection_string = (
            f"host={self.configuration.DATABASE_HOST} "
            f"user={self.configuration.DATABASE_USER} "
            f"password={self.configuration.DATABASE_PASSWORD} "
            f"port={self.configuration.DATABASE_PORT} "
            f"database={self.configuration.DATABASE_NAME}"
        )

        # Attach as 'source_mysql_database' to avoid confusion
        self.connection.execute(f"ATTACH '{connection_string}' AS source_mysql_database (TYPE MYSQL)")

    def _copy_data_from_remote(self):
        """
        Iterates through the table list and copies data from MySQL to DuckDB.
        """
        print(">>> [Step 3] Synchronizing Data (MySQL -> DuckDB)...")

        # Order is critical due to Foreign Key constraints
        tables_to_synchronize = [
            'crystal_structure_dict',
            'material_samples',         # Main Parent Table
            'sample_dopants',           # Child of material_samples
            'sintering_steps',          # Child of material_samples
            'sample_crystal_phases'     # Child of material_samples AND crystal_structure_dict
        ]

        success_count = 0

        for table_name in tables_to_synchronize:
            print(f"    -> Synchronizing table: {table_name} ...", end=" ")
            try:
                # Direct insertion from the attached MySQL source
                query = f"INSERT INTO {table_name} SELECT * FROM source_mysql_database.{table_name}"
                self.connection.execute(query)

                # Count rows for verification
                row_count = self.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                print(f"Done. (Rows: {row_count})")
                success_count += 1
            except Exception as error_message:
                print(f"\n       [Error] Failed to synchronize {table_name}: {error_message}")
                raise error_message

    def _detach_remote_database(self):
        """
        Detaches the remote MySQL connection to clean up resources.
        """
        if self.connection:
            self.connection.execute("DETACH source_mysql_database")

if __name__ == "__main__":
    synchronizer = MySQLToDuckDBSynchronizer()
    synchronizer.synchronize_data()