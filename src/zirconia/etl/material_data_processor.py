import os
import duckdb
import pandas
from .database_configuration import DatabaseConfiguration
from .mysql_to_duckdb_synchronizer import MySQLToDuckDBSynchronizer

class MaterialDataProcessor:
    """
    Responsible for loading raw data from the local DuckDB instance,
    performing SQL-based Feature Engineering (ETL), and returning
    a structured Pandas DataFrame for training.
    """

    def __init__(self, configuration: DatabaseConfiguration = DatabaseConfiguration):
        self.configuration = configuration
        self.database_file_path = str(self.configuration.LOCAL_DATABASE_FILE_PATH)

    def load_and_preprocess_data_for_training_piml(self) -> pandas.DataFrame:
        """
        Main method to retrieve and transform data.

        It performs the following:
        1. Checks if the database exists (syncs if missing).
        2. Connects to DuckDB.
        3. Executes the Feature Engineering SQL query.
        4. Returns a Pandas DataFrame.
        """
        self._ensure_database_exists()

        connection = duckdb.connect(self.database_file_path)

        print(">>> [Data Processor] Executing SQL Extraction and Transformation...")

        # This SQL query handles:
        # 1. Aggregation of dopant properties (weighted averages).
        # 2. Identification of the primary dopant.
        # 3. Aggregation of sintering statistics.
        # 4. Physical unit conversions (Celsius to Kelvin).
        # 5. Target variable transformation (Conductivity to Log10).
        feature_engineering_query = """
                                    WITH dopant_statistics AS (
                                        -- Aggregation: Calculate total dopant fraction and weighted average properties
                                        SELECT
                                            sample_id,
                                            SUM(dopant_molar_fraction) as total_dopant_fraction,

                                            -- Weighted Average Radius: Sum(Radius * Fraction) / Sum(Fraction)
                                            SUM(dopant_ionic_radius * dopant_molar_fraction) / NULLIF(SUM(dopant_molar_fraction), 0) as average_dopant_radius,

                                            -- Weighted Average Valence: Sum(Valence * Fraction) / Sum(Fraction)
                                            SUM(dopant_valence * dopant_molar_fraction) / NULLIF(SUM(dopant_molar_fraction), 0) as average_dopant_valence,

                                            COUNT(*) as number_of_dopants
                                        FROM sample_dopants
                                        GROUP BY sample_id
                                    ),

                                         primary_dopant_calculation AS (
                                             -- Window Function: Identify the dopant with the highest molar fraction
                                             SELECT
                                                 sample_id,
                                                 dopant_element as primary_dopant_element,
                                                 ROW_NUMBER() OVER (PARTITION BY sample_id ORDER BY dopant_molar_fraction DESC) as rank_number
                                             FROM sample_dopants
                                         ),

                                         sintering_statistics AS (
                                             -- Aggregation: Get max temperature and total duration
                                             SELECT
                                                 sample_id,
                                                 MAX(sintering_temperature) as maximum_sintering_temperature,
                                                 SUM(sintering_duration) as total_sintering_duration
                                             FROM sintering_steps
                                             GROUP BY sample_id
                                         )

                                    SELECT
                                        main_table.sample_id,
                                        main_table.material_source_and_purity,
                                        main_table.synthesis_method,

                                        -- Physical Transformation 1: Celsius to Kelvin
                                        (main_table.operating_temperature + 273.15) as temperature_kelvin,

                                        -- Physical Transformation 2: Conductivity to Log10(Conductivity)
                                        LOG10(main_table.conductivity) as log_conductivity,

                                        main_table.operating_temperature,
                                        main_table.conductivity,

                                        -- Dopant Features (Handling NULLs with COALESCE)
                                        COALESCE(stats.total_dopant_fraction, 0) as total_dopant_fraction,
                                        stats.average_dopant_radius,
                                        stats.average_dopant_valence,
                                        COALESCE(stats.number_of_dopants, 0) as number_of_dopants,

                                        -- Primary Dopant Feature (Changed alias from 'primary' to 'pd' to avoid keyword conflict)
                                        pd.primary_dopant_element,

                                        -- Sintering Features
                                        sinter.maximum_sintering_temperature,
                                        sinter.total_sintering_duration

                                    FROM material_samples AS main_table

                                             -- Left Joins to preserve samples even if they lack specific sub-data
                                             LEFT JOIN dopant_statistics AS stats
                                                       ON main_table.sample_id = stats.sample_id

                                        -- FIX: Changed alias from 'primary' (reserved keyword) to 'pd'
                                             LEFT JOIN primary_dopant_calculation AS pd
                                                       ON main_table.sample_id = pd.sample_id AND pd.rank_number = 1

                                             LEFT JOIN sintering_statistics AS sinter
                                                       ON main_table.sample_id = sinter.sample_id

                                    WHERE main_table.conductivity > 0
                                      AND main_table.operating_temperature IS NOT NULL; \
                                    """

        # Execute query and convert directly to Pandas DataFrame
        data_frame = connection.execute(feature_engineering_query).df()

        print(f">>> [Data Processor] Data loaded successfully. Shape: {data_frame.shape}")
        return data_frame

    def _ensure_database_exists(self):
        """
        Checks if the local database file exists.
        If not, it triggers the Synchronizer to create it.
        """
        if not os.path.exists(self.database_file_path):
            print(">>> [Data Processor] Local database not found. Triggering synchronization...")
            synchronizer = MySQLToDuckDBSynchronizer()
            synchronizer.synchronize_data()

if __name__ == "__main__":
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data()
    print(df.head())