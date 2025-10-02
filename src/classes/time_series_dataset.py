import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import sys
import os
import json
from datetime import datetime as dt

# Carrega o .env automaticamente
dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=dotenv_path)

# Project root
PROJECT_ROOT = Path(dotenv_path).resolve().parent if dotenv_path else Path.cwd()

# Adiciona pasta src às variaveis de ambiente do sistema
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Adiciona funções úteis customizadas
from utils.paths import resolve_env_path
from utils.text_processing import normalize, levenshtein_substitute

# Adiciona caminhos importantes
ARRIVALS_PATH = resolve_env_path("ARRIVALS_PATH")
DATAFRAME_ENCODING = os.getenv("DATAFRAME_ENCODING")
ARRIVALS_SCHEMA_PATH = resolve_env_path("ARRIVALS_SCHEMA_PATH")
ARRIVALS_COLUMN_SYNONYMS_PATH = resolve_env_path("ARRIVALS_COLUMN_SYNONYMS_PATH")

# Definindo conjuntos relevantes de colunas
# group_cols = ["continent_id", "entry_route_id", "date"]
# value_col = "arrivals"

class TimeSeriesDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self._read_csv_data_from_path()

        # Table variables are filled in the pipeline.
        self.entry_route_table = None
        self.continent_table = None
        self.state_table = None

        # Runs pipelines
        self._pipeline()


    def _read_csv_data_from_path(self):
        """Reads csv files from a Path object and consolidate it in a dataframe"""

        # Read csv filenames
        csv_files = self.data_path.glob("*.csv")

        # Read column synonyms json: This is used to define the column name pattern
        with open(ARRIVALS_COLUMN_SYNONYMS_PATH, 'r') as f:
            synonyms = json.load(f)

        # Read each csv as a dataframe
        dfs = list()
        for i, file in enumerate(csv_files):
            # Read dataframe
            data = pd.read_csv(file, encoding=DATAFRAME_ENCODING, sep=";")

            # Normalize column names using normalize utils function
            data.columns = [normalize(col) for col in data.columns]

            # Rename each column to canonical set
            data = data.rename(columns=synonyms)
            data["file_index"] = f"File {i}"
            dfs.append(data)

        df = pd.concat(dfs, ignore_index=True)

        return df


    # Data wrangling
    def _standardize_string_columns(self):
        """Normalize the values for all string columns"""
        # Selecting all string columns
        str_columns = self.data.select_dtypes(include=["object", "string"]).columns.tolist()

        # Normalizing all string columns
        for col in str_columns:
            self.data[col] = self.data[col].apply(normalize)

    def _create_datetime_col(self):
        """Creates the datetime column using the year and month columns"""
        self.data["date"] = pd.to_datetime(
            dict(
                year=self.data["year"],
                month=self.data["month_id"],
                day=1)
                )

    def _clean_entry_route_column(self):
        """
        Substitutes values of the entry route column according to the levenshtein
        distance to target values.
        """
        self.data["entry_route"] = self.data["entry_route"].apply(levenshtein_substitute)

    def _create_equivalence_tables(self, id_col, value_col):
        return self.data[[value_col, id_col]].drop_duplicates().reset_index(drop=True)

    # Pipeline function
    def _pipeline(self):
        """Unify the calling for all data processing functions"""
        self._standardize_string_columns()
        self._create_datetime_col()
        self._clean_entry_route_column()

        # Pause: Get equivalences table
        self.entry_route_table = self._create_equivalence_tables(id_col="entry_route_id", value_col="entry_route")
        self.continent_table = self._create_equivalence_tables(id_col="continent_id", value_col="continent")
        self.state_table = self._create_equivalence_tables(id_col="state_id", value_col="state")

    # General methods
    def get_data_without_missing_values(self):
        """
        Removes continente_nao_especificado arrivals, which is equivalent to
        continent_id = 8.
        """
        return self.data.loc[self.data["continent_id"] != 8]

    # Continue pipeline
    # self.remove_missing_continent_arrivals()






