import pandas as pd
from pathlib import Path


class DataLoader:
    """
    CSV, Excel, JSON ve Parquet formatlarını destekleyen akıllı veri yükleyici.
    Yükleme anında temel veri kalitesi kontrolü yapar.
    """

    SUPPORTED = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._validate()

    def _validate(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {self.file_path}")
        if self.file_path.suffix not in self.SUPPORTED:
            raise ValueError(
                f"Desteklenmeyen format: {self.file_path.suffix}. "
                f"Desteklenenler: {self.SUPPORTED}"
            )

    def load(self) -> pd.DataFrame:
        ext = self.file_path.suffix
        loaders = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
            ".json": pd.read_json,
            ".parquet": pd.read_parquet,
        }
        df = loaders[ext](self.file_path)
        print(
            f"✅ Yüklendi: {self.file_path.name} | "
            f"{df.shape[0]:,} satır, {df.shape[1]} sütun"
        )
        return df
