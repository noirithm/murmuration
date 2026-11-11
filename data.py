"""Data loading. customer_id is the index everywhere."""
import numpy as np
import pandas as pd

from config import DATA_PATH

NUM = ["age", "balance_slog", "day", "campaign_log", "pdays_contacted", "previous_log"]
CAT = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
FEATURES = NUM + CAT


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["balance_slog"] = np.sign(out["balance"]) * np.log1p(out["balance"].abs())
    out["campaign_log"] = np.log1p(out["campaign"])
    out["pdays_contacted"] = (out["pdays"] >= 0).astype(int)
    out["previous_log"] = np.log1p(out["previous"])
    return out[FEATURES]


def load_data():
    xls = pd.ExcelFile(DATA_PATH)
    train = xls.parse("Stage1_Train").set_index("customer_id")
    test = xls.parse("Stage1_Test").set_index("customer_id")
    stage2 = xls.parse("Stage2_Data").set_index("customer_id")
    cons = xls.parse("Stage2_Constraints").set_index("symbol")["value"]
    limits = {"N": int(cons["N"]), "K": int(cons["K"]), "C": int(cons["C"]), "H": int(cons["H"])}
    y = (train.pop("y") == "yes").astype(int)
    assert test.index.is_unique and not set(test.index) & set(train.index)
    assert set(test.index) <= set(stage2.index)
    return engineer(train), y, engineer(test), stage2, limits
