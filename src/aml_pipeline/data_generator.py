# IMPORTS
import uuid     # Generates unique IDs (for transaction_id)
import random   
import numpy as np
import pandas as pd
from faker import Faker # Generates realistic fake names/companies

# Will pick randomly from these fixed lists
CURRENCIES = ["USD", "TWD", "EUR", "JPY"]
PURPOSE_CODES = ["TRADE_SETTLEMENT", "PAYROLL", "FAMILY_SUPPORT", "SERVICES", "UNSPECIFIED"]



