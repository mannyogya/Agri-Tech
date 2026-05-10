-- Run once in Neon after deploying main.py that saves treatments.
ALTER TABLE diagnoses ADD COLUMN IF NOT EXISTS treatments_json TEXT;
