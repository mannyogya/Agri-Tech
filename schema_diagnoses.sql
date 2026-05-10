-- Run this in Neon SQL Editor (or psql) once if POST /diagnose returns 500.
-- Table name and columns must match main.py INSERT.

CREATE TABLE IF NOT EXISTS diagnoses (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  client_id TEXT NOT NULL,
  language TEXT NOT NULL,
  symptom_text TEXT,
  disease TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  risk_level TEXT NOT NULL,
  advice TEXT NOT NULL,
  image_url TEXT
);

CREATE INDEX IF NOT EXISTS idx_diagnoses_client_created
  ON diagnoses (client_id, created_at DESC);
