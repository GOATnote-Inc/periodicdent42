-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- HTC Tier 1 Calibration Schema Migration - v0.4.0
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--
-- Adds columns for Tier 1 calibration metadata to htc_predictions table:
-- - lambda_corrected: Material-class-corrected electron-phonon coupling
-- - omega_debye: Literature or Lindemann-estimated Debye temperature  
-- - calibration_tier: Model version (empirical_v0.3, tier_1, tier_2, tier_3)
-- - prediction_uncertainty: Statistical uncertainty from Monte Carlo sampling
--
-- Also adds CHECK constraint to ensure BCS materials (tier_1) have Tc ≤ 200 K.
--
-- Author: GOATnote Autonomous Research Lab Initiative
-- Date: 2025-10-10
-- Dataset SHA256: 3a432837f7f7b00004c673d60ffee8f2e50096298b5d2af74fc081ab9ff98998
--
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEGIN;

-- Add new columns for Tier 1 calibration
ALTER TABLE htc_predictions
    ADD COLUMN IF NOT EXISTS lambda_corrected FLOAT DEFAULT NULL
        COMMENT 'Material-class-corrected electron-phonon coupling (dimensionless)',
    ADD COLUMN IF NOT EXISTS omega_debye FLOAT DEFAULT NULL
        COMMENT 'Debye temperature from literature or Lindemann formula (K)',
    ADD COLUMN IF NOT EXISTS calibration_tier VARCHAR(20) DEFAULT 'empirical_v0.3'
        COMMENT 'Model version: empirical_v0.3, tier_1, tier_2, tier_3',
    ADD COLUMN IF NOT EXISTS prediction_uncertainty FLOAT DEFAULT NULL
        COMMENT 'Statistical uncertainty from Monte Carlo sampling (K)';

-- Backfill existing records with 'empirical_v0.3' calibration tier
UPDATE htc_predictions
SET calibration_tier = 'empirical_v0.3'
WHERE calibration_tier IS NULL;

-- Add CHECK constraint: BCS materials (tier_1) must have Tc ≤ 200 K
-- (Excludes cuprates and high-pressure hydrides which use different physics)
ALTER TABLE htc_predictions
    ADD CONSTRAINT check_tc_bcs_range
    CHECK (
        tc_predicted <= 200.0
        OR calibration_tier != 'tier_1'
        OR calibration_tier IS NULL
    );

-- Create index on calibration_tier for efficient filtering
CREATE INDEX IF NOT EXISTS idx_htc_predictions_calibration_tier
    ON htc_predictions(calibration_tier);

-- Create index on xi_parameter + calibration_tier for stability queries
CREATE INDEX IF NOT EXISTS idx_htc_predictions_xi_tier
    ON htc_predictions(xi_parameter, calibration_tier);

-- Add comments for documentation
COMMENT ON COLUMN htc_predictions.lambda_corrected IS
    'Tier 1 calibration: Base lambda × material class correction factor (LAMBDA_CORRECTIONS)';

COMMENT ON COLUMN htc_predictions.omega_debye IS
    'Tier 1 calibration: Debye temperature from DEBYE_TEMP_DB or Lindemann formula';

COMMENT ON COLUMN htc_predictions.calibration_tier IS
    'Model version: empirical_v0.3 (pre-calibration), tier_1 (v0.4.0), tier_2 (DFT+multi-band), tier_3 (ML-corrected)';

COMMENT ON COLUMN htc_predictions.prediction_uncertainty IS
    'Statistical uncertainty (1σ) from Monte Carlo sampling of Debye temperature uncertainty';

COMMIT;

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-- ROLLBACK (if needed)
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

-- To rollback this migration:
--
-- BEGIN;
-- 
-- ALTER TABLE htc_predictions
--     DROP CONSTRAINT IF EXISTS check_tc_bcs_range;
-- 
-- DROP INDEX IF EXISTS idx_htc_predictions_calibration_tier;
-- DROP INDEX IF EXISTS idx_htc_predictions_xi_tier;
-- 
-- ALTER TABLE htc_predictions
--     DROP COLUMN IF EXISTS lambda_corrected,
--     DROP COLUMN IF EXISTS omega_debye,
--     DROP COLUMN IF EXISTS calibration_tier,
--     DROP COLUMN IF EXISTS prediction_uncertainty;
-- 
-- COMMIT;

