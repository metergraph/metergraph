-- Older calls deliberately retain unknown reliability coverage.
ALTER TABLE calls ADD COLUMN http_status_code integer CHECK (http_status_code BETWEEN 100 AND 599);
ALTER TABLE calls ADD COLUMN availability text CHECK (availability IN ('available', 'unavailable', 'excluded'));
ALTER TABLE calls ADD COLUMN refused boolean;
