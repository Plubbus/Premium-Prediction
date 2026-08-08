CREATE OR REPLACE VIEW "Claim_join" AS
WITH Aggregated_Sev AS(
	SELECT "IDpol", SUM("ClaimAmount") AS "Total_Claim_Amount"
	FROM "ClaimSev"
	GROUP BY "IDpol"
)
SELECT
	f.*, COALESCE("Total_Claim_Amount", 0) AS "ClaimAmount"
	FROM "ClaimFreq" f
		LEFT JOIN Aggregated_Sev s
			ON f."IDpol" = s."IDpol";

SELECT * FROM "Claim_Join";