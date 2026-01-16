// =============================================================================
// LXX CHRISTOLOGICAL DIVERGENCES - Neo4j Schema
// =============================================================================
//
// Schema for storing LXX-MT divergences with Christological significance.
// Part of the Third Impossible Oracle: LXX Christological Extractor.
//
// Node Types:
// - LXXDivergence: Individual divergence between MT and LXX
// - ManuscriptWitness: Manuscript evidence for readings
// - NTQuotation: New Testament quotation data
// - PatristicWitness: Church Father interpretations
//
// Relationships:
// - (:Verse)-[:HAS_LXX_DIVERGENCE]->(:LXXDivergence)
// - (:LXXDivergence)-[:WITNESSED_BY]->(:ManuscriptWitness)
// - (:LXXDivergence)-[:QUOTED_IN_NT]->(:NTQuotation)
// - (:LXXDivergence)-[:INTERPRETED_BY]->(:PatristicWitness)
// =============================================================================

// -----------------------------------------------------------------------------
// Node Constraints and Indexes
// -----------------------------------------------------------------------------

// LXXDivergence node constraints
CREATE CONSTRAINT lxx_divergence_id_unique IF NOT EXISTS
FOR (d:LXXDivergence) REQUIRE d.divergence_id IS UNIQUE;

CREATE INDEX lxx_divergence_verse_id IF NOT EXISTS
FOR (d:LXXDivergence) ON (d.verse_id);

CREATE INDEX lxx_divergence_category IF NOT EXISTS
FOR (d:LXXDivergence) ON (d.christological_category);

CREATE INDEX lxx_divergence_score IF NOT EXISTS
FOR (d:LXXDivergence) ON (d.composite_score);

// ManuscriptWitness node constraints
CREATE CONSTRAINT manuscript_witness_id_unique IF NOT EXISTS
FOR (m:ManuscriptWitness) REQUIRE m.manuscript_id IS UNIQUE;

CREATE INDEX manuscript_witness_type IF NOT EXISTS
FOR (m:ManuscriptWitness) ON (m.manuscript_type);

CREATE INDEX manuscript_witness_century IF NOT EXISTS
FOR (m:ManuscriptWitness) ON (m.century_numeric);

// NTQuotation node constraints
CREATE CONSTRAINT nt_quotation_id_unique IF NOT EXISTS
FOR (q:NTQuotation) REQUIRE q.quotation_id IS UNIQUE;

CREATE INDEX nt_quotation_verse IF NOT EXISTS
FOR (q:NTQuotation) ON (q.nt_verse);

CREATE INDEX nt_quotation_follows_lxx IF NOT EXISTS
FOR (q:NTQuotation) ON (q.follows_lxx);

// PatristicWitness node constraints
CREATE CONSTRAINT patristic_witness_id_unique IF NOT EXISTS
FOR (p:PatristicWitness) REQUIRE p.witness_id IS UNIQUE;

CREATE INDEX patristic_witness_father IF NOT EXISTS
FOR (p:PatristicWitness) ON (p.father);

CREATE INDEX patristic_witness_era IF NOT EXISTS
FOR (p:PatristicWitness) ON (p.era);

CREATE INDEX patristic_christological IF NOT EXISTS
FOR (p:PatristicWitness) ON (p.christological_reading);

// -----------------------------------------------------------------------------
// Example Data Insertion Queries
// -----------------------------------------------------------------------------

// Example: Create LXX Divergence node
MERGE (d:LXXDivergence {divergence_id: 'div_ISA.7.14_0'})
SET d.verse_id = 'ISA.7.14',
    d.divergence_type = 'lexical',
    d.christological_category = 'virgin_birth',
    d.mt_text = 'העלמה',
    d.mt_text_transliterated = 'ha\'almah',
    d.mt_gloss = 'the young woman',
    d.lxx_text = 'ἡ παρθένος',
    d.lxx_text_transliterated = 'hē parthenos',
    d.lxx_gloss = 'the virgin',
    d.divergence_score = 0.90,
    d.christological_score = 0.95,
    d.manuscript_confidence = 0.85,
    d.composite_score = 0.91,
    d.updated_at = datetime()
RETURN d;

// Example: Link divergence to verse
MATCH (v:Verse {id: 'ISA.7.14'})
MATCH (d:LXXDivergence {divergence_id: 'div_ISA.7.14_0'})
MERGE (v)-[r:HAS_LXX_DIVERGENCE]->(d)
SET r.significance = d.composite_score,
    r.category = d.christological_category
RETURN v, r, d;

// Example: Create manuscript witness
MERGE (m:ManuscriptWitness {manuscript_id: '4QIsaᵃ'})
SET m.manuscript_type = 'DSS',
    m.display_name = 'Dead Sea Scrolls - Isaiah',
    m.date_range = '125-100 BCE',
    m.century_start = -1,
    m.century_end = -1,
    m.century_numeric = -1,
    m.reliability_score = 1.0,
    m.notes = 'Great Isaiah Scroll from Qumran Cave 1'
RETURN m;

// Example: Link divergence to manuscript witness
MATCH (d:LXXDivergence {divergence_id: 'div_PSA.22.16_0'})
MATCH (m:ManuscriptWitness {manuscript_id: '4QPsᶠ'})
MERGE (d)-[r:WITNESSED_BY]->(m)
SET r.reading = 'כארו',
    r.reading_transliterated = 'ka\'aru',
    r.supports_lxx = true,
    r.supports_mt = false
RETURN d, r, m;

// Example: Create NT quotation
MERGE (q:NTQuotation {quotation_id: 'quote_ISA.7.14_MAT.1.23'})
SET q.ot_verse = 'ISA.7.14',
    q.nt_verse = 'MAT.1.23',
    q.nt_text_greek = 'ἰδοὺ ἡ παρθένος ἐν γαστρὶ ἕξει καὶ τέξεται υἱόν',
    q.quote_type = 'exact',
    q.follows_lxx = true,
    q.follows_mt = false,
    q.verbal_agreement_lxx = 0.95,
    q.verbal_agreement_mt = 0.30,
    q.theological_significance = 'Matthew quotes LXX parthenos for virgin birth prophecy'
RETURN q;

// Example: Link divergence to NT quotation
MATCH (d:LXXDivergence {verse_id: 'ISA.7.14'})
MATCH (q:NTQuotation {quotation_id: 'quote_ISA.7.14_MAT.1.23'})
MERGE (d)-[r:QUOTED_IN_NT]->(q)
RETURN d, r, q;

// Example: Create patristic witness
MERGE (p:PatristicWitness {witness_id: 'patristic_ISA.7.14_justin'})
SET p.father = 'Justin Martyr',
    p.era = 'ante-nicene',
    p.work = 'Dialogue with Trypho',
    p.citation = 'Chapter 43',
    p.interpretation = 'The prophet Isaiah spoke of a virgin conceiving and bearing a son',
    p.text_preference = 'lxx',
    p.christological_reading = true
RETURN p;

// Example: Link divergence to patristic witness
MATCH (d:LXXDivergence {verse_id: 'ISA.7.14'})
MATCH (p:PatristicWitness {witness_id: 'patristic_ISA.7.14_justin'})
MERGE (d)-[r:INTERPRETED_BY]->(p)
RETURN d, r, p;

// -----------------------------------------------------------------------------
// Query Examples
// -----------------------------------------------------------------------------

// Query 1: Find high-significance Christological divergences
MATCH (v:Verse)-[r:HAS_LXX_DIVERGENCE]->(d:LXXDivergence)
WHERE d.christological_category IS NOT NULL
  AND d.composite_score >= 0.7
RETURN v.id, d.christological_category, d.composite_score
ORDER BY d.composite_score DESC
LIMIT 20;

// Query 2: Find divergences with NT quotation support
MATCH (d:LXXDivergence)-[:QUOTED_IN_NT]->(q:NTQuotation)
WHERE q.follows_lxx = true
RETURN d.verse_id, d.christological_category, q.nt_verse, q.verbal_agreement_lxx
ORDER BY q.verbal_agreement_lxx DESC;

// Query 3: Find divergences with strong manuscript evidence
MATCH (d:LXXDivergence)-[w:WITNESSED_BY]->(m:ManuscriptWitness)
WHERE w.supports_lxx = true
  AND m.century_numeric <= 0  // Pre-Christian or early Christian
RETURN d.verse_id, d.christological_category, m.manuscript_id, m.date_range, m.reliability_score
ORDER BY m.reliability_score DESC;

// Query 4: Find divergences with patristic consensus
MATCH (d:LXXDivergence)-[:INTERPRETED_BY]->(p:PatristicWitness)
WHERE p.christological_reading = true
WITH d, COUNT(p) as witness_count
WHERE witness_count >= 3
RETURN d.verse_id, d.christological_category, witness_count, d.composite_score
ORDER BY witness_count DESC, d.composite_score DESC;

// Query 5: Find virgin birth prophecies
MATCH (v:Verse)-[:HAS_LXX_DIVERGENCE]->(d:LXXDivergence)
WHERE d.christological_category = 'virgin_birth'
OPTIONAL MATCH (d)-[:QUOTED_IN_NT]->(q:NTQuotation)
OPTIONAL MATCH (d)-[:INTERPRETED_BY]->(p:PatristicWitness)
RETURN v.id, d.lxx_text, d.composite_score,
       COUNT(DISTINCT q) as nt_quotations,
       COUNT(DISTINCT p) as patristic_witnesses
ORDER BY d.composite_score DESC;

// Query 6: Manuscript timeline for a verse
MATCH (d:LXXDivergence {verse_id: $verse_id})-[:WITNESSED_BY]->(m:ManuscriptWitness)
RETURN m.manuscript_id, m.date_range, m.century_numeric,
       m.supports_lxx, m.supports_mt, m.reliability_score
ORDER BY m.century_numeric ASC;

// Query 7: NT preference analysis
MATCH (d:LXXDivergence)-[:QUOTED_IN_NT]->(q:NTQuotation)
WITH d,
     SUM(CASE WHEN q.follows_lxx THEN 1 ELSE 0 END) as lxx_count,
     SUM(CASE WHEN q.follows_mt THEN 1 ELSE 0 END) as mt_count,
     COUNT(q) as total_count
WHERE total_count > 0
RETURN d.verse_id, d.christological_category,
       lxx_count, mt_count, total_count,
       toFloat(lxx_count) / total_count as lxx_preference_ratio
ORDER BY lxx_preference_ratio DESC;

// Query 8: Comprehensive analysis for a book
MATCH (v:Verse)-[:HAS_LXX_DIVERGENCE]->(d:LXXDivergence)
WHERE v.id STARTS WITH $book_code
OPTIONAL MATCH (d)-[:QUOTED_IN_NT]->(q:NTQuotation)
OPTIONAL MATCH (d)-[:INTERPRETED_BY]->(p:PatristicWitness)
WITH v, d,
     COUNT(DISTINCT q) as nt_count,
     COUNT(DISTINCT CASE WHEN p.christological_reading THEN p END) as patristic_count
WHERE d.christological_category IS NOT NULL
RETURN v.id, d.christological_category, d.composite_score,
       nt_count, patristic_count
ORDER BY d.composite_score DESC;

// -----------------------------------------------------------------------------
// Cleanup Queries (for testing/development)
// -----------------------------------------------------------------------------

// Delete all LXX divergence data (USE WITH CAUTION!)
// MATCH (d:LXXDivergence) DETACH DELETE d;
// MATCH (m:ManuscriptWitness) DETACH DELETE m;
// MATCH (q:NTQuotation) DETACH DELETE q;
// MATCH (p:PatristicWitness) DETACH DELETE p;

// Delete divergences for a specific verse
// MATCH (d:LXXDivergence {verse_id: $verse_id}) DETACH DELETE d;
