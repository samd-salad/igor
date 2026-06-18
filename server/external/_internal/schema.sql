PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS episodes (
    episode_id          TEXT PRIMARY KEY,
    occurred_at         TEXT NOT NULL,
    speaker_id          TEXT,
    participants        TEXT,
    intent              TEXT,
    raw_utterance       TEXT NOT NULL,
    tool_calls          TEXT,
    emotional_tone      TEXT,
    summary             TEXT,
    consolidated_at     TEXT
);
CREATE INDEX IF NOT EXISTS episodes_unconsolidated
    ON episodes(occurred_at) WHERE consolidated_at IS NULL;

CREATE TABLE IF NOT EXISTS facts (
    fact_id             TEXT PRIMARY KEY,
    category            TEXT NOT NULL,
    key                 TEXT NOT NULL,
    value               TEXT NOT NULL,
    tags                TEXT,
    source_episode_id   TEXT REFERENCES episodes(episode_id),
    embedding           BLOB,
    valid_at            TEXT NOT NULL,
    invalid_at          TEXT,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_active
    ON facts(category, key) WHERE invalid_at IS NULL;
CREATE INDEX IF NOT EXISTS facts_episode ON facts(source_episode_id);

CREATE TABLE IF NOT EXISTS identity (
    id                            INTEGER PRIMARY KEY CHECK (id = 1),
    narrative                     TEXT NOT NULL,
    last_consolidated_at          TEXT,
    last_consolidated_episode_id  TEXT
);

CREATE TABLE IF NOT EXISTS reflections (
    reflection_id      TEXT PRIMARY KEY,
    occurred_at        TEXT NOT NULL,
    note               TEXT NOT NULL,
    source_episode_id  TEXT REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id        TEXT PRIMARY KEY,
    occurred_at        TEXT NOT NULL,
    issue              TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'open',
    source_episode_id  TEXT REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS reminders (
    reminder_id        TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    fire_at            TEXT NOT NULL,
    room_id            TEXT,
    status             TEXT NOT NULL DEFAULT 'pending',
    source_episode_id  TEXT REFERENCES episodes(episode_id)
);
