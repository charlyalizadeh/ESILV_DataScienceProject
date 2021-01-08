DROP TABLE IF EXISTS model;
DROP TABLE IF EXISTS comparison;
DROP TABLE IF EXISTS comparisonmodelpair;
DROP TABLE IF EXISTS user;

CREATE TABLE model (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    storage_path TEXT NOT NULL DEFAULT "flaskr/static/models",
    title TEXT NOT NULL
);

CREATE TABLE comparison (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    solved BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE comparisonmodelpair (
    comparison_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    FOREIGN KEY (comparison_id) REFERENCES comparison (id),
    FOREIGN KEY (model_id) REFERENCES model (id),
    PRIMARY KEY (comparison_id, model_id)
);
