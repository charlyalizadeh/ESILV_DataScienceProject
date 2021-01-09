def db_get_models_comparison(db, comparison_id):
    models = db.execute(
            'SELECT m.title, m.id, m.storage_path'
            ' FROM model m JOIN comparisonmodelpair cmp ON m.id = cmp.model_id'
            ' WHERE cmp.comparison_id = ?', (comparison_id,)
        ).fetchall()
    db.commit()
    return models


def db_get_predictions(db, models_id, scale_method):
    predictions = db.execute(
            'SELECT p.model_id, p.accuracy_test FROM prediction p'
            ' WHERE p.model_id IN (?) AND scale_method = (?)',
            (models_id, scale_method,)
    )
    db.commit()
    return predictions


def db_delete_comparison(db, comparison_id, delete_models=True):
    if delete_models:
        models_queries = db.execute(
                'SELECT m.id'
                ' FROM model m INNER JOIN comparisonmodelpair cmp ON m.id = cmp.model_id'
                ' WHERE cmp.comparison_id = ?', (comparison_id,)
                ).fetchall()
        models_id = [query['id'] for query in models_queries]
        query = f"DELETE FROM model WHERE id IN ({','.join(['?']*len(models_id))})"
        db.execute(query, models_id)
    db.execute('DELETE FROM comparison WHERE id = (?)', (comparison_id,))
    db.commit()


def db_get_comparison(db, comparison_id):
    comparison = db.execute("SELECT * FROM comparison WHERE id = ?", (comparison_id,)).fetchone()
    return comparison
    
