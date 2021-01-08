def get_models_comparison(db, comparison_id):
    models = db.execute(
            'SELECT m.title, m.id, m.storage_path'
            ' FROM model m JOIN comparisonmodelpair cmp ON m.id = cmp.model_id'
            ' WHERE cmp.comparison_id = ?', (comparison_id,)
        ).fetchall()
    db.commit()
    return models


def get_predictions(db, models_id, scale_method):
    predictions = db.execute(
            'SELECT p.model_id, p.accuracy_test FROM prediction p'
            ' WHERE p.model_id IN (?) AND scale_method = (?)',
            (models_id, scale_method,)
    )
    db.commit()
    return predictions
