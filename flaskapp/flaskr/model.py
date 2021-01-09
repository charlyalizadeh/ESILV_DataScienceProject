from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from dataanalyse.datawrapper import DataWrapper
from dataanalyse.models import build_model, cross_validate, predict
from flaskr.db import get_db
from flaskr.queries import (
    db_get_models_comparison,
    db_get_predictions,
    db_delete_comparison,
    db_get_comparison
)
from flaskr.forms import (
        ComparisonSettingsForm,
        RFBaggingForm,
        DecisionTreeForm,
        RandomForestForm,
        LogisticRegressionForm
        )
from joblib import dump, load

bp = Blueprint('model', __name__, url_prefix='/model')


@bp.route('/init_comparison', methods=('GET', 'POST'))
def init_comparison():
    settings_form = ComparisonSettingsForm(request.form)
    if settings_form.validate_on_submit():
        name = settings_form.name.data
        cv = settings_form.cv.data
        scale_method = settings_form.scale_method.data
        db = get_db()
        cursor = db.cursor()
        cursor.execute('INSERT INTO comparison (name, cv, scale_method) VALUES (?, ?, ?);',
                       (name, cv, scale_method,))
        db.commit()
        comparison_id = cursor.lastrowid
        return redirect(url_for('model.comparison_settings', comparison_id=comparison_id))
    return render_template('model/init_comparison.html', settings_form=settings_form)


@bp.route('/comparison_settings/<int:comparison_id>', methods=('GET', 'POST'))
def comparison_settings(comparison_id):
    db = get_db()
    models = db.execute(
            'SELECT m.title, m.id, m.storage_path'
            ' FROM model m JOIN comparisonmodelpair cmp ON m.id = cmp.model_id'
            ' WHERE cmp.comparison_id = ?',
            (comparison_id,)
    ).fetchall()
    return render_template('model/comparison_settings.html',
                           models=models,
                           comparison_id=comparison_id)


@bp.route('/model_choice/<int:comparison_id>', methods=('GET', 'POST'))
def model_choice(comparison_id):
    return render_template('model/model_choice.html', comparison_id=comparison_id)


@bp.route('/model_settings/<int:comparison_id>/<model_type>', methods=('GET', 'POST'))
def model_settings(comparison_id, model_type):
    settings_form = get_forms(model_type, None)
    if request.method == 'POST':
        settings_form = get_forms(model_type, request.form)
        if request.method == 'POST':
            if settings_form.validate():
                add_model(comparison_id=comparison_id,
                          model_type=model_type,
                          **settings_form.data)
            return redirect(url_for('model.comparison_settings', comparison_id=comparison_id))
    return render_template('model/model_settings.html',
                           comparison_id=comparison_id,
                           model_type=model_type,
                           settings_form=settings_form)


@bp.route('/compare/<int:comparison_id>', methods=('GET', 'POST'))
def compare(comparison_id):
    db = get_db()
    comparison = db_get_comparison(db, comparison_id)
    data = get_datawrapper(scale=comparison['scale_method'])
    model_rows = db_get_models_comparison(db, comparison_id)
    models = [load(f"{query['storage_path']}/{query['id']}.joblib") for query in model_rows]
    accuracies_cv, best_model, best_accuracy = cross_validate(models, data.train_set, cv=comparison['cv'])
    title = [query['title'] for query in model_rows]
    accuracies_cv = dict(zip(title, accuracies_cv))
    accuracies_test = []
    for model in models:
        accuracy = predict(model, data)
        accuracies_test.append(accuracy)
    accuracies_test = dict(zip(title, accuracies_test))
    return render_template('model/compare.html',
                           comparison_id=comparison_id,
                           accuracies_cv=accuracies_cv,
                           accuracies_test=accuracies_test)


@bp.route('/delete_comparison/<int:comparison_id>', methods=('GET', 'POST'))
def delete_comparison(comparison_id):
    db = get_db()
    db_delete_comparison(db, comparison_id)
    return redirect(url_for('model.comparison_list'))


@bp.route('/delete_model/<int:comparison_id>/<int:model_id>', methods=('GET', 'POST'))
def delete_model(comparison_id, model_id):
    db = get_db()
    db.execute('DELETE FROM comparisonmodelpair WHERE model_id = ?', (model_id,))
    db.commit()
    return redirect(url_for('model.comparison_settings', comparison_id=comparison_id))


@bp.route('/comparison_list', methods=('GET', 'POST'))
def comparison_list():
    db = get_db()
    comparisons = db.execute(
        'SELECT *, count(*) AS nb_model FROM comparison cp \
        INNER JOIN comparisonmodelpair pair ON pair.comparison_id = cp.id \
        GROUP BY cp.id'
     ).fetchall()
    if comparisons:
        print(comparisons[0].keys())
    return render_template('model/comparison_list.html', comparisons=comparisons)



def get_datawrapper(path_trn="flaskr/static/data/sat.trn", path_tst="flaskr/static/data/sat.tst", scale="none"):
    data = DataWrapper()
    data.import_train_set_from_txt(path_trn)
    data.import_test_set_from_txt(path_tst)
    if scale != "none":
        data.scale(scale)
    return data


def add_model(comparison_id, model_type, **kwargs):
    # Build model
    model = None
    if model_type == "Random Forest Bagging":
        # Not the best I wrote. Don't judge
        keys_decision_tree = list(DecisionTreeForm().data.keys())
        decision_tree_args = {}
        for key in keys_decision_tree:
            if key in ('csrf_token', 'submit'):
                kwargs.pop(key, None)
            else:
                decision_tree_args[key] = kwargs.pop(key, None)
        base_estimator = build_model("Decision Tree", **decision_tree_args)
        kwargs['base_estimator'] = base_estimator
        model = build_model(model_type, **kwargs)
    else:
        kwargs.pop('csrf_token', None)
        kwargs.pop('submit', None)
        model = build_model(model_type, **kwargs)

    # Insert model in database
    db = get_db()
    cursor = db.cursor()
    cursor.execute('INSERT INTO model (title) VALUES (?)', (model.__str__(),))
    db.commit()
    model_id = cursor.lastrowid

    # Save the model in statics files
    dump(model, f'flaskr/static/models/{model_id}.joblib')

    # Create a comparison/model pair in DB
    db.execute('INSERT INTO comparisonmodelpair (comparison_id, model_id) VALUES (?, ?)',
               (comparison_id, model_id,))
    db.commit()


def get_forms(model_type, form):
    options_form = None
    if model_type == "Random Forest Bagging":
        options_form = RFBaggingForm(form)
    elif model_type == "Random Forest":
        options_form = RandomForestForm(form)
    elif model_type == "Decision Tree":
        options_form = DecisionTreeForm(form)
    elif model_type == "Logistic Regression":
        options_form = LogisticRegressionForm(form)
    return options_form
