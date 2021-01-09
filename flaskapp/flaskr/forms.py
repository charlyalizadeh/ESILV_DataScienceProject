from flask_wtf import FlaskForm
from wtforms import (
    IntegerField,
    SelectField,
    BooleanField,
    FloatField,
    SubmitField,
    StringField,
)
from wtforms.validators import DataRequired, NumberRange, Optional, InputRequired


class ComparisonSettingsForm(FlaskForm):
    name = StringField("Name model", validators=[InputRequired()])
    cv = IntegerField("Cross-validation folds", default=5, validators=[InputRequired()])
    scale_method = SelectField("Scale", choices=["none", "standard", "minmax", "normalize"], default="standard")
    submit = SubmitField("Compare")


class DecisionTreeForm(FlaskForm):
    criterion = SelectField("Criterion", choices=["gini", "entropy"], default="gini", validators=[DataRequired()])
    splitter = SelectField("Splitter", choices=["best", "random"], default="best", validators=[DataRequired()])
    max_depth = IntegerField("Maximum depth", default=None, filters=[lambda x: x or None], validators=[Optional(), NumberRange(2)])
    min_weight_fraction_leaf = FloatField("Minimum weight fraction leaf", default=0.0, validators=[NumberRange(0, 0.5)])
    max_leaf_nodes = IntegerField("Maximum leaf nodes", default=None, filters=[lambda x: x or None], validators=[Optional(), NumberRange(2)])
    submit = SubmitField("Add")


class RFBaggingForm(DecisionTreeForm):
    n_estimators = IntegerField("Number of estimators", default=10, validators=[DataRequired()])
    bootstrap = BooleanField("Bootstrap", default=True)
    bootstrap_features = BooleanField("Bootstrap features", default=False)
    random_state = IntegerField("Seed", default=None, filters=[lambda x: x or None], validators=[Optional()])


class RandomForestForm(FlaskForm):
    n_estimators = IntegerField("Number of estimators", default=100, validators=[DataRequired()])
    criterion = SelectField("Criterion", choices=["gini", "entropy"], default="gini", validators=[DataRequired()])
    max_depth = IntegerField("Maximum depth", default=None, filters=[lambda x: x or None], validators=[Optional(), NumberRange(2)])
    min_weight_fraction_leaf = FloatField("Minimum weight fraction leaf", default=0.0, validators=[NumberRange(0, 0.5)])
    max_leaf_nodes = IntegerField("Maximum leaf nodes", default=None, filters=[lambda x: x or None], validators=[Optional(), NumberRange(2)])
    bootstrap = BooleanField("Bootstrap", default=True)
    random_state = IntegerField("Seed", default=None, filters=[lambda x: x or None], validators=[Optional()])
    submit = SubmitField("Add")


class LogisticRegressionForm(FlaskForm):
    tol = FloatField("Tolerance", default=1e-4, validators=[DataRequired()])
    C = FloatField("Inverse of refularization strength", default=1.0, validators=[DataRequired()])
    fit_intercept = BooleanField("Add constant to decision function", default=True)
    random_state = IntegerField("Seed", default=None, filters=[lambda x: x or None], validators=[Optional()])
    solver = SelectField("Solver",
                         choices=["newton-cg",
                                  "lbfgs",
                                  "liblinear",
                                  "sag",
                                  "saga"],
                         default="lbfgs",
                         validators=[DataRequired()])
    max_iter = IntegerField("Max iteration", default=10000, validators=[DataRequired(), NumberRange(2)])
    submit = SubmitField("Add")
