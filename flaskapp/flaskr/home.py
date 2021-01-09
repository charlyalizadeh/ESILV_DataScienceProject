from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('home', __name__)


@bp.route('/')
def index():
    return render_template('home/home.html')


@bp.route('/not_implemented')
def not_implemented():
    return render_template('home/not_implemented.html')
