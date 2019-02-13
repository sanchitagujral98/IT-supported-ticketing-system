from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired, length

class ATSForm(FlaskForm):
    user_string = TextAreaField('Ticket String', validators=[DataRequired()])
    submit = SubmitField('Submit')
