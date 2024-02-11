from . import db
from flask_login import UserMixin
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.sql import func

class Folder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    files = db.relationship('File', backref='folder', lazy=True)


class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String) 
    student_number = db.Column(db.String(20))
    name = db.Column(db.String(100))
    system_score = db.Column(db.Float)
    criteria_results = db.Column(JSON)
    folder_id = db.Column(db.Integer, db.ForeignKey('folder.id'), nullable=False)
    folder_rel = db.relationship('Folder', backref=db.backref('files_rel', lazy=True, cascade='all, delete-orphan'))
    


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    folders = db.relationship('Folder')


