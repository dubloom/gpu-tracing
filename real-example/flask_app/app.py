from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
import torch

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///results.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define ORM Model
class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.Float, nullable=False)

# Initialize the database
with app.app_context():
    db.create_all()

@app.route('/matmul')
def matmul():
    # Perform a large matrix multiplication
    A = torch.randn(10_000, 10_000)
    B = torch.randn(10_000, 10_000)
    result_value = torch.matmul(A, B).sum().item()

    # Store result in the database
    new_result = Result(value=result_value)
    db.session.add(new_result)
    db.session.commit()

    return jsonify({'result': result_value})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=56000)
