from flask import Blueprint, request, jsonify
from .model import predict_spam

main = Blueprint('main', __name__)

# Home route
@main.route('/')
def home():
    return jsonify({
        'message': 'Spam Detection API!'
    }), 200


# Prediction route
@main.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        message = data.get('message')

        if message == '':
            return jsonify({
                'reason': 'Message cannot be empty.'
            }), 400
        
        
        label = predict_spam(message)

        return jsonify({
            'message': message,
            'label': label
        }), 200
    else:
        return jsonify({
            'reason': 'Wrong method used, please use POST'
        }), 405