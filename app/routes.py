from flask import Blueprint, request, jsonify

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return jsonify({
        'message': 'Spam Detection API!'
    }), 200