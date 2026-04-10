import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env 
load_dotenv()

app = Flask(__name__)

# Securely access credentials 
DOCKER_USER = os.getenv("DOCKER_USERNAME")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "user_context": DOCKER_USER})