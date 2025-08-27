from flask import Blueprint, request, jsonify
from app.resume_qa import ResumeQA
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

routes = Blueprint("routes", __name__)

# Initialize ResumeQA (load once for performance)
pdf_path = os.path.join(BASE_DIR, "..", "static", "Nachiket_Shinde_Resume_v6.pdf")
qa_engine = ResumeQA(pdf_path)

@routes.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        answer = qa_engine.ask(question)
        return jsonify({"question": question, "answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
