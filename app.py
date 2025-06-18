from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from utils import extract_text, chunk_text, store_embeddings, generate_qa, parse_mcqs
from werkzeug.utils import secure_filename
import PyPDF2
from dotenv import load_dotenv

# Great Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a random secret key
load_dotenv()

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Main index route for pdf upload
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files["pdf"]
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            #Extract and chunk text from PDF
            text = extract_text(filepath)
            chunks = chunk_text(text)
            namespace = os.path.splitext(filename)[0]
            store_embeddings(chunks, namespace)
            flash("PDF processed and embeddings stored successfully.", "success")

            # Create context and generate questions
            context = " ".join(chunks[:3])
            qa_output = generate_qa(context)
            print("Generated QA Output:", qa_output)
            questions = parse_mcqs(qa_output)
            session['questions'] = questions

            return redirect(url_for('quiz'))
        
    return render_template('index.html')

# Route for the quiz page
@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    questions = session.get('questions', [])
    if request.method == "POST":
        user_answers = [request.form.get(f"q{idx}") for idx in range(len(questions))]
        session['user_answers'] = user_answers
        return redirect(url_for('results'))
    return render_template('quiz.html', questions=questions)

# Route for displaying results
@app.route('/results', methods=['GET'])
def results():
    questions = session.get('questions', [])
    user_answers = session.get('user_answers', [])
    score = 0
    results = []

    for i, q in enumerate(questions):
        user_ans = user_answers[i]
        correct = q['correct']
        explanation = q['explanation']
        is_correct = user_ans == correct
        if is_correct:
            score += 1
        results.append({
            "index": i + 1,
            "question": q['question'],
            "user_ans": user_ans,
            "correct_answer": correct,
            "is_correct": is_correct,
            "explanation": explanation
        })

    return render_template('results.html', score=score, total=len(questions), results=results)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
