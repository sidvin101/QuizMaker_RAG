from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from utils import extract_text, chunk_text, store_embeddings, generate_qa, parse_mcqs, generate_qa_with_retry, clear_entire_index
from werkzeug.utils import secure_filename
import PyPDF2
from dotenv import load_dotenv
import random

# Create  Flask application
application = Flask(__name__)
application.secret_key = os.urandom(24)
load_dotenv()

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Main index route for pdf upload
@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files["pdf"]
        
        #Get the parameter from the form
        chunk_size = int(request.form.get("chunk_size", 1000))
        num_questions = int(request.form.get("num_questions", 2))
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            #Extract and chunk text from PDF
            text = extract_text(filepath)
            #chunks = chunk_text(text)
            chunks = chunk_text(text, chunk_size=chunk_size)
            namespace = os.path.splitext(filename)[0]
            store_embeddings(chunks, namespace)
            flash("PDF processed and embeddings stored successfully.", "success")
            print(f"PDF stored into {namespace}")

            # Create context and generate questions
            random_chunks = random.sample(chunks, min(3, len(chunks)))
            context = " ".join(random_chunks)
            #qa_output = generate_qa(context)
            qa_output = generate_qa(context, num_questions=num_questions)
            #Apply retry logic for question generation (Comment the above and uncomment the below if you would like to use this)
            '''
            try:
                qa_output = generate_qa_with_retry(context, num_questions=num_questions)
                flash("Questions generated successfully.", "success")
            except Exception as e:
                flash(f"Error generating questions: {str(e)}", "error")
                return redirect(url_for('index'))
            '''
            print("Generated QA Output:", qa_output)
            questions = parse_mcqs(qa_output)
            session['questions'] = questions

            # Clear Pinecone index
            clear_entire_index()

            return redirect(url_for('quiz'))
        
    return render_template('index.html')

# Route for the quiz page
@application.route('/quiz', methods=['GET', 'POST'])
def quiz():
    questions = session.get('questions', [])
    if request.method == "POST":
        user_answers = [request.form.get(f"q{idx}") for idx in range(len(questions))]
        session['user_answers'] = user_answers
        return redirect(url_for('results'))
    return render_template('quiz.html', questions=questions)

# Route for displaying results
@application.route('/results', methods=['GET'])
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

# Run the Flask application
if __name__ == '__main__':
    application.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
