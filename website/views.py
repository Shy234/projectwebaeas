import os
from flask import Blueprint, render_template, request, flash, jsonify, redirect, url_for,send_file
from flask_login import login_required, current_user
from .models import Folder, File
from . import db
from textblob import TextBlob 
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import docx2txt 
from docx import Document
from nltk import pos_tag
from nltk.tokenize import word_tokenize ,sent_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import pandas as pd
import re
from fpdf import FPDF


views = Blueprint('views', __name__)

nlp = spacy.load("en_core_web_sm")

def score_thesis(statement):
    if statement:
        score = 0
        if len(statement) > 10:
            score += 1
        if '?' in statement:
            score += 1
        return score
    else:
        return 0

def score_introduction(intro):
    if intro:
        score = 0
        if "Background:" in intro:
            score += 1
            if re.search(r'\b(?:grab|capture|hook)\b', intro, re.IGNORECASE):
                score += 1
        return score
    else:
        return 0

def tokenize_sentences(essay_text):
    return [sent.text for sent in nlp(essay_text).sents]

def vectorize_sentences(sentences):
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    return sentence_vectors

def train_organization_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def process_docx(essay):
    return docx2txt.process(essay)

def assess_focus_and_details(essay_text, additional_question):
    try:

        doc = nlp(essay_text)
        focus_score = len(doc)

        named_entities_score = len(doc.ents)


        sentence_score = len(list(doc.sents))
        unique_words_score = len(set(token.text.lower() for token in doc if token.is_alpha))

        additional_question_score = len(additional_question.split())

        total_score = focus_score + named_entities_score + sentence_score + unique_words_score + additional_question_score


        min_possible_score = 5  
        max_possible_score = 25  
        scaled_score = 1 + ((total_score - min_possible_score) / (max_possible_score - min_possible_score)) * 4

  
        percentage_score = min(5, max(1, scaled_score))

        return percentage_score

    except Exception as e:
        print(f"Error in assess_focus_and_details: {str(e)}")
        return None

def assess_relevance(essay, additionalQuestion1, relevance_words):
    question_doc = nlp(additionalQuestion1)
    essay_doc = nlp(essay)

    all_relevance_words = set(relevance_words)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([essay, additionalQuestion1] + list(all_relevance_words))

    similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]

    relevance_scale = {
        0.0: 1,
        0.1: 2,
        0.2: 3,
        0.4: 4,
        0.6: 5,
    }

    closest_match = min(relevance_scale.keys(), key=lambda x: abs(x - similarity_score))
    assigned_relevance_score = relevance_scale[closest_match]

    return assigned_relevance_score

#end

def assess_clarity(essay_text):
    doc = nlp(essay_text)
    complex_sentences = [sent for sent in doc.sents if len(sent) > 20] 
    
    total_sentences = len(list(doc.sents))

    if total_sentences == 0:
        return 1 
    
    clarity_score = 1 - len(complex_sentences) / total_sentences
    

    if clarity_score <= 0.2:
        return 1
    elif clarity_score <= 0.35:
        return 2
    elif clarity_score <= 0.5:
        return 3
    elif clarity_score <= 0.7:
        return 4
    else:
        return 5
    
def assess_organization(essay_text):

    sentences = [sent.text for sent in nlp(essay_text).sents]
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    similarities = cosine_similarity(sentence_vectors[:-1], sentence_vectors[1:])
    organization_score = round((similarities.mean() * 4) + 1)
    organization_score = max(1, min(5, 6 - organization_score)) 
    return organization_score

def assess_organization_level(organization_score):
    inverted_organization_score = 6 - organization_score

    if inverted_organization_score >= 5:
        return 5 
    elif inverted_organization_score >= 4:
        return 4 
    elif inverted_organization_score >= 3:
        return 3  
    elif inverted_organization_score >= 2:
        return 2 
    else:
        return 1  


def assess_creativity_and_originality(essay_text):
    doc = nlp(essay_text)
    unique_lemmas = set(token.lemma_ for token in doc)
    diversity_score = len(unique_lemmas) / len(list(doc))
    unique_words = set(token.text for token in doc if token.is_alpha)
    uncommon_words = [word for word in unique_words if word not in nlp.vocab]
    uncommon_words_score = len(uncommon_words) / len(list(unique_words)) if len(unique_words) > 0 else 0
    sentence_structures = set((token.dep_, token.head.dep_) for token in doc if token.dep_ != "punct")
    sentence_structure_score = len(sentence_structures) / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0
    creativity_score = 0.5 * diversity_score + 0.5 * sentence_structure_score
    originality_score = 0.5 * uncommon_words_score + 0.5 * sentence_structure_score
    overall_score = 0.4 * creativity_score + 0.4 * originality_score + 0.2 * sentence_structure_score
    scaled_score = max(1, min(5, round(overall_score * 4) + 1))

    return scaled_score

import spacy

def assess_grammar(essay_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(essay_text)


    grammar_score = 5  

    cohesive_conjunctions = ["and", "but", "however", "therefore", "moreover", "furthermore", "nevertheless",
                             "nonetheless", "consequently", "thus", "in addition", "in contrast", "on the other hand",
                             "meanwhile", "subsequently", "as a result", "likewise", "simultaneously", "indeed", "hence",
                             "in conclusion", "for example", "for instance"]
    total_conjunctions = sum(1 for token in doc if token.text.lower() in cohesive_conjunctions)
    total_tokens = len(doc)
    cohesion_ratio = total_conjunctions / total_tokens

    if cohesion_ratio < 0.05:
        grammar_score -= 1  

    mapped_score = max(1, min(grammar_score, 5))

    return mapped_score

def criteria1(essay_text):
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(essay_text)

    criteria_weights = {
        'grammar': 0.25,
        'evidence': 0.25,
        'focus_on_details': 0.25,
        'structure_organization': 0.25,
    }

    scores = {key: round(sum(token.similarity(nlp(key)) for token in doc) / len(doc) * 4 + 1) for key in criteria_weights}

    total_score = round(sum(scores[key] * criteria_weights[key] for key in scores))

    return {'scores': scores, 'total_score': total_score}
    
def assess_essay(essay_text, selected_criteria):
    assessment_results = {}
    docx_file = request.files['essay']
    question = request.form['additionalQuestion1']


    essay_text = process_docx(docx_file)

    tokens = word_tokenize(essay_text)
    tagged_tokens = pos_tag(tokens)


    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

    total_score = 0

    for criterion in selected_criteria:
        if criterion == "clarity":
            score = assess_clarity(essay_text)
        elif criterion == "Organization":
            score = assess_organization(essay_text)
        elif criterion.lower() == 'creativity':
            score = assess_creativity_and_originality(essay_text)
        elif criterion == 'Relevance':
            relevance_words = [
                request.form['word1'].lower(),
                request.form['word2'].lower(),
                request.form['word3'].lower(),
                request.form['word4'].lower(),
                request.form['word5'].lower()
            ]
            score = assess_relevance(essay_text, question, relevance_words)
        elif criterion == 'Focus and Details':
            score = assess_focus_and_details(essay_text, question)
        elif criterion == 'Grammar':
            score = assess_grammar(essay_text)

        total_score += score
        assessment_results[criterion] = score

    overall_average = total_score / len(selected_criteria)
    assessment_results["Overall Average"] = min(5, round(overall_average, 2))

    return assessment_results





@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    user_folders = Folder.query.filter_by(user_id=current_user.id).all()
    question = request.form.get('additionalQuestion1', '')
    student_number = request.form.get('studentNumber', '')
    uploaded_file_name = request.form.get('uploadedfilename', '')


    if 'essay' in request.files:
        uploaded_file = request.files['essay']
        uploaded_file_name = uploaded_file.filename

    return render_template('home.html', folders=user_folders, user=current_user,
                           question=question, student_number=student_number,
                           uploaded_file_name=uploaded_file_name)


@views.route('/create_folder', methods=['POST'])
@login_required
def create_folder():
        folder_name = request.form.get('folder_name')
    
        user_id = current_user.id

        existing_folder = Folder.query.filter_by(name=folder_name, user_id=user_id).first()

        if existing_folder:
                flash('Folder with the same name already exists.', 'error')
        else:
            new_folder = Folder(name=folder_name, user_id=user_id)
            db.session.add(new_folder)
            db.session.commit()
            flash('Folder created successfully!', 'success')

        return redirect(url_for('views.home'))

@views.route('/upload_file/<int:folder_id>', methods=['POST'])
@login_required
def upload_file(folder_id):
    folder = Folder.query.get_or_404(folder_id)

    if 'file' in request.files:
        file = request.files['file']
        
        if file:
            file_data = file.read()

            new_file = File(name=file.filename, data=file_data, folder=folder)
            db.session.add(new_file)
            db.session.commit()

    return redirect(url_for('views.home'))


@views.route('/delete-folder', methods=['POST'])
@login_required
def delete_folder():
    data = request.get_json()
    folder_id = data.get('folderId')
    
    folder = Folder.query.get(folder_id)

    if folder:
        if folder.user_id == current_user.id:  
            db.session.delete(folder)
            db.session.commit()

    return jsonify({}) 

@views.route('/delete-file/<int:folder_id>/<int:file_id>', methods=['POST'])
@login_required
def delete_file(folder_id, file_id):
    folder = Folder.query.get_or_404(folder_id)
    file = File.query.get_or_404(file_id)

    if folder.user_id == current_user.id and file.folder_id == folder.id:
        db.session.delete(file)
        db.session.commit()

    return jsonify({}) 


@views.route('/assess', methods=['POST'])
@login_required
def assess():
    user_folders = Folder.query.filter_by(user_id=current_user.id).all()
    if request.method == "POST":
        essay_file = request.files.get("essay")
        selected_criteria = request.form.getlist("criteria")
        question = request.form.get('additionalQuestion1', '')
        student_number = request.form.get('studentNumber', '')
        uploaded_file_name = request.form.get('uploadedfilename', '')

        if essay_file:
            uploaded_file_name = essay_file.filename

        try:
            if essay_file:
                essay_text = essay_file.read().decode("utf-8", errors="ignore")
                assessment_results = assess_essay(essay_text, selected_criteria)
                return render_template("home.html", folders=user_folders, results=assessment_results, user=current_user,
                                       question=question, student_number=student_number,
                                       uploaded_file_name=uploaded_file_name, system_score=assessment_results["Overall Average"])

            else:
                assessment_results = {}
                return render_template("home.html", folders=user_folders, results=assessment_results, user=current_user,
                                       question=question, student_number=student_number,
                                       uploaded_file_name=uploaded_file_name)

        except Exception as e:
            print(f"Error in assess_essay: {str(e)}")
            flash('Error occurred during assessment.', 'error')
            return redirect(url_for('views.home', folders=user_folders, user=current_user,
                                    question=question, student_number=student_number,
                                    uploaded_file_name=uploaded_file_name))


@views.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html',user=current_user)


@views.route('/result')
@login_required
def result():
    folders = Folder.query.filter_by(user_id=current_user.id).all()
    print(f"Folders: {folders}")
    
    return render_template('result.html', user=current_user, folders=folders)
@views.route('/result-folder/<int:folder_id>', methods=["GET"])
@login_required
def resultFolder(folder_id):

    folder = Folder.query.filter_by(id=folder_id).first()
    files = File.query.order_by(File.system_score.desc()).filter_by(folder_id=folder_id).all()
     
    return render_template('result-folder.html' ,user=current_user, files=files, folder=folder)


@views.route('/saveToFolder', methods=['POST'])
def save_to_folder():
    system_score = float(request.form.get('systemScore', 0.0))
    question = request.form.get('question', '')
    student_number = request.form.get('student_number', '')
    uploaded_file_name = request.form.get('essay', '')
    selected_folder_id = request.form.get('selectedFolderId')
    criteria_results = request.form.getlist('result')
    results = {}

    if not uploaded_file_name:
        uploaded_file_name = ''


    if not question:
        question = ''
    if not student_number:
        student_number = ''

    if not system_score:
        system_score = 0.0

    criteria_results = request.form.getlist('result') or []
    criteria_results_string = ", ".join(criteria_results)

    results = criteria_results_string

    folder = Folder.query.get(selected_folder_id)

    if not folder:
        return render_template('error.html', message='Selected folder not found')

    new_file = File(
        question=question,
        student_number=student_number,
        name=uploaded_file_name,
        system_score=system_score,
        criteria_results=results, 
        folder_id=selected_folder_id
    )

    db.session.add(new_file)
    db.session.commit()

    return redirect(url_for('views.home', message='File saved successfully to the selected folder'))


@views.route('/export_and_download', methods=['POST'])
def export_and_download():
   
    question = request.form.get('question')
    student_number = request.form.get('student_number')
    uploaded_file_name = request.form.get ('essay','') 
    results = request.form.getlist('result')
  
    processed_data = f"Question: {question}\nStudent Number: {student_number}\nUpload File Name: {uploaded_file_name}\n"
    processed_data += "Assessment Results:\n"
    processed_data += '\n'.join([f"- {result}" for result in results])

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, processed_data)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    pdf_output_path = os.path.join(script_directory, 'result.pdf')
    pdf.output(pdf_output_path)

    return send_file(pdf_output_path, as_attachment=True, download_name='result.pdf')