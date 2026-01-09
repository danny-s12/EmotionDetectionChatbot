from flask import Flask, render_template, request, jsonify
import os
from main import run as run_models
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd
app = Flask(__name__)
# ==============================
# Load Label Encoder (fit on train set)
# ==============================
train_df = pd.read_csv("small_train.csv")   # use your reduced train data
le = LabelEncoder()
le.fit(train_df["Emotion"])

# ==============================
# Reload Model & Tokenizer
# ==============================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        out = self.dropout(cls_output)
        return self.fc(out)

model = BertClassifier(num_labels=len(le.classes_))
model.load_state_dict(torch.load("bert_emotion_classifier.pth", map_location=device))
model.to(device)
model.eval()
# ==============================
# Prediction function
# ==============================
def predict_emotion(text):
    enc = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        pred = torch.argmax(outputs, dim=1).item()
    return le.inverse_transform([pred])[0]

# ==============================
# Chatbot responses
# ==============================
responses = {
    "joy": "😊 That's wonderful! Tell me more about what made you happy.",
    "sadness": "😢 I'm sorry to hear that. Do you want to talk about it?",
    "anger": "😡 I sense some frustration. Take a deep breath—what’s bothering you?",
    "fear": "😨 That sounds scary. I'm here to listen.",
    "neutral": "😐 Thanks for sharing. Tell me more."
}

# Folders
recordings_dir = os.path.join(app.root_path, 'videos')
os.makedirs(recordings_dir, exist_ok=True)  # ensure videos folder exists
q = 0  # tracks question number


def make_data():
    data = []
    print("DEBUG: Scanning videos directory...")
    for file in os.listdir('videos'):
        if not (file.endswith(".mp4") or file.endswith(".webm")):
            print(f"DEBUG: Skipping non-video file: {file}")
            continue

        print("DEBUG: Found video file:", file)
        try:
            results = run_models(f'videos/{file}')
            print("DEBUG: Results from run_models:", results)

            video = str(results[0])
            audio = str(results[1])
            final = str(results[2])

            data.append({
                'file': file,
                'video': video,
                'audio': audio,
                'final': final
            })
        except Exception as e:
            print(f"ERROR processing {file}: {e}")
    return data
@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def get_response():
    user_message = request.json["message"]
    emotion = predict_emotion(user_message)
    bot_reply = responses.get(emotion, "🤖 Hmm, I'm not sure how to respond.")
    return jsonify({"emotion": emotion, "response": bot_reply})


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/question/<int:question_no>')
def question(question_no):
    questions = [
        {"question_no": 1, "question": "What is your opinion on video games?"},
        {"question_no": 2, "question": "What are your hobbies?"},
    ]

    if question_no == len(questions) + 1:
        return render_template('fin.html')

    if question_no < 1 or question_no > len(questions):
        return "Invalid question number", 404

    selected_question = questions[question_no - 1]
    global q
    q = question_no
    return render_template('question.html',
                           question_no=selected_question["question_no"],
                           question=selected_question["question"])


@app.route('/record')
def record():
    return render_template('record.html')


@app.route('/save-video', methods=['POST'])
def save_video():
    if 'video' in request.files:
        video = request.files['video']
        if video:
            # save as .webm to match Chrome MediaRecorder output
            video_path = os.path.join(recordings_dir, f'recording{q}.webm')
            video.save(video_path)
            print(f"DEBUG: Saved video -> {video_path}")
            return 'Video saved successfully', 200
    return 'Error saving video', 400


@app.route('/get_data')
def get_data():
    data = make_data()
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # disable reloader so it won’t restart mid-request
