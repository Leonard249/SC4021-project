import json
import os
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder='.')
DATA_FILE = os.path.join(os.path.dirname(__file__), 'sample75.json')

def load_data():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/data')
def get_data():
    data = load_data()
    # Return only what the UI needs (not the huge POS_Tags etc.)
    result = []
    for i, item in enumerate(data):
        entry = {
            'index': i,
            'ID': item.get('ID'),
            'Source': item.get('Source'),
            'Type': item.get('Type'),
            'Author': item.get('Author'),
            'Title': item.get('Title'),
            'Text': item.get('Text'),
            'Date': item.get('Date'),
            'Overall_Document_Polarity': item.get('Overall_Document_Polarity'),
            'user_check': item.get('user_check'),
            'Comments': []
        }
        for j, c in enumerate(item.get('Comments', [])):
            entry['Comments'].append({
                'comment_index': j,
                'comment_id': c.get('comment_id'),
                'Author': c.get('Author'),
                'Text': c.get('Text'),
                'Date': c.get('Date'),
                'Overall_Document_Polarity': c.get('Overall_Document_Polarity'),
                'user_check': c.get('user_check'),
            })
        result.append(entry)
    return jsonify(result)

@app.route('/api/save', methods=['POST'])
def save():
    payload = request.get_json()
    idx = payload['index']
    user_check = payload['user_check']
    comment_checks = payload.get('comment_checks', {})  # {comment_index: user_check}

    data = load_data()
    data[idx]['user_check'] = user_check

    for ci_str, ck in comment_checks.items():
        ci = int(ci_str)
        if ci < len(data[idx].get('Comments', [])):
            data[idx]['Comments'][ci]['user_check'] = ck

    save_data(data)
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("Labeling tool running at http://localhost:5050")
    app.run(port=5050, debug=False)
