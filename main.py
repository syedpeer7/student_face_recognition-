from flask import Flask, render_template, jsonify
import os
import json

app = Flask(__name__)

DATA_FILE = "static/student_distraction_data.json"


@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/data')
def get_data():
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "Data file not found"})

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    total_students = len(data)
    distracted_students = sum(1 for d in data.values() if d["talking"] > 0 or d["turning"] > 0)

    # Calculate focus level: (1 - total distractions/ (total_students * 10)) * 100
    focus_levels = [(1 - ((d["talking"] + d["turning"]) / 10)) * 100 for d in data.values()]
    average_focus = sum(focus_levels) / len(focus_levels) if focus_levels else 100

    response_data = {
        "total_students": total_students,
        "distracted_students": distracted_students,
        "average_focus": average_focus,
        "student_records": {}
    }

    for student_id, details in data.items():
        response_data["student_records"][student_id] = {
            "image": details["image"],
            "distractions": details["talking"] + details["turning"]
        }

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)
