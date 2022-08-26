from flask import *
import json, time
app= Flask(__name__)

@app.route('/', methods=["GET"])
def home_page():
    dataset = dict(page='Home', message=f'Succesfully loaded Home page', timestamp=time.time())

    json_dump=json.dumps(dataset)
    return json_dump

@app.route('/user/', methods=["GET"])
def request_page():
    user_query=str(request.args.get('user'))
    dataset=dict(page='Home',message=f'Succesfully loaded {user_query}', timestamp=time.time())
    json_dump=json.dumps(dataset)
    return json_dump

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(port=7777)
