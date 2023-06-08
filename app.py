import os
import datetime
import traceback
from dotenv import load_dotenv
import json
load_dotenv()

from flask import Flask, request, jsonify
from flask_restx import Resource, Api, fields

from chatbot import *

def InitModel():
    print("Init models")
    return

InitModel()

print("Server initializing...")
app = Flask(__name__)
api = Api(app, default="QnA Chat Service", default_label="QnA machine learning for chat service API", doc = "/docs/")

# Payload API
question_payload = api.model('Question', {
    'text': fields.String,
})

# Route Define
@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

@api.route('/')
class RootDir(Resource):
    def get(self):
        return {'hello': 'root'}

@api.route("/question")
@api.doc(responses={500: 'Internal error'})
class Question(Resource):
    @api.doc(body=question_payload)
    def post(self):
        data = request.get_data()
        query = json.loads(data)
        content = query.get('text') #input your questions on Body of endpoint (String)
        print(content)

        try:
            
            ## do something
            
            responses = responseQuery(content)

            result = {
                "code": 0,
                "request": content,
                "result": {
                "text": ""
                },
                "time_usage": 0,
                "meta": []
            }
            result["result"]['text'] = responses['answer']
            return result
        except Exception as e:
            traceback.print_exc()
            print(str(e))
            return {"code":500, "message": str(e)}

if __name__ == '__main__':
    app.run(port=8001, debug=True)

