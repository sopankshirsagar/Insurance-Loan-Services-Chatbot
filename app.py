from flask import Flask, request, jsonify, render_template
import pickle
from langchain.agents import create_openai_tools_agent, AgentExecutor
import os
print(os.getcwd())

# Load the pickle files
with open(r"C:\Users\sopan\Desktop\LIC Loan ChatBot\ChatBot\chatbot\vectordb.pkl", "rb") as vectordb_file:
    vectordb = pickle.load(vectordb_file)

with open(r"C:\Users\sopan\Desktop\LIC Loan ChatBot\ChatBot\chatbot\tools.pkl", "rb") as tools_file:
    tools = pickle.load(tools_file)

with open(r"C:\Users\sopan\Desktop\LIC Loan ChatBot\ChatBot\chatbot\llm_model.pkl", "rb") as llm_file:
    llm = pickle.load(llm_file)

with open(r"C:\Users\sopan\Desktop\LIC Loan ChatBot\ChatBot\chatbot\prompt.pkl", "rb") as prompt_file:
    prompt = pickle.load(prompt_file)


retriver = vectordb.as_retriever()
from langchain.tools.retriever import create_retriever_tool

retrival_tool = create_retriever_tool(retriver, 'insurance_tool', "this is tool for Insurance Chatbot")
tools = [retrival_tool]
# Recreate the agent and agent executor
agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "Query is required"}), 400
    
    # Use the agent executor to generate a response
    try:
        response = agent_executor.invoke({'context': retriver, 'input': user_query})
        return jsonify({"response": response['output']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


