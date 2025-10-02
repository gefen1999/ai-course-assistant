# DASHA: Data-driven Academic Smart Helper Assistant
![DASHA](https://i.ibb.co/NdZFwskX/HEADER.png)
Dasha is an intelligent assistant designed to help students interact with course materials more efficiently.
By leveraging advanced language models and semantic search, Dasha enables users to ask questions about course content and receive accurate, context-aware answers. The motivation behind Dasha is to streamline the learning process, making it easier to find relevant information, clarify concepts, and enhance the overall educational experience.

## Setup
1. **Clone the repository**

   ```sh
   git clone https://github.com/gefen1999/ai-course-assistant.git
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
3. **Create the Pinecone index**
   Open and run the Jupyter notebook to create the Pinecone index:
   ```sh
   jupyter notebook indexing_ENG.ipynb
   ```
   or, if using JupyterLab:
   ```sh
   jupyter lab indexing_ENG.ipynb
   ```

   Run all cells in the notebook to process your data and generate the necessary index files.

3. **Run the Streamlit app**

   Start the Streamlit interface:
   ```sh
   pip install streamlit
   streamlit run streamlit_app.py --server.port 8888
   ```
   Follow the generated link to see the app in your browser.

## Notes
- Place your own API keys in src/api_keys.json - see the file for reference.
- Make sure to run `indexing_ENG.py` whenever your data changes.
- The application relies on OpenAI's GPT-4.1 model. Ensure all required API credentials (deployment, version, etc.) are properly configured in your environment variables.

## Usage
1. **Choose a course on the sidebar menu.**

    You may also choose "Search in all documents", but this option may yield less accurate results.
2. **Example general queries you can ask:**
- "Who teaches the course and what is the format of the exam?"
- "Which additional literature is recommended for this course?"
- "List the main topics covered in the syllabus"

**Course-specific queries: **

    Algebraic Methods: 
    - "Explain SVD and it's usages in Data Science."
    - "Provide a brief explanation of Principal Component Analysis (PCA)."
  
    Introduction to AI:
    - "What are common applications of AI discussed in the course?"
    - "Explain shortly the difference between the types of Reinforcement Learning"

3. **Optionally, upload your own course materials (PDF or text).**