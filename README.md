# QuizMaker_RAG

## Architecture

```
|- /templates
|   - index.html: The main UI for PDF uploading
|   - quiz.html: The main UI for quiz
|   - results.html: The main UI for the quiz results
|- application.py: The main Flask application
|- utils.py: A helper function containing the key NLP and Text Generation functions
|- requirements.txt: A list of packages needed to install
```

## Process
- The User uploads a PDF
- The PDF is saved, and the text is extracted and chunks
- The chunks are stored into Pinecone for storage
- Some chunks will be randomly extracted to generate the context for questions in a specific format
- The QA is parsed into MCQ's and stored in the current session
- The questions are then rendered into a quiz format
- The user selects and submits the answers
- The system calculates and displays the results, with an explanation

## Performance and Scalability
Overall, this is just a demo of the potential app. Through elastic beanstalk, it performs extremely well. There are a couple of performance and scability issues that will need to be resolved in the future.
- LLM's are prone to hallucinations. The questions may not generate properly.
- Currently, the elastic beanstalk deployment only allows for PDF submissions of 1MB or less. There is no limit locally
- There is an error, spefically on the first MCQ question. 

## Testing
To test, please use the added pdf to the repo. Alternatively, feel free to use your own pdf file, but make sure the size is less than 1 MB.

## Deployment
To deploy locally:
- Clone the repo
- Install all packages in requirements.py
- Fill the .env file with your information
OPENAI_API_KEY = 'your-key'

PINECONE_API_KEY = 'your-key'

PINECONE_INDEX_HOST = 'your-host'
- run application.py, and open the Flask application

Alternatively, you can access it through this link: http://quizmaker-env.eba-3pdfrnjm.us-east-2.elasticbeanstalk.com/ 
* Note that there may be errors due to openai rate limits *

## Security and Responsibility
- While the pdf is inserted into the vector store, it will clear the contents after the questions are generated
- Do not put any personal information and sensitive data. The generation is using OpenAI's backend.
- There is a possibility for incorrect answers 

## Future Development
- Finish the retry mechanism for the code
- Provide UI for the user specify the number of questions and the chunking size
