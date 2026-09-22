Fake News Detection 

A hybrid **Natural Language Processing (NLP)** application that detects fake news by combining DistilBERT-based text classification with domain credibility verification.
The system analyzes the content of a news article, validates its source against whitelist/blacklist domains, and predicts whether the news is **Real** or **Fake** through a simple Flask web interface.

Features

 DistilBERT-based fake news classification
 Domain whitelist & blacklist verification
 News text preprocessing and cleaning
 Real-time prediction through a Flask web application
 Confidence-based classification
 Modular training and inference pipeline

 Tech Stack

 Category         Technologies                           
 Language         Python                                 
 NLP Model        DistilBERT  
 Deep Learning    PyTorch                                
 Web Framework    Flask                                  
 Data Processing  Pandas, NumPy                          
 Dataset          Kaggle Fake & Real News Dataset        

Project Structure

Fake-News-Detection/

     data/
     
     saved_model_distilbert/      
     
     templates - index.html               
     
     app.py                       
     
     comb.py                      
     
     pre_process2.py              
     
     Fake_News_Detection.ipynb    
     
     requirements.txt
     
     README.md


Methodology

1. User enters a news article.
2. The text is cleaned and preprocessed.
3. The news source is validated using whitelist/blacklist domain datasets.
4. DistilBERT tokenizes and encodes the input text.
5. The trained model predicts whether the news is **Real** or **Fake**.
6. The prediction is displayed through the Flask interface.

Dataset

This project utilizes publicly available fake and real news datasets along with domain credibility datasets for whitelist and blacklist verification. 
The preprocessing pipeline generates cleaned and merged datasets used during model training.


 How to Run

 Clone the repository

git clone https://github.com/kneerajk2005/Fake-News-Detection.git
cd Fake-News-Detection


 Install dependencies


pip install -r requirements.txt


Start the application


python app.py


Open the browser and visit:


http://127.0.0.1:5000


Future Improvements

* Multilingual fake news detection
* Explainable AI for prediction reasoning
* Real-time news article verification using live web sources
* Enhanced domain reputation scoring

## Author

Neeraj Kumar

Sasidhar

Integrated M.Tech in Computer Science and Engineering

Cloud Computing • NLP • Machine Learning • Azure
