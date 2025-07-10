# News Classification using DistilBERT

An end-to-end, serverless NLP pipeline that classifies news headlines using a fine-tuned **DistilBERT** model. This project demonstrates how to train, deploy, and interact with a machine learning model entirely on **AWS** using **SageMaker**, **Lambda**, and **API Gateway**, with a modern **Streamlit frontend** for real-time inference.

---

## Problem Statement

The goal is to automatically classify news headlines into one of four categories:

- **Business**
- **Science**
- **Entertainment**
- **Health**

This mirrors a real-world application where headlines must be routed to the appropriate editorial team or content feed.

**Dataset**: [News Aggregator Dataset](https://archive.ics.uci.edu/dataset/359/news+aggregator) from UCI Machine Learning Repository

---

## Overview

- Fine-tuned the pre-trained `distilbert-base-uncased` model using the UCI News Aggregator dataset
- Added a **custom linear classification head** on top of DistilBERT to better learn headline representations, increasing capacity with over **600,000 additional trainable weights**
- Introduced **Dropout(30%)** to prevent overfitting during training
- Final model had over **50M trainable parameters**, enabling better generalisation for nuanced classification
- Trained and deployed using **AWS SageMaker Training Jobs** and **SageMaker Inference Endpoints**
- Served via a **serverless API** using **AWS Lambda + API Gateway**
- Built a clean, interactive **Streamlit frontend** to allow real-time headline classification with confidence scores

---

## Tech Stack

| Layer       | Tool / Service                       |
|-------------|--------------------------------------|
| LLM Model   | DistilBERT (`distilbert-base-uncased`)        |
| Training    | AWS SageMaker Training Job         |
| Inference   | AWS SageMaker Endpoint              |
| API         | AWS Lambda + API Gateway           |
| Frontend    | Streamlit                         |
| Storage    | S3 Buckets                         |
| Deployment  | Streamlit Cloud / Local            |
| Infra Mgmt  | IAM Roles, Boto3, Python SDK       |
| API Testing  | POSTMAN       |

---

## Project Structure

```
distilbert-news-classification/
│
├── notebooks/
│   ├── 01_EDA_MultiClassTextClassification.ipynb
│   ├── 02_TrainingNotebook.ipynb
│   └── 03_DeploymentNotebook.ipynb
│
├── scripts/
│   ├── script.py               # Training script
│   ├── inference.py            # Inference handler
│   ├── load_test.py            # Load testing on EC2 instances
│
├── app/
│   └── app.py                  # Streamlit frontend
│
├── lambda/
│   └── lambda_function.py      # Lambda calling SageMaker endpoint
│
├── data/
│   └── news+aggregator/
│       └── newsCorpora.csv    # UCI News Aggregator dataset (headlines + labels)
│
├── assets/
│   └── live-demo.mp4
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Sample Prediction

> Input Headline:  
`"NASA launches telescope to study deep space radiation"`

> Output:
```
Predicted Category: Science

Class Probabilities:
* Business: 3.21%
* Science: 91.42%
* Entertainment: 2.05%
* Health: 3.32%
```

---

## 🛠️ How to Deploy on AWS

### 1. Data Preparation & Upload

```bash
# Upload your dataset to S3
aws s3 cp your_dataset.csv s3://your-bucket-name/data/news-headlines/
```

### 2. SageMaker Training Job

- **Create SageMaker Notebook Instance** or use **SageMaker Studio**
- **Upload Training Script** (`script.py`) to your SageMaker environment
- **Configure Training Job**:
   ```python
   from sagemaker.huggingface import HuggingFace
   
   # Define the HuggingFace estimator
   huggingface_estimator = HuggingFace(
       entry_point='script.py',
       source_dir='scripts',
       instance_type='ml.p3.2xlarge',
       instance_count=1,
       role=sagemaker_role,
       transformers_version='4.21',
       pytorch_version='1.12',
       py_version='py39'
   )
   ```
- **Start Training Job** and monitor progress in **SageMaker Console**
- **Check Training Logs** in **AWS CloudWatch** under `/aws/sagemaker/TrainingJobs/`

### 3. Model Deployment

- **Create SageMaker Endpoint** using the trained model:
   ```python
   # Deploy model to endpoint
   predictor = huggingface_estimator.deploy(
       initial_instance_count=1,
       instance_type='ml.m5.large',
       endpoint_name='distilbert-news-classifier'
   )
   ```
- **Upload Inference Script** (`inference.py`) for custom prediction logic
- **Test Endpoint** directly in SageMaker before connecting to Lambda

### 4. Lambda Function Setup

- **Create Lambda Function** in AWS Console
- **Configure Runtime**: Python 3.9
- **Set Execution Role** with SageMaker invoke permissions:
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "sagemaker:InvokeEndpoint"
               ],
               "Resource": "arn:aws:sagemaker:region:account:endpoint/distilbert-news-classifier"
           }
       ]
   }
   ```
- **Deploy Lambda Code** (`lambda_function.py`)

### 5. API Gateway Configuration

- **Create REST API** in API Gateway Console
- **Create POST Method** for `/predict` resource
- **Configure Integration** with Lambda function
- **Enable CORS** for cross-origin requests
- **Deploy API** to a stage (e.g., `dev`, `prod`)
- **Get API Endpoint URL**: `https://your-api-id.execute-api.region.amazonaws.com/dev/predict`

### 6. Testing with Postman

```json
POST https://your-api-id.execute-api.region.amazonaws.com/dev/predict

Headers:
Content-Type: application/json

Body:
{
    "query": {
        "headline": "NASA launches telescope to study deep space radiation"
    }
    
}
```

### 7. Monitoring & Logging

- **CloudWatch Logs**: Monitor Lambda function logs
- **SageMaker Logs**: Check endpoint invocation logs
- **API Gateway Logs**: Track API usage and errors

---

## Future Work

* Support for multilingual classification with `xlm-roberta`

---

## Author

**Sanyog Chavhan**  
Data Scientist | Specialised in Building Scalable AI Applications  
[LinkedIn](https://www.linkedin.com/in/sanyog-chavhan/) | [Portfolio](https://www.datascienceportfol.io/sanyog)

---
