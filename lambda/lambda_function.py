import json
import boto3

def lambda_handler(event, context):
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    body = json.loads(event['body'])
    headline = body['query']['headline']

    #headline = "New York listed as a major hub for clinical trials on drugs"

    endpoint_name = 'news-headline-classification-endpoint-v1'  # Replace with your actual endpoint name

    payload = json.dumps({"inputs":headline})

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )

    result = json.loads(response['Body'].read().decode())

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
