def lambda_handler(event, context):
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/html"
        },
        "body": "<h1>Hello World!</h1>"
    }