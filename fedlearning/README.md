F.L.A.R.E. - made by me and my business partner [David](https://github.com/davidd234).
F.L.A.R.E. is a privacy-first cybersecurity infrastructure designed for Internet Service Providers. It deploys lightweight AI models to edge routers to detect network anomalies locally, transmitting only mathematical model updates to a central server using Federated Learning strategies. This ensures fleet-wide immunity against botnets while remaining GDPR compliant by keeping raw user data on-device.

Quick Start

Initialize data: python preprocess_ton_iot.py

Start Server: python federated_server.py

Start a few Clients: ex. CLIENT_ID=0 python federated_client.py

View Dashboard: python app.py (http://localhost:5000)

Here you can also find a presentation of the project:
https://gamma.app/docs/FEDERATED-LEARNING-ANOMALY-RESPONSE-ENGINE-o3zwkb7gsmfffvw
