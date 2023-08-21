## Goal of the project
- The objective of this MLOps project is to develop predictive models using the customer data of a Bank to identify factors contributing to customer churn in the banking industry.Specifically, the project focuses on a classification task, where the goal is to classify customers as 'exited' or 'not exited' based on various features and behavioral patterns.

- By integrating machine learning and operational processes, the project aims to provide the bank with real-time insights, enabling proactive actions to retain customers on the brink of leaving.


### Techstack used for the project
-----------------------------------
python 3.11   
Configuration management - hydra   
Centralized logging in a log file - logging    
Experiment Tracking, Model Registry - MLFlow    
Orchestration - Prefect    
Monitoring - Evidently    
Unit/Integration Testing - pytest   
Deployment - Docker    
CI/CD: Github actions   
Cloud - AWS    


### DataSet Used 
-----------------
https://www.kaggle.com/code/arhitbosetagore/advanced-data-visualization/input?select=Customer-Churn-Records.csv

The dataset contains the following attributes   

RowNumber: Sequential record identifier with no impact on outcomes.   
CustomerId: Randomly assigned identifier unrelated to churn.   
Surname: Customer's last name, irrelevant for churn prediction.   
CreditScore: Affects churn; higher scores mean lower likelihood to leave.   
Geography: Location's potential influence on churn.   
Gender: Possible role in determining churn propensity.   
Age: Key factor; older customers less likely to churn.   
Tenure: Reflects loyalty; longer tenure, lower churn likelihood.   
Balance: Strong indicator; higher balances, lower churn probability.  
NumOfProducts: Count of products bought, affects churn.   
HasCrCard: Presence of credit card lowers churn chances.   
IsActiveMember: Active customers tend to stay, lowering churn.  
EstimatedSalary: Impact on churn; higher salaries, lower likelihood to leave.   
Exited: Main outcome; indicates if customer left the bank.   
Complain: Complaints might relate to higher churn.   
Satisfaction Score: Feedback impacting churn potential.   
Card Type: Card held may influence churn decision.   
Points Earned: Reflects engagement; impacts churn chance.   

#### Detailed Workflow and Steps in the project
------------------------------------------------
The entire pipeline can be executed by installing the dependencies from requirements.txt and running two Python files located in the src folder, with Docker running on the machine.
1) training_flow.py  - Data preparation to registering the best model to model registry   
2) runtest_deploy.py - orchestrates the crucial process of executing tests and managing the deployment pipeline using Docker.   

##### The steps executed in training_flow.py are below    
1) Flow Management: Utilized Prefect Flow to organize tasks, simplifying the workflow for machine learning operations.   
2) Data Handling: Raw data is located at the 'data' folder. Loaded and prepared the data, establishing the foundation for subsequent analytical tasks. This involved creating different subsets of the dataset to serve various purposes within the project. Specifically:      
	- Model Training and Testing Data: A subset of the original dataset was carefully selected and organized into a file named 'train_cv_test_data_raw.csv'. This subset was used for model training and testing purposes, enabling effective model development and performance evaluation.
 	- Unit and Integration Testing Data: To ensure the accuracy and functionality of individual components, a separate subset of data was extracted and saved as 'unit_integration_test_data_raw.csv'. This dataset played a crucial role in validating small-scale processes through unit and integration testing.
 	- Unseen Data for Model Deployment Testing: For comprehensive testing of the deployed model, a separate set of entirely unseen data was curated and stored as 'batch_test_data_raw.csv'. This data was exclusively used to assess the performance of the deployed model within the Docker environment, ensuring data security and minimizing overfitting risks.    
3) Robust Logging: Implemented a robust logging mechanism using a singleton logger, ensuring transparent error tracking and management throughout the process.
4) EDA/Preprocessing: To get deeper insights about the data. Encoding the categorical data and scaling the numerical data and shuffle splitting. Encoders are stored at the 'data' folder
5) Model Training: Developed core machine learning components, including baseline Logistic Regression and Random Forest models. Model artifacts are stored at the folder 'model_artifacts'
6) Hyperparameter Tuning: Conducted grid search to fine-tune Random Forest model performance through optimized parameter selection.
7) Model Assessment: Evaluated models using diverse metrics like accuracy, precision, recall, and F1-score on test data. Applied the Random Forest model for predictions.
8) MLflow Integration: Integrated MLflow to effectively manage model tracking, enabling tracking of results through specified tracking URI and experiment name.
9) Unseen Data Handling: Expanded the workflow to predict outcomes for new batch data. Monitored model performance changes over time to detect any drift.
10) Exception Handling: Employed structured exception handling to ensure comprehensive error logging and systematic error management.
11) Script Orchestration: Managed the entire workflow using the main script, capturing and logging the workflow's completion status along with any potential errors.
	
##### The steps executed in runtest_deploy.py are below
1) Running Tests: Executes both unit and integration tests on the dataset located at the data folder 'unit_integration_test_data_raw.csv'.    
2) Unit tests validate small components, like data preprocessing.
3) Integration tests : Ensures the entire Prefect Flow operates smoothly, testing the integrated workflow.   
4) Checking Test Results: Collects results from both test types and determines the overall success of the tests.
5) Conditional Deployment: Proceeds to build and deploy a Docker image if both test types pass.Stops deployment if any of the tests fail, ensuring potential issues are avoided.
6) Docker Image Creation: Constructs a Docker image with the name "myproject-image". The image encapsulates the project's environment, relavent configuration and datas.
7) Final Outcome: Deploys the project if tests and image creation were successful. Safeguards project integrity by preventing deployment in case of test failures.
8) The deployed project, runs end to end prediction over the raw data located at the data folder 'batch_test_data_raw.csv' 
9) By encapsulating these steps, the runtest_deploy.py script assures thorough testing and secure deployment, contributing to the project's reliability and stability.


### How to run the project in Windows Machine or AWS EC2 that have python 3.11, docker and github installed
------------------------------------------------------------------------------------------------------------
1) Download the code from github
Open the command prompt and clone the project's GitHub repository using the following command:    
git clone https://github.com/padmapria/customerchurn_prediction_mlops_project.git

2. Navigate to Project Folder from command prompt or Terminal:   
cd customerchurn_prediction_mlops_project   

3. Install the required dependencies from requirements.txt   
pip install -r requirements.txt   

4. Start the MLFlow and prefect server    
Open a new command prompt window and start the MLFlow server and Prefect server using the following commands:    

start "MLflow Server" mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0  

prefect server start & prefect cloud login -k pnu_BSMcUoknw1  
	
5. Launch docker.Ensure Docker Desktop is running if you're on Windows.  

6. Open a new command prompt window and execute the following commands to run the project scripts:  
For Training Flow: 
   python src/training_flow.py   
   
For Test and Deployment Flow:  
   python src/runtest_deploy.py   

7. Run Docker Container:   
After running the above scripts at runtest_deploy.py, a Docker image named 'myproject-image' should have been created. Run the image as a Docker container:   
docker run -d myproject-image   

8. To view logs of the docker container. Run the following command     
Find the Container ID or Name  by running the following command from command prompt:    
docker ps   

Access Container Logs:    
Once you have the container ID or name, you can use the docker logs command to access the logs. For ex:    
docker logs <container_id_or_name>   


### Deployment with github actions CI/CD
-----------------------------------------
I have created a workflow located at the folder .github/workflows that triggers following actions one by one   

1. Code Setup and Environment Configuration:   
The workflow triggers on push or pull request events to the master branch.   
It runs on an Ubuntu environment and sets up Python 3.11 for execution.   
Dependencies are installed using pip based on the requirements.txt file.   

2. MLflow and Prefect Server Start:   
The MLflow server is started with specific settings, including the backend store URI and artifact root.   
If a Prefect server is already running, it is stopped to prevent conflicts.   
A new Prefect server is started in the background with logs redirected to a file.    

3. Prefect Cloud Login:   
The workflow logs into the Prefect Cloud using a secret API key and workspace name.   

4. Model Training and Deployment:   
The workflow executes the training_flow.py script to train the model, including data preprocessing, hyperparameter tuning, and model registration in MLflow.   
Docker service is started to enable containerization.  
 
5. Testing and Docker Deployment:   
Unit and integration tests are run using the runtest_deploy.py script to ensure code correctness.   
If tests pass, the application is deployed into a Docker container.   

6. Docker Container Execution:   
A Docker container is launched using the built image named myproject-image.   
By following these steps, the workflow automates the process of setting up the environment, running tests, and deploying the project using Docker, enhancing the development process and ensuring code quality   .

### Things to improve in the project
1) better exception Handling  
2) unit Testing for all the steps   
3) To include Iaac like Terraform to deploy in AWS, instead of manual AWS deployment
