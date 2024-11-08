pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/SrujanaBudiga/TweetVibe-Insights.git'
            }
        }
        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Run Script') {
            steps {
                sh 'python TweetVibe\\ Insights.py'
            }
        }
    }
}
