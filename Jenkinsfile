pipeline {
  agent any
  stages {
    stage('Train') {
      steps {
        sh 'make train'
      }
    }
    stage('Test') {
      steps {
        sh 'make test'
      }
    }
    stage('Build Image') {
      steps {
        sh 'make build'
      }
    }
  }
}
