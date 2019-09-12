pipeline {
  agent any
  stages {
    stage('Build') {
      steps {
        sh '''wget -O s2i.tar.gz https://github.com/openshift/source-to-image/releases/download/v1.1.14/source-to-image-v1.1.14-874754de-linux-amd64.tar.gz
tar zxf s2i.tar.gz
mv s2i /usr/local/bin'''
        sh 'make build'
      }
    }
  }
}