import yaml
import json
import kfp
from kfp.dsl import PipelineParam, PipelineVolume
from kubernetes.client.models import *

__MODEL_IMAGE = 'gcr.io/innovation-lab-sandbox/bank-marketing-model'
__MODEL_NAME = 'bank-marketing-model'
__MODEL_SERVICE_PORT = 5000


@kfp.dsl.component
def train_component(payload, cortex_local_volume: PipelineVolume):
    op = kfp.dsl.ContainerOp(
        name='model-training',
        image=__MODEL_IMAGE,
        pvolumes={"/model/cortex": cortex_local_volume}
    )

    op.container.set_image_pull_policy('Always')
    op.container.add_env_variable(V1EnvVar('TRAIN', 'True'))
    op.container.add_env_variable(V1EnvVar('PAYLOAD', payload))
    
    # TODO usage of outputs, artifacts, metrics, etc.
    
    return op

def create_model_deployment(model_name, model_image, container_port, cortex_local_pvc):
    labels = {'c12e.service': model_name}
    return V1Deployment(
        api_version='extensions/v1beta1', 
        kind='Deployment', 
        metadata=V1ObjectMeta(name=model_name, labels=labels),
        spec=V1DeploymentSpec(
            replicas=1,
            selector=V1LabelSelector(match_labels=labels),
            template=V1PodTemplateSpec(
                metadata=V1ObjectMeta(labels=labels),
                spec=V1PodSpec(
                    volumes=[V1Volume(name='cortex-local-volume', persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name=cortex_local_pvc))],
                    containers=[V1Container(
                        image=model_image, 
                        image_pull_policy='Always', 
                        name=model_name, 
                        ports=[V1ContainerPort(container_port=container_port)],
                        volume_mounts=[V1VolumeMount(mount_path='/model/cortex', name='cortex-local-volume')])],
                    restart_policy='Always'
                )
            )
        ))

def create_model_service(model_name, port=5000, target_port=5000):
    return f"""
apiVersion: v1
kind: Service
metadata:
  labels:
    c12e.service: {model_name}
  name: {model_name}
spec:
  type: LoadBalancer
  ports:
    - name: "{target_port}"
      port: {port}
      targetPort: {target_port}
  selector:
    c12e.service: {model_name}
    """

@kfp.dsl.pipeline(
  name='Bank Marketing Model',
  description='Pipeline for the Bank Markeing Model example'
)
def pipeline(train_payload: str):
    cortex_local_volume = kfp.dsl.VolumeOp(
        name='cortex-local-volume',
        resource_name='cortex-local-pvc',
        modes=['ReadWriteOnce'],
        size='1Gi'
    )

    train = train_component(train_payload, cortex_local_volume.volume)

    deployment_op = kfp.dsl.ResourceOp(
        name='model-deployment',
        k8s_resource=create_model_deployment(__MODEL_NAME, __MODEL_IMAGE, __MODEL_SERVICE_PORT, cortex_local_volume.outputs['name']),
        action='apply'
    ).after(train)

    serve_op = kfp.dsl.ResourceOp(
        name='model-serving',
        k8s_resource=yaml.load(create_model_service(__MODEL_NAME), Loader=yaml.Loader),
        action='apply'
    ).after(deployment_op)

    # wait for service and get IP
    wait_op = kfp.dsl.ContainerOp(
        name='model-ready',
        image='c12e/kubectl-get-svc-ip',
        arguments=['--name', __MODEL_NAME],
        file_outputs={
            'ip': '/tmp/ip.txt'
        },
        output_artifact_paths={
            'ip': '/tmp/ip.txt'
        }
    ).after(serve_op)
    wait_op.container.set_image_pull_policy('Always')

    # kick off Certifai scan

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pipeline, './bank-marketing-model-pipeline.tar.gz')
