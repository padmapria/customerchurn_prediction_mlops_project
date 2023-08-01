import docker
import subprocess

def build_docker_image():
    client = docker.from_env()
    client.images.build(path="./", tag="my-model-image")

def deploy_docker_image():
    client = docker.from_env()
    client.containers.run("my-model-image", detach=True, ports={"5000": "5000"})

def setup_monitoring():
    # Use subprocess to run commands to configure Prometheus and Grafana
    # For example:
    # subprocess.run(["docker-compose", "up", "-d", "prometheus", "grafana"])

# Call the functions to build, deploy, and setup monitoring
build_docker_image()
deploy_docker_image()
setup_monitoring()
