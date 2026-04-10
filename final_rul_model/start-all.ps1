Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   STARTING FULL ML STREAMING STACK     " -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan

# Check Available Memory
$availableMem = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024
Write-Host "Available RAM: $([math]::Round($availableMem, 2)) MB" -ForegroundColor Yellow

if ($availableMem -lt 4000) {
    Write-Warning "Low Available RAM detected (< 4GB). The stack might be unstable."
    Write-Warning "Recommended: Close other heavy applications (Chrome, IDs, etc.)"
}

# --------------------------------------------------
# 1 Start Minikube
# --------------------------------------------------
$minikubeStatus = minikube status -o json 2>$null | ConvertFrom-Json

if ($null -eq $minikubeStatus -or $minikubeStatus.Host -ne "Running") {
    Write-Host "Starting Minikube (3GB RAM / 3 CPUs)..."
    minikube start --memory=3072 --cpus=3 --wait=all
}
else {
    Write-Host "Minikube already running."
}

# --------------------------------------------------
# 2 Ensure Namespaces
# --------------------------------------------------
kubectl create namespace kafka --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace elastic --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace numaflow-system --dry-run=client -o yaml | kubectl apply -f -

# --------------------------------------------------
# 3 Install Numaflow (if not installed)
# --------------------------------------------------
if (-not (kubectl get pods -n numaflow-system 2>$null)) {
    Write-Host "Installing Numaflow..."
    kubectl apply -f https://github.com/numaproj/numaflow/releases/latest/download/install.yaml
    Start-Sleep -Seconds 25
}

# --------------------------------------------------
# 4 Switch Docker to Minikube
# --------------------------------------------------
minikube docker-env --shell powershell | Invoke-Expression

# --------------------------------------------------
# 5 Build ML Runtime Image
# --------------------------------------------------
Write-Host "Building ML Runtime Docker image..."
docker build -t final-rul-model:latest .

# --------------------------------------------------
# 6 Deploy Infrastructure
# --------------------------------------------------
kubectl apply -f .\deploy\zookeeper.yaml
kubectl apply -f .\deploy\kafka.yaml
kubectl apply -f .\deploy\elasticsearch.yaml
kubectl apply -f .\deploy\kibana.yaml
kubectl apply -f .\deploy\isb.yaml
kubectl apply -f .\deploy\belt-pipeline.yaml
kubectl apply -f .\deploy\logstash.yaml

Write-Host "Waiting for pods to stabilize..."
Start-Sleep -Seconds 40

# --------------------------------------------------
# 7 Create Kafka Topics
# --------------------------------------------------
$kafkaPod = kubectl get pod -n kafka -o jsonpath="{.items[0].metadata.name}"

if ($kafkaPod) {
    Write-Host "Ensuring Kafka topics exist..."

    kubectl exec -n kafka $kafkaPod -- kafka-topics --create --if-not-exists `
        --topic belt-data `
        --bootstrap-server localhost:9092 `
        --partitions 1 --replication-factor 1 2>$null

    kubectl exec -n kafka $kafkaPod -- kafka-topics --create --if-not-exists `
        --topic belt-predictions `
        --bootstrap-server localhost:9092 `
        --partitions 1 --replication-factor 1 2>$null
}

# --------------------------------------------------
# 8 Restart ML Runtime (clean refresh)
# --------------------------------------------------
kubectl delete pod -l numaflow.numaproj.io/vertex-name=ml-runtime -n default 2>$null
Start-Sleep -Seconds 15

# --------------------------------------------------
# 9 Start Port Forwards (background)
# --------------------------------------------------
Write-Host "Starting Port-Forwards..."

Start-Process powershell -ArgumentList "kubectl port-forward svc/kibana 5601:5601 -n elastic"
Start-Process powershell -ArgumentList "kubectl port-forward svc/numaflow-server 8443:8443 -n numaflow-system"
Start-Process powershell -ArgumentList "kubectl port-forward svc/elasticsearch 9200:9200 -n elastic"
Start-Process powershell -ArgumentList "kubectl port-forward svc/kafka 9092:9093 -n kafka"

Start-Sleep -Seconds 5

# --------------------------------------------------
#  Open UIs Automatically
# --------------------------------------------------
Start-Process "http://localhost:5601"
Start-Process "https://localhost:8443"
Start-Process "http://localhost:9200/_cat/indices?v"

# --------------------------------------------------
# 11 Setup Kibana Patterns
# --------------------------------------------------
Write-Host " Installing required Python dependencies..." -ForegroundColor Yellow
python -m pip install kafka-python elasticsearch requests pandas -q --disable-pip-version-check

Write-Host " Configuring Kibana index patterns..." -ForegroundColor Yellow
$env:KIBANA_URL = "http://localhost:5601"
python scripts/setup_kibana.py

Write-Host "=========================================" -ForegroundColor Green
Write-Host "  STREAMING + ELK STACK IS READY   " -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

# 12 Automatic Delta Catch-up
# --------------------------------------------------
Write-Host " Triggering Delta Catch-up Logic..." -ForegroundColor Yellow
$env:ELASTIC_URL = "http://localhost:9200"
$env:KAFKA_BOOTSTRAP = "localhost:9092"
python scripts/delta_catchup.py

# --------------------------------------------------
# 13 Start IoT Gateway Simulation
# --------------------------------------------------
Write-Host " Starting IoT Gateway Simulation (Background)..." -ForegroundColor Green
Start-Process powershell -ArgumentList '-NoExit -Command python -m app.iot_gateway --simulate --interval 2'

kubectl get pods -A

