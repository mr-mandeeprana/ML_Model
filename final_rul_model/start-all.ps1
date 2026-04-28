param(
    [bool]$StartSimulation = $true,
    [bool]$SkipDeltaCatchup = $false,
    [bool]$SkipKibanaSetup = $false
)

$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PSScriptRoot

$ErrorActionPreference = "Stop"


function kubectl {
    if ($input) {
        $input | & minikube kubectl -- @args
    } else {
        & minikube kubectl -- @args
    }
}




function Write-Section($text, $color = "Cyan") {
    Write-Host "=========================================" -ForegroundColor $color
    Write-Host " $text " -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor $color
}

function Run-Step($message, [scriptblock]$block) {
    Write-Host "`n>> $message" -ForegroundColor Yellow
    $global:LASTEXITCODE = 0
    try {
        & $block
    } catch {
        throw "Step failed: $message`n$($_.Exception.Message)"
    }
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Step failed: $message (exit code: $LASTEXITCODE)"
    }
}

function Ensure-Namespace($name) {
    minikube kubectl -- create namespace $name --dry-run=client --output yaml | minikube kubectl -- apply -f -
}



function Wait-Deployment($name, $namespace, $timeout = "180s") {
    kubectl wait --for=condition=available deployment/$name -n $namespace --timeout=$timeout
}

function Wait-PodsByLabel($namespace, $label, $timeoutSeconds = 180) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $pods = kubectl get pods -n $namespace -l $label --output json | ConvertFrom-Json
        if ($pods.items.Count -gt 0) {
            $allReady = $true
            foreach ($pod in $pods.items) {
                $statuses = @($pod.status.containerStatuses)
                if ($statuses.Count -eq 0) {
                    $allReady = $false
                    break
                }
                foreach ($status in $statuses) {
                    if (-not $status.ready) {
                        $allReady = $false
                        break
                    }
                }
                if (-not $allReady) { break }
            }
            if ($allReady) { return }
        }
        Start-Sleep -Seconds 5
    }
    throw "Timeout waiting for pods in namespace '$namespace' with label '$label'"
}

function Ensure-PortFree($port) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connections) {
        $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $pids) {
            try {
                $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
                if ($proc -and $proc.ProcessName -match "kubectl|powershell|pwsh") {
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                }
            } catch {}
        }
        Start-Sleep -Seconds 2
    }
}

function Start-PortForward($name, $namespace, $localPort, $remotePort) {
    Ensure-PortFree $localPort
    Start-Process powershell -WindowStyle Hidden -ArgumentList "-Command", "minikube kubectl -- port-forward svc/$name ${localPort}:${remotePort} -n $namespace"
}

function Get-MinikubeStatusWithTimeout($timeoutSeconds = 20) {
    $job = Start-Job -ScriptBlock {
        try {
            & minikube status --output json 2>$null | ConvertFrom-Json
        } catch {
            $null
        }
    }

    if (Wait-Job -Job $job -Timeout $timeoutSeconds) {
        $result = Receive-Job -Job $job -ErrorAction SilentlyContinue
        Remove-Job -Job $job -Force | Out-Null
        return $result
    }

    Stop-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
    Remove-Job -Job $job -Force | Out-Null
    Write-Warning "Timed out waiting for 'minikube status'. Will attempt startup path."
    return $null
}


function Test-KubernetesApiReady() {
    $job = Start-Job -ScriptBlock {
        # Redirect stderr to $null to avoid NativeCommandError when context is missing
        & minikube kubectl -- cluster-info --request-timeout=5s 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        return $false
    }

    if (Wait-Job -Job $job -Timeout 15) {
        $ready = Receive-Job -Job $job -ErrorAction SilentlyContinue
        Remove-Job -Job $job -Force | Out-Null
        $global:LASTEXITCODE = 0
        return ($ready -eq $true)
    }

    Stop-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
    Remove-Job -Job $job -Force | Out-Null
    $global:LASTEXITCODE = 0
    return $false
}



Write-Section "STARTING FULL ML STREAMING STACK"

$availableMem = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024
Write-Host "Available RAM: $([math]::Round($availableMem, 2)) MB" -ForegroundColor Yellow
if ($availableMem -lt 4000) {
    Write-Warning "Low Available RAM detected (< 4GB). The stack might be unstable."
    Write-Warning "Recommended: Close other heavy applications before continuing."
}

Write-Host "`n>> Checking Kubernetes API status" -ForegroundColor Yellow
$script:kubernetesApiReady = Test-KubernetesApiReady
$global:LASTEXITCODE = 0

if ($kubernetesApiReady) {
    Write-Host "Kubernetes API is already reachable; skipping Minikube bootstrap." -ForegroundColor Green
} else {
    Run-Step "Checking Minikube status" {
        $script:minikubeStatus = Get-MinikubeStatusWithTimeout 20
    }

    if ($null -ne $minikubeStatus -and $minikubeStatus.Host -eq "Running") {
        Write-Host "Minikube already running." -ForegroundColor Green
    } else {
        $minikubeMem = 3000
        if ($availableMem -lt 4000) {
            $minikubeMem = 2048
            Write-Host "Limiting Minikube memory to ${minikubeMem}MB due to low RAM." -ForegroundColor Yellow
        }
        
        Run-Step "Starting Minikube" {
            minikube start --wait=all --memory $minikubeMem
        }
    }
}


Run-Step "Ensuring namespaces" {
    Ensure-Namespace "kafka"
    Ensure-Namespace "elastic"
    Ensure-Namespace "numaflow-system"
}


$nfInstalled = $false
try {
    $deployments = kubectl get deployments -n numaflow-system --output json | ConvertFrom-Json
    if ($deployments.items.Count -gt 0) {
        $nfInstalled = $true
    }
} catch {
    $nfInstalled = $false
}


if (-not $nfInstalled) {
    Run-Step "Installing Numaflow" {
        kubectl apply -f https://github.com/numaproj/numaflow/releases/latest/download/install.yaml
    }
}

Run-Step "Waiting for Numaflow control plane" {
    try {
        Wait-PodsByLabel "numaflow-system" "app.kubernetes.io/part-of=numaflow" 240
    } catch {
        Write-Warning "Numaflow control plane wait failed: $($_.Exception.Message)"
        Write-Warning "Continuing because the cluster is already serving workloads."
        $global:LASTEXITCODE = 0
    }
}

Run-Step "Switching Docker to Minikube environment" {
    minikube docker-env --shell powershell | Invoke-Expression
}

Run-Step "Building ML Runtime Docker image" {
    docker build -t final-rul-model:latest .
}

Run-Step "Deploying Zookeeper" { kubectl apply -f .\deploy\zookeeper.yaml }
Run-Step "Deploying Kafka" { kubectl apply -f .\deploy\kafka.yaml }
Run-Step "Deploying Elasticsearch" { kubectl apply -f .\deploy\elasticsearch.yaml }
Run-Step "Deploying Kibana" { kubectl apply -f .\deploy\kibana.yaml }
Run-Step "Deploying Logstash" { kubectl apply -f .\deploy\logstash.yaml }
Run-Step "Deploying thresholds ConfigMap" { kubectl apply -f .\deploy\thresholds-configmap.yaml }
Run-Step "Deploying ISB" { kubectl apply -f .\deploy\isb.yaml }
Run-Step "Deploying Numaflow pipeline" { kubectl apply -f .\deploy\belt-pipeline.yaml }

Run-Step "Waiting for infrastructure readiness" {
    Wait-Deployment "zookeeper" "kafka" "300s"
    Wait-Deployment "kafka" "kafka" "300s"
    Wait-Deployment "elasticsearch" "elastic" "300s"
    Wait-Deployment "kibana" "elastic" "300s"
    Wait-Deployment "logstash" "elastic" "300s"
    Wait-PodsByLabel "default" "numaflow.numaproj.io/vertex-name=ml-runtime" 300
}


Run-Step "Ensuring Kafka topics exist" {
    $kafkaPod = kubectl get pod -n kafka -l app=kafka --output jsonpath="{.items[0].metadata.name}"
    if (-not $kafkaPod) {
        throw "Kafka pod not found."
    }

    kubectl exec -n kafka $kafkaPod -- kafka-topics.sh --create --if-not-exists `
        --topic belt-data `
        --bootstrap-server kafka.kafka.svc.cluster.local:9092 `
        --partitions 1 --replication-factor 1

    kubectl exec -n kafka $kafkaPod -- kafka-topics.sh --create --if-not-exists `
        --topic belt-predictions `
        --bootstrap-server kafka.kafka.svc.cluster.local:9092 `
        --partitions 1 --replication-factor 1
}

Write-Host "`n>> Restarting ml-runtime vertex pod for clean refresh" -ForegroundColor Yellow
kubectl delete pod -l numaflow.numaproj.io/vertex-name=ml-runtime -n default --ignore-not-found=true | Out-Null
Start-Sleep -Seconds 5

Run-Step "Waiting for ml-runtime vertex to return" {
    Wait-PodsByLabel "default" "numaflow.numaproj.io/vertex-name=ml-runtime" 180
}

Run-Step "Starting port-forwards" {
    Start-PortForward "kibana" "elastic" 5601 5601
    Start-PortForward "numaflow-server" "numaflow-system" 8443 8443
    Start-PortForward "elasticsearch" "elastic" 9200 9200
    Start-PortForward "kafka" "kafka" 9092 9092
}

Start-Sleep -Seconds 5

Write-Host "`nOpening dashboards..." -ForegroundColor Green
Start-Process "http://localhost:5601"
Start-Process "https://localhost:8443"
Start-Process "http://localhost:9200/_cat/indices?v"

if (-not $SkipKibanaSetup) {
    Run-Step "Installing helper Python dependencies" {
        python -m pip install kafka-python elasticsearch requests pandas -q --disable-pip-version-check
    }

    Run-Step "Configuring Kibana index patterns" {
        $env:KIBANA_URL = "http://localhost:5601"
        python .\scripts\setup_kibana.py
    }
}

Write-Section "STREAMING + ELK STACK IS READY" "Green"

if (-not $SkipDeltaCatchup) {
    Run-Step "Triggering Delta Catch-up Logic" {
        $env:ELASTIC_URL = "http://localhost:9200"
        $env:KAFKA_BOOTSTRAP = "localhost:9092"
        python .\scripts\delta_catchup.py
    }
}

if ($StartSimulation) {
    Write-Host "`n>> Starting IoT Gateway Simulation (Background)" -ForegroundColor Green
    Start-Process powershell -WindowStyle Hidden -ArgumentList '-Command', 'python .\app\iot_gateway.py --simulate --interval 2'
} else {
    Write-Host "`nSimulation skipped." -ForegroundColor Yellow
}

Write-Host "`nCluster pod status:" -ForegroundColor Cyan
kubectl get pods -A