<#
.SYNOPSIS
    QMemory Benchmark 快速验证脚本 (skip-ingest 模式)

.DESCRIPTION
    两步流程：
    Step 1 (一次性): bench.ps1 golden  — 创建 golden DB (ingest + eval 基准)
    Step 2 (每次):   bench.ps1 eval    — skip-ingest 快速 eval (~3 分钟)

    Golden DB 只需建一次。之后每次代码改动，重启 server 再跑 eval 即可。

.PARAMETER Action
    golden  — 创建 golden DB (首次执行，需 30-50 分钟)
    eval    — skip-ingest 快速评测 (~3 分钟)
    server  — 启动 server (使用 golden DB)
    status  — 检查 server 状态和 golden DB

.EXAMPLE
    # 首次（仅需一次）：启动 server + 创建 golden DB
    .\bench.ps1 server
    .\bench.ps1 golden

    # 后续每次：改代码 → 重启 server → 快速 eval
    .\bench.ps1 server
    .\bench.ps1 eval -Tag v030e

.NOTES
    所有路径相对于 workspace root (e:/memory)
#>

param(
    [Parameter(Position=0, Mandatory=$true)]
    [ValidateSet("golden", "eval", "server", "status")]
    [string]$Action,

    [string]$Tag = "",               # eval 报告标签 (如 v030e)
    [int]$Port = 18800,
    [string]$Preset = "release-full",
    [string]$Scale = "quick"
)

$ErrorActionPreference = "Stop"

# ── Paths ──────────────────────────────────────────────────────
$ROOT = "e:/memory"
$PYTHON = "$ROOT/.venv/Scripts/python.exe"
$env:PYTHONPATH = "$ROOT/qmemory/src;$ROOT/qmemory-bench/src"
$DEEPSEEK_KEY = "sk-c951f3e7a5924e05b83e67a03968c4af"
$GOLDEN_DB = "$ROOT/memory_golden.db"
$EVAL_USER = "golden"    # 固定 eval user prefix
$REPORT_DIR = "$ROOT/artifacts/qmemory-eval/reports"

# ── Helpers ────────────────────────────────────────────────────
function Test-Server {
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/health/" -TimeoutSec 3
        return $r.status -eq "ok"
    } catch {
        return $false
    }
}

function Get-ServerInfo {
    try {
        return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/health/" -TimeoutSec 3
    } catch {
        return $null
    }
}

# ── Actions ────────────────────────────────────────────────────

switch ($Action) {

    "status" {
        Write-Host "=== QMemory Bench Status ===" -ForegroundColor Cyan
        $info = Get-ServerInfo
        if ($info) {
            Write-Host "  Server:    RUNNING (port $Port)" -ForegroundColor Green
            Write-Host "  Memories:  $($info.memory_count)"
            Write-Host "  DB size:   $([math]::Round($info.db_size_bytes/1024, 1)) KB"
            Write-Host "  Uptime:    $([math]::Round($info.uptime_seconds, 0))s"
            Write-Host "  Embedding: $($info.embedding_model)"
        } else {
            Write-Host "  Server:    NOT RUNNING" -ForegroundColor Red
        }
        $dbExists = Test-Path $GOLDEN_DB
        Write-Host "  Golden DB: $(if ($dbExists) { 'EXISTS (' + [math]::Round((Get-Item $GOLDEN_DB).Length/1MB, 1) + ' MB)' } else { 'NOT FOUND' })" -ForegroundColor $(if ($dbExists) { 'Green' } else { 'Yellow' })

        # List recent reports
        $reports = Get-ChildItem "$REPORT_DIR/*.json" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 5
        if ($reports) {
            Write-Host "`n  Recent reports:" -ForegroundColor Cyan
            foreach ($r in $reports) {
                Write-Host "    $($r.Name)  ($($r.LastWriteTime.ToString('MM-dd HH:mm')))"
            }
        }
    }

    "server" {
        if (Test-Server) {
            Write-Host "Server already running on port $Port" -ForegroundColor Yellow
            $info = Get-ServerInfo
            Write-Host "  Memories: $($info.memory_count), Uptime: $([math]::Round($info.uptime_seconds))s"
            return
        }

        # Kill lingering processes on port
        $conn = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
        if ($conn) {
            $conn | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
            Start-Sleep -Seconds 2
        }

        $db = if (Test-Path $GOLDEN_DB) { $GOLDEN_DB } else { "$ROOT/memory_golden_new.db" }
        Write-Host "Starting server with DB: $db" -ForegroundColor Cyan
        Start-Process -FilePath $PYTHON -ArgumentList "-m", "qmemory", "serve", "--port", "$Port", "--db", "$db" -WorkingDirectory $ROOT -WindowStyle Minimized

        # Wait for ready
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 2
            if (Test-Server) {
                $info = Get-ServerInfo
                Write-Host "Server ready! Memories: $($info.memory_count)" -ForegroundColor Green
                return
            }
        }
        Write-Host "Server failed to start within 60s" -ForegroundColor Red
        exit 1
    }

    "golden" {
        Write-Host "=== Creating Golden DB ===" -ForegroundColor Cyan
        Write-Host "  Preset: $Preset, Scale: $Scale"
        Write-Host "  This will take 30-50 minutes (one-time only)"
        Write-Host ""

        if (-not (Test-Server)) {
            Write-Host "ERROR: Server not running. Run '.\bench.ps1 server' first." -ForegroundColor Red
            exit 1
        }

        $outFile = "$REPORT_DIR/golden_baseline.json"
        $env:OPENAI_API_KEY = $DEEPSEEK_KEY

        & $PYTHON -m qmemory_bench run `
            --preset $Preset `
            --scale $Scale `
            --target "http://127.0.0.1:$Port" `
            -o $outFile `
            --provider deepseek `
            --model deepseek-chat `
            --api-key $DEEPSEEK_KEY `
            --eval-user $EVAL_USER `
            --no-cleanup

        if ($LASTEXITCODE -eq 0) {
            # Read and show results
            $report = Get-Content $outFile -Encoding UTF8 | ConvertFrom-Json
            Write-Host "`n=== Golden Baseline Created ===" -ForegroundColor Green
            Write-Host "  Overall: $($report.overall)%"
            Write-Host "  Duration: $($report.duration)s"
            Write-Host "  Eval user: $EVAL_USER"
            Write-Host "  Report: $outFile"

            # Copy DB as golden snapshot
            $serverDb = "$ROOT/memory_golden_new.db"
            if (Test-Path $serverDb) {
                Copy-Item $serverDb $GOLDEN_DB -Force
                Write-Host "  Golden DB saved: $GOLDEN_DB" -ForegroundColor Green
            }
        } else {
            Write-Host "Golden benchmark FAILED" -ForegroundColor Red
            exit 1
        }
    }

    "eval" {
        if (-not $Tag) {
            $Tag = "eval_$(Get-Date -Format 'MMdd_HHmm')"
        }

        Write-Host "=== Quick Eval: $Tag ===" -ForegroundColor Cyan

        if (-not (Test-Server)) {
            Write-Host "ERROR: Server not running. Run '.\bench.ps1 server' first." -ForegroundColor Red
            exit 1
        }

        $info = Get-ServerInfo
        if ($info.memory_count -lt 10) {
            Write-Host "ERROR: Server has only $($info.memory_count) memories. Run '.\bench.ps1 golden' first." -ForegroundColor Red
            exit 1
        }
        Write-Host "  Server memories: $($info.memory_count)"

        $outFile = "$REPORT_DIR/${Tag}.json"
        $env:OPENAI_API_KEY = $DEEPSEEK_KEY

        $t0 = Get-Date
        & $PYTHON -m qmemory_bench run `
            --preset $Preset `
            --scale $Scale `
            --target "http://127.0.0.1:$Port" `
            -o $outFile `
            --provider deepseek `
            --model deepseek-chat `
            --api-key $DEEPSEEK_KEY `
            --eval-user $EVAL_USER `
            --skip-ingest

        $elapsed = ((Get-Date) - $t0).TotalSeconds

        if ($LASTEXITCODE -eq 0 -and (Test-Path $outFile)) {
            $report = Get-Content $outFile -Encoding UTF8 | ConvertFrom-Json
            Write-Host "`n=== Result: $Tag ===" -ForegroundColor Green
            Write-Host "  Overall: $($report.overall)%" -ForegroundColor $(if ($report.overall -ge 72) { 'Green' } elseif ($report.overall -ge 68) { 'Yellow' } else { 'Red' })
            Write-Host "  Duration: $([math]::Round($elapsed, 0))s"
            Write-Host "  Report: $outFile"

            # Compare with golden baseline if exists
            $goldenReport = "$REPORT_DIR/golden_baseline.json"
            if (Test-Path $goldenReport) {
                $golden = Get-Content $goldenReport -Encoding UTF8 | ConvertFrom-Json
                $diff = [math]::Round($report.overall - $golden.overall, 1)
                $sign = if ($diff -ge 0) { "+" } else { "" }
                $color = if ($diff -gt 5) { 'Green' } elseif ($diff -lt -5) { 'Red' } else { 'Yellow' }
                Write-Host "  vs Golden: ${sign}${diff}%" -ForegroundColor $color
            }
        } else {
            Write-Host "Eval FAILED" -ForegroundColor Red
            exit 1
        }
    }
}
