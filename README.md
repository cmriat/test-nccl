# NCCL Test Environment

A pixi-managed environment for running NCCL (NVIDIA Collective Communications Library) tests.

## Prerequisites

- Linux x86_64
- CUDA 12.9+

## Quick Start

### 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

After installation, restart your shell or run:

```bash
source ~/.bashrc
```

### 2. Install dependencies

```bash
pixi install
```

This will install all dependencies defined in `pixi.toml`:
- nccl 2.19.3
- nccl-tests

### 3. Enter the environment shell

```bash
pixi shell
```

You are now in the pixi environment with all tools available.
