# SIMD vs SIMT Performance Comparison

Este projeto implementa uma comparação de performance entre arquiteturas SIMD (CPU) e SIMT (GPU) para operações de multiplicação e acumulação (MAC), fundamentais em aceleradores de redes neurais convolucionais.

## Motivação

O desenvolvimento de aceleradores de redes neurais convolucionais requer otimização das operações mais computacionalmente intensivas. A operação de multiplicação e acumulação (MAC) é uma das mais críticas, sendo executada milhões de vezes durante o processamento de uma única camada convolucional. Este projeto fornece uma base de comparação entre diferentes arquiteturas de processamento paralelo para auxiliar na tomada de decisões de design.

## Estrutura do Projeto

```
.
│
├── Makefile          # Makefile para compilação
├── simd_cpu.c        # Implementação SIMD com AVX2
├── simt_gpu.c        # Host code para OpenCL
└── simt_gpu.cl       # Kernel OpenCL
```

## Implementações

### SIMD (CPU - AVX2)
- **Arquivo**: `simd_cpu.c`
- **Tecnologia**: Intel AVX2 (Advanced Vector Extensions 2)
- **Paralelismo**: 8 operações float simultâneas por instrução
- **Características**:
  - Utiliza instruções vetoriais `_mm256_*`
  - Memória alinhada para otimização
  - Processamento de 10 milhões de elementos

### SIMT (GPU - OpenCL)
- **Arquivos**: `simt_gpu.c` e `simt_gpu.cl`
- **Tecnologia**: OpenCL (Open Computing Language)
- **Paralelismo**: Milhares de threads executando simultaneamente
- **Características**:
  - Kernel OpenCL para execução massivamente paralela
  - Cada thread processa um elemento
  - Transferência de dados CPU ↔ GPU incluída na medição

## Operação Avaliada

A operação implementada é a **Multiplicação e Acumulação (MAC)**:
```
C[i] += A[i] * B[i]
```

Esta operação é executada iterativamente para simular a carga computacional típica de:
- Convoluções 2D
- Multiplicações matriz-vetor
- Produtos escalares em redes neurais

## Compilação

### Pré-requisitos
- GCC com suporte a AVX2
- OpenCL runtime e headers
- CPU com suporte a AVX2
- GPU compatível com OpenCL

### Compilação
```bash
#Testado em Ubuntu 22.04.2 LTS
sudo apt install ocl-icd-opencl-dev intel-opencl-icd clinfo
make
```

### Execução
```bash
#Executado em:
#Cpu: 12th Gen Intel(R) Core(TM) i5-12400
#GPU integrada: Intel UHD Graphics 730
make run
```

## Métricas de Performance

O projeto mede e compara:

1. **Tempo de execução** (segundos)
2. **Throughput** (GMAC/s - Giga MAC operations per second)

### Exemplo de Saída
```
make run
CPU (AVX2) MAC time (1000 iters): 5.190659 s
CPU throughput: 1.93 GMAC/s
GPU (OpenCL) MAC time (1000 iters): 0.082614 s
GPU throughput: 121.05 GMAC/s
```

## Considerações para Aceleradores de CNN

### Vantagens SIMD (CPU):
- **Latência baixa**: Ideal para inferência em tempo real
- **Controle fino**: Flexibilidade para otimizações específicas
- **Memória unificada**: Sem overhead de transferência de dados
- **Determinístico**: Comportamento previsível para sistemas críticos

### Vantagens SIMT (GPU):
- **Throughput alto**: Excelente para processamento em lote
- **Paralelismo massivo**: Milhares de operações simultâneas
- **Eficiência energética**: Melhor performance por watt
- **Escalabilidade**: Facilmente adaptável para diferentes tamanhos de problema

## Aplicações em Aceleradores de CNN

Este benchmark é relevante para:

1. **Escolha de arquitetura**: SIMD vs SIMT vs híbrida
2. **Dimensionamento de recursos**: Quantos cores/ALUs incluir
3. **Análise de gargalos**: Identificar limitações de memória vs computação
4. **Validação de design**: Comparar implementações customizadas
5. **Estimativa de performance**: Projetar throughput esperado

## Extensões Possíveis

- [ ] Implementação com diferentes precisões (int8, fp16, bf16)
- [ ] Comparação com bibliotecas otimizadas (Intel MKL, cuBLAS)
- [ ] Análise de consumo de energia
- [ ] Implementação em FPGA para comparação
- [ ] Convoluções 2D completas ao invés de MAC simples
- [ ] Análise de cache miss e padrões de acesso à memória
- [ ] Comparar consumo de energia (com perf, powercap, etc.) → GMAC/s/Watt
- [ ] Verificar uso de memória (profiler OpenCL ou /proc)
- [ ] Validar resultado da saída C para bater com ambas as versões
- [ ] Testar escalabilidade com N = 10⁶, 10⁷, 10⁸