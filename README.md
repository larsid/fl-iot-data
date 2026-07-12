# fl-iot-data

Código de execução e dados de experimentos de *Federated Learning* (FL) sob emulação
baseada em containers. O ambiente usa o **Fogbed** (Containernet/Mininet) para emular
dispositivos IoT com restrição progressiva de CPU, memória e largura de banda, e o
**Flower** para orquestrar o treinamento federado. As métricas coletadas permitem confrontar
o comportamento emulado com as medições em **Raspberry Pi 3 reais** publicadas por
Wong et al. (2026).

## Cenário experimental

O cenário replica o de Wong et al.: **FedAvg**, modelo **CNN3** (878.538 parâmetros),
**CIFAR-10** particionado de forma IID (64 partições, 1 por cliente), **8 clientes**,
**500 rounds**, 2 *epochs* locais, *batch* 16, *learning rate* 0,01, participação total.
Servidor e clientes rodam em containers sobre a rede virtual do Fogbed, com a mesma stack
de FL da referência (PyTorch 1.13.1, Flower 1.11.0).

Ao todo são **35 execuções**, cada uma com seed independente registrada no seu
`manifest.json`:

- **Experimento principal** (20 execuções): 4 níveis progressivos de emulação × 5 repetições.
- **Experimento secundário** (15 execuções): N3 com banda de 100 / 50 / 25 Mbps × 5 repetições.

| Nível | Emulação aplicada ao cliente |
|---|---|
| N0 | Container sem restrições (baseline, análogo à simulação de referência) |
| N1 | CPU limitada (1 vCPU, fixada em núcleo dedicado) |
| N2 | N1 + memória limitada (1 GiB) |
| N3 | N2 + rede limitada (100 Mbps) - perfil completo do Pi 3 |

## Estrutura

```
fl-iot-data/
├── experiments/        # código que executa os experimentos
│   ├── orchestrate.py          # orquestra as 35 execuções (status, seeds, tempos)
│   ├── config.py               # parâmetros de FL e perfis de emulação por nível
│   ├── experiment_runner.py    # monta a topologia Fogbed e roda uma execução
│   ├── experiment_level{0..3}.py   # wrappers por nível de emulação
│   ├── reduce_run.py           # reduz os logs brutos de uma execução em summary.json
│   ├── fl/                     # implementação FL: CNN3, FedAvg, cliente e servidor (Flower)
│   ├── collector/              # coletor de CPU/memória dos containers (cgroups v2, 1 Hz)
│   ├── docker/                 # Dockerfile e entrypoint da imagem dos containers FL
│   ├── clean.sh                # limpeza de estado do Fogbed entre execuções
│   └── data/                   # CIFAR-10 (baixado sob demanda; não versionado)
├── results/            # dados brutos e reduzidos das 35 execuções
└── requirements.txt    # dependências do host
```

## Resultados

Cada execução fica em `results/<nível>/<timestamp>/`:

| Arquivo | Conteúdo |
|---|---|
| `manifest.json` | configuração da execução: nível, seed, recursos, banda e imagens |
| `accuracy.csv` | acurácia e perda no conjunto de teste, por round |
| `client_1..8.jsonl` | timestamps por round de cada cliente (treino e troca de atualização) |
| `stats.csv` | uso de CPU e memória por container, amostrado a 1 Hz |
| `summary.json` | redução da execução, produzida por `reduce_run.py` |

O bloco `aggregate` do `summary.json` traz as métricas agregadas usadas no estudo, com a
mesma nomenclatura: `test_accuracy_final`, `convergence_speed_rounds`, `avg_training_time_s`,
`avg_update_exchange_time_s` (e sua decomposição `avg_eval_time_s` no servidor e
`avg_update_exchange_net_s` de transmissão pela rede), `avg_cpu_pct`, `avg_mem_pct` e
`avg_mem_bytes`, nas janelas de treino e de round completo (`diagnostic_full_window`).

## Reexecução

As execuções são conduzidas pelo orquestrador, que mostra quais já rodaram, o tempo
estimado e o real de cada uma, e permite rodar as 35 de uma vez ou apenas as escolhidas.
Requer Docker, Mininet + Open vSwitch e privilégio de root. O host precisa de Python 3.9 ou
3.10 (exigência do PyTorch 1.13.1 usado pelo torchvision no host) e de pelo menos 12 CPUs,
pois o servidor é fixado nos núcleos 0–3 e os clientes nos núcleos 4–11 (N1–N3).

```bash
pip install -r requirements.txt

# Setup (uma vez): baixar o CIFAR-10 e construir a imagem Docker dos containers FL
cd experiments
python -c "from torchvision import datasets; datasets.CIFAR10(root='data', download=True)"
docker build -f docker/fl.Dockerfile -t fl-base:1.13.1 .
docker tag fl-base:1.13.1 fl-server:latest
docker tag fl-base:1.13.1 fl-client:latest

# Terminal 1 (worker daemon do Fogbed, deixar aberto):
sudo python -m clusternet.server.worker_app

# Terminal 2:
cd experiments
sudo -v && bash clean.sh     # limpa estado de execuções anteriores
python orchestrate.py        # abre o menu interativo
```

Cada execução chama automaticamente `reduce_run.py` ao terminar, gravando o `summary.json`
em `results/`. O estado é persistido em `results/orchestrator_state.json`, então é seguro
interromper e retomar. `python orchestrate.py status` imprime a tabela de status sem rodar
nada.

## Ambiente

| Componente | Versão |
|---|---|
| SO do host | Ubuntu 24.04 LTS (kernel 6.17) |
| Docker | 29.4.3 (cgroups v2) |
| Fogbed / clusternet | 1.3.0 / 0.9.3 |
| Python nos containers | 3.9 |
| Python no host | 3.9 ou 3.10 |
| PyTorch / torchvision (containers) | 1.13.1+cpu / 0.14.1 |
| Flower (containers) | 1.11.0 |

A stack dos containers é fixada em `experiments/docker/fl.Dockerfile`, o que congela as
versões independentemente do host.

## Referências

**Referência base** (estudo replicado):

- Wong, K.-S. et al. (2026), *An Empirical Study of Federated Learning on IoT-Edge Devices:
  Resource Allocation and Heterogeneity*, IEEE Transactions on Neural Networks and Learning
  Systems, vol. 37, n. 2, pp. 753–765. DOI: 10.1109/TNNLS.2025.3611415.

**Ferramentas e recursos utilizados:**

- Coutinho, A.; Greve, F.; Prazeres, C.; Cardoso, J. (2018), *Fogbed: A Rapid-Prototyping
  Emulation Environment for Fog Computing*, IEEE ICC 2018, pp. 1–7.
  DOI: 10.1109/ICC.2018.8423003.
- Coutinho, A. et al. (2023), *Rapid-Prototyping of Integrated Edge/Fog and DLT/Blockchain
  Systems with Fogbed*, IEEE ICC 2023, pp. 622–627 — versão distribuída do Fogbed
  (`fogbed`/`clusternet`) usada neste repositório. DOI: 10.1109/ICC45041.2023.10279234.
- Beutel, D. J. et al. (2020), *Flower: A Friendly Federated Learning Framework*,
  arXiv:2007.14390.
- McMahan, H. B. et al. (2017), *Communication-Efficient Learning of Deep Networks from
  Decentralized Data*, AISTATS 2017, PMLR vol. 54, pp. 1273–1282 — algoritmo FedAvg.
- Lantz, B.; Heller, B.; McKeown, N. (2010), *A Network in a Laptop: Rapid Prototyping for
  Software-Defined Networks*, ACM HotNets '10 — Mininet. DOI: 10.1145/1868447.1868466.
- Peuster, M.; Karl, H.; van Rossem, S. (2016), *MeDICINE: Rapid Prototyping of
  Production-Ready Network Services in Multi-PoP Environments*, IEEE NFV-SDN 2016,
  pp. 148–153 — Containernet. DOI: 10.1109/NFV-SDN.2016.7919490.
- Paszke, A. et al. (2019), *PyTorch: An Imperative Style, High-Performance Deep Learning
  Library*, NeurIPS 2019, pp. 8024–8035.
- Krizhevsky, A. (2009), *Learning Multiple Layers of Features from Tiny Images*, relatório
  técnico, University of Toronto — dataset CIFAR-10.

## Licença

Distribuído sob a licença MIT - ver [LICENSE](LICENSE).
