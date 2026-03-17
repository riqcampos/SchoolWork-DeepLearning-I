"""
Trabalho I - Aprendizado Profundo
Exploração do comportamento do erro de treino vs erro de generalização.
"""

import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# ETAPA 1: Geração de dados
# =============================================================================

def f1(x):
    """f1(x) = 0.5x + 0.3 + rand(-0.1, 0.1)"""
    noise = np.random.uniform(-0.1, 0.1, size=x.shape)
    return 0.5 * x + 0.3 + noise

def f2(x):
    """f2(x) = 0.5x² - 0.3x + 0.8 + rand(-0.1, 0.1)"""
    noise = np.random.uniform(-0.1, 0.1, size=x.shape)
    return 0.5 * x**2 - 0.3 * x + 0.8 + noise

def generate_dataset(func, n_samples):
    """Amostra n_samples pontos de x ~ U[0,1] e aplica func."""
    x = np.random.uniform(0, 1, size=(n_samples, 1)).astype(np.float32)
    y = func(x).astype(np.float32)
    return x, y

SAMPLE_SIZES = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                150, 200, 250, 300, 350, 400, 450, 500]

# Gerar dados de treino para cada tamanho e distribuição
datasets_f1 = {}
datasets_f2 = {}
for n in SAMPLE_SIZES:
    np.random.seed(n)  # reprodutibilidade por tamanho
    datasets_f1[n] = generate_dataset(f1, n)
    datasets_f2[n] = generate_dataset(f2, n)

# Gerar dados de teste (1000 amostras) - ETAPA 3
np.random.seed(9999)
test_x_f1, test_y_f1 = generate_dataset(f1, 1000)
np.random.seed(8888)
test_x_f2, test_y_f2 = generate_dataset(f2, 1000)

# =============================================================================
# ETAPA 2: Definição dos modelos
# =============================================================================

class ModelA(nn.Module):
    """Modelo A: apenas um neurônio (regressão linear simples)."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

class ModelB(nn.Module):
    """Modelo B: 2 neurônios na camada escondida + 1 de saída. Ativação linear."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = self.hidden(x)  # ativação linear (identidade)
        return self.output(x)

class ModelC(nn.Module):
    """Modelo C: 30 neurônios na camada escondida + 1 de saída. Ativação linear."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 30)
        self.output = nn.Linear(30, 1)

    def forward(self, x):
        x = self.hidden(x)  # ativação linear (identidade)
        return self.output(x)

# Criar modelos e salvar backup dos pesos iniciais
torch.manual_seed(42)
model_a_backup = copy.deepcopy(ModelA().state_dict())
torch.manual_seed(42)
model_b_backup = copy.deepcopy(ModelB().state_dict())
torch.manual_seed(42)
model_c_backup = copy.deepcopy(ModelC().state_dict())

# =============================================================================
# Funções de treino e avaliação
# =============================================================================

def train_model(model, x_train, y_train, epochs=2000, lr=0.01):
    """Treina o modelo e retorna o loss final de treino."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_t = torch.tensor(x_train)
    y_t = torch.tensor(y_train)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

    # Loss final de treino
    model.eval()
    with torch.no_grad():
        pred = model(x_t)
        train_loss = criterion(pred, y_t).item()

    return train_loss

def evaluate_model(model, x_test, y_test):
    """Avalia o modelo e retorna MSE, MAE e RMSE."""
    model.eval()
    x_t = torch.tensor(x_test)
    y_t = torch.tensor(y_test)

    with torch.no_grad():
        pred = model(x_t)
        mse = nn.MSELoss()(pred, y_t).item()
        mae = torch.mean(torch.abs(pred - y_t)).item()
        rmse = np.sqrt(mse)

    return {"mse": mse, "mae": mae, "rmse": rmse, "predictions": pred.numpy()}

# =============================================================================
# ETAPAS 2 e 3: Treino e avaliação para todos os cenários
# =============================================================================

print("=" * 70)
print("ETAPAS 1-3: Treinamento e avaliação dos modelos A, B, C")
print("=" * 70)

model_configs = {
    "A": (ModelA, model_a_backup),
    "B": (ModelB, model_b_backup),
    "C": (ModelC, model_c_backup),
}

results = {}  # results[(func_name, model_name, n_samples)] = {...}

for func_name, datasets, test_x, test_y in [
    ("f1", datasets_f1, test_x_f1, test_y_f1),
    ("f2", datasets_f2, test_x_f2, test_y_f2),
]:
    for model_name, (ModelClass, backup_state) in model_configs.items():
        for n in SAMPLE_SIZES:
            x_train, y_train = datasets[n]

            # Restaurar pesos iniciais (backup)
            model = ModelClass()
            model.load_state_dict(copy.deepcopy(backup_state))

            # Treinar
            train_loss = train_model(model, x_train, y_train)

            # Avaliar no treino
            train_metrics = evaluate_model(model, x_train, y_train)

            # Avaliar no teste (1000 amostras)
            test_metrics = evaluate_model(model, test_x, test_y)

            results[(func_name, model_name, n)] = {
                "train_mse": train_loss,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "test_mse": test_metrics["mse"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            }

            print(f"{func_name} | Modelo {model_name} | N={n:>3d} | "
                  f"Train MSE={train_loss:.6f} | Test MSE={test_metrics['mse']:.6f} | "
                  f"Test MAE={test_metrics['mae']:.6f}")

# =============================================================================
# Gráficos principais
# =============================================================================

REAL_MAE = 0.1  # erro intrínseco (MAE teórico do ruído uniforme [-0.1, 0.1])
# MSE teórico do ruído: E[U(-0.1,0.1)^2] = (0.2)^2/12 = 1/300 ≈ 0.00333
REAL_MSE = (0.2**2) / 12

def plot_train_vs_test(func_name, metric="mse", ylabel="MSE"):
    """Gráfico de erro de treino vs teste para cada modelo."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"A": "blue", "B": "orange", "C": "green"}

    for model_name in ["A", "B", "C"]:
        train_vals = [results[(func_name, model_name, n)][f"train_{metric}"] for n in SAMPLE_SIZES]
        test_vals = [results[(func_name, model_name, n)][f"test_{metric}"] for n in SAMPLE_SIZES]

        ax.plot(SAMPLE_SIZES, train_vals, f"--", color=colors[model_name],
                label=f"Modelo {model_name} - Treino", marker="o", markersize=3)
        ax.plot(SAMPLE_SIZES, test_vals, f"-", color=colors[model_name],
                label=f"Modelo {model_name} - Teste (1000)", marker="s", markersize=3)

    if metric == "mae":
        ax.axhline(y=REAL_MAE, color="red", linestyle=":", linewidth=2,
                    label=f"Erro real (MAE = {REAL_MAE})")
    elif metric == "mse":
        ax.axhline(y=REAL_MSE, color="red", linestyle=":", linewidth=2,
                    label=f"Risco real (MSE ≈ {REAL_MSE:.5f})")

    ax.set_xlabel("Número de amostras de treino")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Erro de Treino vs Teste - {func_name.upper()} ({ylabel})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{func_name}_train_vs_test_{metric}.png", dpi=150)
    plt.close(fig)
    print(f"  Salvo: {func_name}_train_vs_test_{metric}.png")

def plot_gap(func_name):
    """Gráfico do gap (teste - treino) para identificar overfitting."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"A": "blue", "B": "orange", "C": "green"}

    for model_name in ["A", "B", "C"]:
        gaps = [
            results[(func_name, model_name, n)]["test_mse"] -
            results[(func_name, model_name, n)]["train_mse"]
            for n in SAMPLE_SIZES
        ]
        ax.plot(SAMPLE_SIZES, gaps, "-o", color=colors[model_name],
                label=f"Modelo {model_name}", markersize=4)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Número de amostras de treino")
    ax.set_ylabel("Gap (MSE Teste - MSE Treino)")
    ax.set_title(f"Gap de Generalização - {func_name.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{func_name}_gap.png", dpi=150)
    plt.close(fig)
    print(f"  Salvo: {func_name}_gap.png")

def plot_predictions_example(func_name, datasets, test_x, test_y, n_sample=10):
    """Visualiza as predições de cada modelo para um tamanho de amostra específico."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_train, y_train = datasets[n_sample]

    # Função verdadeira (sem ruído)
    x_line = np.linspace(0, 1, 200).reshape(-1, 1).astype(np.float32)
    if func_name == "f1":
        y_true = 0.5 * x_line + 0.3
    else:
        y_true = 0.5 * x_line**2 - 0.3 * x_line + 0.8

    for idx, model_name in enumerate(["A", "B", "C"]):
        ax = axes[idx]
        ModelClass, backup_state = model_configs[model_name]
        model = ModelClass()
        model.load_state_dict(copy.deepcopy(backup_state))
        train_model(model, x_train, y_train)

        model.eval()
        with torch.no_grad():
            y_pred_line = model(torch.tensor(x_line)).numpy()

        ax.scatter(x_train, y_train, color="blue", s=20, label="Treino", zorder=5)
        ax.plot(x_line, y_true, "g-", linewidth=2, label="Função verdadeira")
        ax.plot(x_line, y_pred_line, "r--", linewidth=2, label="Predição do modelo")
        ax.set_title(f"Modelo {model_name} (N={n_sample})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Predições - {func_name.upper()} com N={n_sample} amostras", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{func_name}_predictions_N{n_sample}.png", dpi=150)
    plt.close(fig)
    print(f"  Salvo: {func_name}_predictions_N{n_sample}.png")

print("\nGerando gráficos das Etapas 1-3...")
for func_name in ["f1", "f2"]:
    plot_train_vs_test(func_name, "mse", "MSE")
    plot_train_vs_test(func_name, "mae", "MAE")
    plot_train_vs_test(func_name, "rmse", "RMSE")
    plot_gap(func_name)

for func_name, datasets in [("f1", datasets_f1), ("f2", datasets_f2)]:
    test_x = test_x_f1 if func_name == "f1" else test_x_f2
    test_y = test_y_f1 if func_name == "f1" else test_y_f2
    for n in [5, 10, 50, 200, 500]:
        plot_predictions_example(func_name, datasets, test_x, test_y, n)

# =============================================================================
# Tabela resumo
# =============================================================================

print("\n" + "=" * 70)
print("TABELA RESUMO: MSE Treino vs MSE Teste vs Risco Real")
print("=" * 70)
for func_name in ["f1", "f2"]:
    print(f"\n--- {func_name.upper()} ---")
    print(f"{'Modelo':<8} {'N':>5} {'MSE Treino':>12} {'MSE Teste':>12} {'MAE Teste':>12} "
          f"{'RMSE Teste':>12} {'Risco Real':>12}")
    print("-" * 75)
    for model_name in ["A", "B", "C"]:
        for n in SAMPLE_SIZES:
            r = results[(func_name, model_name, n)]
            print(f"{model_name:<8} {n:>5d} {r['train_mse']:>12.6f} {r['test_mse']:>12.6f} "
                  f"{r['test_mae']:>12.6f} {r['test_rmse']:>12.6f} {REAL_MSE:>12.6f}")

# =============================================================================
# EXPERIMENTO EXTRA 1: Variar quantidade de camadas
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENTO EXTRA 1: Impacto do aumento de camadas (ativação linear)")
print("=" * 70)

class ModelDeep2(nn.Module):
    """2 camadas escondidas de 10 neurônios cada."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 1),
        )
    def forward(self, x):
        return self.layers(x)

class ModelDeep4(nn.Module):
    """4 camadas escondidas de 10 neurônios cada."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 1),
        )
    def forward(self, x):
        return self.layers(x)

class ModelDeep8(nn.Module):
    """8 camadas escondidas de 10 neurônios cada."""
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1, 10)]
        for _ in range(7):
            layers.append(nn.Linear(10, 10))
        layers.append(nn.Linear(10, 1))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

torch.manual_seed(42)
deep2_backup = copy.deepcopy(ModelDeep2().state_dict())
torch.manual_seed(42)
deep4_backup = copy.deepcopy(ModelDeep4().state_dict())
torch.manual_seed(42)
deep8_backup = copy.deepcopy(ModelDeep8().state_dict())

deep_configs = {
    "1 camada (B)": (ModelB, model_b_backup),
    "2 camadas": (ModelDeep2, deep2_backup),
    "4 camadas": (ModelDeep4, deep4_backup),
    "8 camadas": (ModelDeep8, deep8_backup),
}

SAMPLE_SIZES_EXTRA = [10, 50, 100, 200, 500]
results_depth = {}

for func_name, datasets, test_x, test_y in [
    ("f1", datasets_f1, test_x_f1, test_y_f1),
    ("f2", datasets_f2, test_x_f2, test_y_f2),
]:
    for depth_name, (ModelClass, backup_state) in deep_configs.items():
        for n in SAMPLE_SIZES_EXTRA:
            x_train, y_train = datasets[n]
            model = ModelClass()
            model.load_state_dict(copy.deepcopy(backup_state))
            train_loss = train_model(model, x_train, y_train)
            test_metrics = evaluate_model(model, test_x, test_y)

            results_depth[(func_name, depth_name, n)] = {
                "train_mse": train_loss,
                "test_mse": test_metrics["mse"],
                "test_mae": test_metrics["mae"],
            }
            print(f"{func_name} | {depth_name:<15} | N={n:>3d} | "
                  f"Train MSE={train_loss:.6f} | Test MSE={test_metrics['mse']:.6f}")

# Gráficos do experimento de profundidade
for func_name in ["f1", "f2"]:
    fig, ax = plt.subplots(figsize=(10, 6))
    for depth_name in deep_configs:
        test_vals = [results_depth[(func_name, depth_name, n)]["test_mse"]
                     for n in SAMPLE_SIZES_EXTRA]
        ax.plot(SAMPLE_SIZES_EXTRA, test_vals, "-o", label=depth_name, markersize=5)

    ax.axhline(y=REAL_MSE, color="red", linestyle=":", linewidth=2, label="Risco real")
    ax.set_xlabel("Número de amostras de treino")
    ax.set_ylabel("MSE Teste")
    ax.set_title(f"Impacto da Profundidade (Ativação Linear) - {func_name.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{func_name}_depth_experiment.png", dpi=150)
    plt.close(fig)
    print(f"  Salvo: {func_name}_depth_experiment.png")

# =============================================================================
# EXPERIMENTO EXTRA 2: Ativação ReLU
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENTO EXTRA 2: Impacto da mudança para ReLU")
print("=" * 70)

class ModelB_ReLU(nn.Module):
    """Modelo B com ReLU: 2 neurônios escondidos + ReLU + 1 saída."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2, 1)
    def forward(self, x):
        return self.output(self.relu(self.hidden(x)))

class ModelC_ReLU(nn.Module):
    """Modelo C com ReLU: 30 neurônios escondidos + ReLU + 1 saída."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 30)
        self.relu = nn.ReLU()
        self.output = nn.Linear(30, 1)
    def forward(self, x):
        return self.output(self.relu(self.hidden(x)))

class ModelDeep2_ReLU(nn.Module):
    """2 camadas escondidas com ReLU, 10 neurônios cada."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 10), nn.ReLU(),
            nn.Linear(10, 10), nn.ReLU(),
            nn.Linear(10, 1),
        )
    def forward(self, x):
        return self.layers(x)

torch.manual_seed(42)
model_b_relu_backup = copy.deepcopy(ModelB_ReLU().state_dict())
torch.manual_seed(42)
model_c_relu_backup = copy.deepcopy(ModelC_ReLU().state_dict())
torch.manual_seed(42)
deep2_relu_backup = copy.deepcopy(ModelDeep2_ReLU().state_dict())

relu_configs = {
    "B Linear": (ModelB, model_b_backup),
    "B ReLU": (ModelB_ReLU, model_b_relu_backup),
    "C Linear": (ModelC, model_c_backup),
    "C ReLU": (ModelC_ReLU, model_c_relu_backup),
}

results_relu = {}

for func_name, datasets, test_x, test_y in [
    ("f1", datasets_f1, test_x_f1, test_y_f1),
    ("f2", datasets_f2, test_x_f2, test_y_f2),
]:
    for relu_name, (ModelClass, backup_state) in relu_configs.items():
        for n in SAMPLE_SIZES_EXTRA:
            x_train, y_train = datasets[n]
            model = ModelClass()
            model.load_state_dict(copy.deepcopy(backup_state))
            train_loss = train_model(model, x_train, y_train)
            test_metrics = evaluate_model(model, test_x, test_y)

            results_relu[(func_name, relu_name, n)] = {
                "train_mse": train_loss,
                "test_mse": test_metrics["mse"],
                "test_mae": test_metrics["mae"],
            }
            print(f"{func_name} | {relu_name:<12} | N={n:>3d} | "
                  f"Train MSE={train_loss:.6f} | Test MSE={test_metrics['mse']:.6f} | "
                  f"Test MAE={test_metrics['mae']:.6f}")

# Gráficos do experimento ReLU
for func_name in ["f1", "f2"]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Comparação MSE Teste
    ax = axes[0]
    for relu_name in relu_configs:
        test_vals = [results_relu[(func_name, relu_name, n)]["test_mse"]
                     for n in SAMPLE_SIZES_EXTRA]
        ax.plot(SAMPLE_SIZES_EXTRA, test_vals, "-o", label=relu_name, markersize=5)
    ax.axhline(y=REAL_MSE, color="red", linestyle=":", linewidth=2, label="Risco real")
    ax.set_xlabel("Número de amostras de treino")
    ax.set_ylabel("MSE Teste")
    ax.set_title(f"Linear vs ReLU - MSE Teste ({func_name.upper()})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Comparação MAE Teste
    ax = axes[1]
    for relu_name in relu_configs:
        test_vals = [results_relu[(func_name, relu_name, n)]["test_mae"]
                     for n in SAMPLE_SIZES_EXTRA]
        ax.plot(SAMPLE_SIZES_EXTRA, test_vals, "-o", label=relu_name, markersize=5)
    ax.axhline(y=REAL_MAE, color="red", linestyle=":", linewidth=2, label="MAE real")
    ax.set_xlabel("Número de amostras de treino")
    ax.set_ylabel("MAE Teste")
    ax.set_title(f"Linear vs ReLU - MAE Teste ({func_name.upper()})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{func_name}_relu_experiment.png", dpi=150)
    plt.close(fig)
    print(f"  Salvo: {func_name}_relu_experiment.png")

# Gráfico de predições: Linear vs ReLU para f2 (onde a diferença é mais visível)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x_line = np.linspace(0, 1, 200).reshape(-1, 1).astype(np.float32)
y_true_f2 = 0.5 * x_line**2 - 0.3 * x_line + 0.8

for idx, (relu_name, (ModelClass, backup_state)) in enumerate(relu_configs.items()):
    ax = axes[idx // 2][idx % 2]
    x_train, y_train = datasets_f2[50]

    model = ModelClass()
    model.load_state_dict(copy.deepcopy(backup_state))
    train_model(model, x_train, y_train)

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(x_line)).numpy()

    ax.scatter(x_train, y_train, color="blue", s=15, label="Treino", zorder=5)
    ax.plot(x_line, y_true_f2, "g-", linewidth=2, label="Função verdadeira")
    ax.plot(x_line, y_pred, "r--", linewidth=2, label="Predição")
    ax.set_title(f"{relu_name} (N=50, f2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Comparação Linear vs ReLU - Predições em f2", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/f2_relu_predictions.png", dpi=150)
plt.close(fig)
print(f"  Salvo: f2_relu_predictions.png")

# Comparação profundidade com ReLU
print("\n--- Profundidade com ReLU ---")
deep_relu_configs = {
    "1 camada ReLU": (ModelB_ReLU, model_b_relu_backup),
    "2 camadas ReLU": (ModelDeep2_ReLU, deep2_relu_backup),
}

results_depth_relu = {}
for func_name, datasets, test_x, test_y in [
    ("f1", datasets_f1, test_x_f1, test_y_f1),
    ("f2", datasets_f2, test_x_f2, test_y_f2),
]:
    for depth_name, (ModelClass, backup_state) in deep_relu_configs.items():
        for n in SAMPLE_SIZES_EXTRA:
            x_train, y_train = datasets[n]
            model = ModelClass()
            model.load_state_dict(copy.deepcopy(backup_state))
            train_loss = train_model(model, x_train, y_train)
            test_metrics = evaluate_model(model, test_x, test_y)
            results_depth_relu[(func_name, depth_name, n)] = {
                "train_mse": train_loss,
                "test_mse": test_metrics["mse"],
                "test_mae": test_metrics["mae"],
            }
            print(f"{func_name} | {depth_name:<18} | N={n:>3d} | "
                  f"Train MSE={train_loss:.6f} | Test MSE={test_metrics['mse']:.6f}")

for func_name in ["f1", "f2"]:
    fig, ax = plt.subplots(figsize=(10, 6))
    all_depth_configs = {**deep_relu_configs}
    for depth_name in all_depth_configs:
        test_vals = [results_depth_relu[(func_name, depth_name, n)]["test_mse"]
                     for n in SAMPLE_SIZES_EXTRA]
        ax.plot(SAMPLE_SIZES_EXTRA, test_vals, "-o", label=depth_name, markersize=5)
    ax.axhline(y=REAL_MSE, color="red", linestyle=":", linewidth=2, label="Risco real")
    ax.set_xlabel("Número de amostras de treino")
    ax.set_ylabel("MSE Teste")
    ax.set_title(f"Profundidade com ReLU - {func_name.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{func_name}_depth_relu_experiment.png", dpi=150)
    plt.close(fig)
    print(f"  Salvo: {func_name}_depth_relu_experiment.png")

# =============================================================================
# Resumo final
# =============================================================================

print("\n" + "=" * 70)
print("EXECUÇÃO CONCLUÍDA COM SUCESSO!")
print(f"Todos os gráficos foram salvos em: {OUTPUT_DIR}/")
print("=" * 70)

# Listar todos os arquivos gerados
print("\nArquivos gerados:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")
